/**
 * @file
 *
 * Tests for MPI helper classes.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif

#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/extensions/HelperMacros.h>
#include <boost/thread.hpp>
#include <boost/smart_ptr/shared_ptr.hpp>
#include <boost/smart_ptr/make_shared.hpp>
#include <vector>
#include "../testutil.h"
#include "../../src/worker_group_mpi.h"
#include <mpi.h>

namespace
{

class Item
{
private:
    int value;

public:
    Item() : value(-1) {}
    explicit Item(int value) : value(value) {}

    void set(int value) { this->value = value; }

    void send(MPI_Comm comm, int dest);
    void recv(MPI_Comm comm, int source);
};

void Item::send(MPI_Comm comm, int dest)
{
    MPI_Send(&value, 1, MPI_INT, dest, MLSGPU_TAG_WORK, comm);
}

void Item::recv(MPI_Comm comm, int source)
{
    MPI_Recv(&value, 1, MPI_INT, source, MLSGPU_TAG_WORK, comm, MPI_STATUS_IGNORE);
}

class ScatterGroup : public WorkerGroupScatter<Item, ScatterGroup>
{
public:
    ScatterGroup(std::size_t numWorkers, std::size_t spare, std::size_t requesters, MPI_Comm comm)
        : WorkerGroupScatter<Item, ScatterGroup>("ScatterGroup", numWorkers, spare, requesters, comm)
    {
        for (std::size_t i = 0; i < numWorkers + spare; i++)
        {
            addPoolItem(boost::make_shared<Item>());
        }
    }
};

/**
 * Class that receives work-items and sends to one process.
 */
class ReturnWorker : public WorkerBase
{
private:
    MPI_Comm comm;
    int dest;
public:
    ReturnWorker(const std::string &name, int idx, MPI_Comm comm, int dest)
        : WorkerBase(name, idx), comm(comm), dest(dest)
    {
    }

    void operator()(Item &item)
    {
        item.send(comm, dest);
    }

    void stop()
    {
        Item marker(-2);
        marker.send(comm, dest);
    }
};

class ReturnGroup : public WorkerGroup<Item, ReturnWorker, ReturnGroup>
{
public:
    ReturnGroup(std::size_t numWorkers, std::size_t spare, MPI_Comm comm, int dest)
        : WorkerGroup<Item, ReturnWorker, ReturnGroup>("ReturnGroup", numWorkers, spare)
    {
        for (std::size_t i = 0; i < numWorkers; i++)
            addWorker(new ReturnWorker("ReturnWorker", i, comm, dest));
        for (std::size_t i = 0; i < numWorkers + spare; i++)
            addPoolItem(boost::make_shared<Item>());
    }
};

class ScatterProducer
{
private:
    MPI_Comm comm;
    std::size_t requesters;
    int items;

public:
    ScatterProducer(MPI_Comm comm, std::size_t requesters, int items)
        : comm(comm), requesters(requesters), items(items)
    {
    }

    void operator()() const
    {
        Timeplot::Worker tworker("test");
        ScatterGroup sendGroup(3, 3, requesters, comm);
        sendGroup.start();
        for (int i = 0; i < items; i++)
        {
            boost::shared_ptr<Item> item;
            item = sendGroup.get(tworker);
            item->set(i);
            sendGroup.push(item, tworker);
        }
        sendGroup.stop();
    }
};

} // anonymous namespace

class TestWorkerGroupScatter : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestWorkerGroupScatter);
    CPPUNIT_TEST(testIntracomm);
    CPPUNIT_TEST_SUITE_END();
private:
    MPI_Comm comm;          ///< Set up by setUp or by tests, removed by teardown
    MPI_Comm masterComm;    ///< For passing results back to the master

    void testIntracomm();   ///< Test distribution with an intracommunicator

public:
    virtual void setUp();
    virtual void tearDown();
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestWorkerGroupScatter, TestSet::perBuild());

void TestWorkerGroupScatter::setUp()
{
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Comm_dup(MPI_COMM_WORLD, &comm);
    // TODO: can simplify it to a point-to-point intercomm
    MPI_Comm_dup(MPI_COMM_WORLD, &masterComm);
}

void TestWorkerGroupScatter::tearDown()
{
    if (comm != MPI_COMM_NULL)
        MPI_Comm_free(&comm);
    if (masterComm != MPI_COMM_NULL)
        MPI_Comm_free(&masterComm);
    MPI_Barrier(MPI_COMM_WORLD);
}

void TestWorkerGroupScatter::testIntracomm()
{
    const int root = 0;
    ReturnGroup returnGroup(3, 1, masterComm, root);
    RequesterScatter<Item, ReturnGroup> req("scatter", returnGroup, comm, root);
    boost::thread thread(boost::ref(req));
    returnGroup.start();

    int rank;
    int size;
    bool pass = true;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    if (rank == root)
    {
        const int items = 100;
        boost::thread producer(ScatterProducer(comm, size, items));
        std::vector<bool> seen(items, false);
        int shutdowns = 0;

        for (int i = 0; i < items + size; i++)
        {
            int value = -1;
            MPI_Recv(&value, 1, MPI_INT, MPI_ANY_SOURCE, MLSGPU_TAG_WORK, masterComm, MPI_STATUS_IGNORE);
            if (value == -2)
            {
                shutdowns++;
            }
            else
            {
                if (value < 0 || value >= items || seen[value])
                    pass = false;
                seen[value] = true;
            }
        }
        producer.join();
    }

    thread.join();
    returnGroup.stop();

    CPPUNIT_ASSERT(pass);
}
