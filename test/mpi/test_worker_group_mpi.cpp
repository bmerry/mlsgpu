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

    int get() const { return value; }
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

class GatherGroup : public WorkerGroupGather<Item, GatherGroup>
{
public:
    GatherGroup(std::size_t spare, MPI_Comm comm, int root)
        : WorkerGroupGather<Item, GatherGroup>("GatherGroup", spare, comm, root)
    {
        for (std::size_t i = 0; i < 1 + spare; i++)
        {
            addPoolItem(boost::make_shared<Item>());
        }
    }
};

/**
 * Class that receives work-items, doubles them and sends on the result.
 */
class ProcessWorker : public WorkerBase
{
private:
    GatherGroup &outGroup;
public:
    ProcessWorker(const std::string &name, int idx, GatherGroup &outGroup)
        : WorkerBase(name, idx), outGroup(outGroup)
    {
    }

    void operator()(Item &item)
    {
        boost::shared_ptr<Item> outItem = outGroup.get(getTimeplotWorker());
        outItem->set(2 * item.get());
        outGroup.push(outItem, getTimeplotWorker());
    }
};

class ProcessGroup : public WorkerGroup<Item, ProcessWorker, ProcessGroup>
{
public:
    ProcessGroup(std::size_t numWorkers, std::size_t spare, GatherGroup &outGroup)
        : WorkerGroup<Item, ProcessWorker, ProcessGroup>("ProcessGroup", numWorkers, spare)
    {
        for (std::size_t i = 0; i < numWorkers; i++)
            addWorker(new ProcessWorker("ProcessWorker", i, outGroup));
        for (std::size_t i = 0; i < numWorkers + spare; i++)
            addPoolItem(boost::make_shared<Item>());
    }
};

class ConsumerWorker : public WorkerBase
{
private:
    std::vector<int> &values;

public:
    explicit ConsumerWorker(std::vector<int> &values)
        : WorkerBase("ConsumerWorker", 0), values(values) {}

    void operator()(Item &item)
    {
        values.push_back(item.get());
    }
};

class ConsumerGroup : public WorkerGroup<Item, ConsumerWorker, ConsumerGroup>
{
public:
    ConsumerGroup(std::size_t spare, std::vector<int> &values)
        : WorkerGroup<Item, ConsumerWorker, ConsumerGroup>("ConsumerGroup", 1, spare)
    {
        for (std::size_t i = 0; i < 1 + spare; i++)
            addPoolItem(boost::make_shared<Item>());
        addWorker(new ConsumerWorker(values));
    }
};

/**
 * Thread class for doing the work on each slave node. This is split into a thread
 * so that it can also be used on the master (which is actually both master and
 * a slave), in parallel to the master work.
 */
class Slave
{
private:
    MPI_Comm outComm;
    int outRoot;
    MPI_Comm inComm;
    int inRoot;

public:
    Slave(MPI_Comm outComm, int outRoot, MPI_Comm inComm, int inRoot)
        : outComm(outComm), outRoot(outRoot), inComm(inComm), inRoot(inRoot)
    {
    }

    void operator()() const
    {
        GatherGroup gatherGroup(2, inComm, inRoot);
        ProcessGroup processGroup(3, 1, gatherGroup);
        RequesterScatter<Item, ProcessGroup> requester("scatter", processGroup, outComm, outRoot);
        processGroup.start();
        gatherGroup.start();
        requester();
        processGroup.stop();
        gatherGroup.stop();
    }
};

} // anonymous namespace

class TestWorkerGroupScatter : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestWorkerGroupScatter);
    CPPUNIT_TEST(testIntracomm);
    CPPUNIT_TEST_SUITE_END();
private:
    MPI_Comm outComm;       ///< For scattering items to processors
    MPI_Comm inComm;        ///< For gathering results

    void testIntracomm();   ///< Test distribution with an intracommunicator

public:
    virtual void setUp();
    virtual void tearDown();
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestWorkerGroupScatter, TestSet::perBuild());

void TestWorkerGroupScatter::setUp()
{
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Comm_dup(MPI_COMM_WORLD, &outComm);
    MPI_Comm_dup(MPI_COMM_WORLD, &inComm);
}

void TestWorkerGroupScatter::tearDown()
{
    if (outComm != MPI_COMM_NULL)
        MPI_Comm_free(&outComm);
    if (inComm != MPI_COMM_NULL)
        MPI_Comm_free(&inComm);
    MPI_Barrier(MPI_COMM_WORLD);
}

void TestWorkerGroupScatter::testIntracomm()
{
    const std::size_t items = 1000;
    const int root = 0;
    int rank;
    bool pass = true;

    boost::thread slaveThread(Slave(outComm, root, inComm, root));
    MPI_Comm_rank(outComm, &rank);
    if (rank == root)
    {
        std::vector<int> values;
        int size;
        MPI_Comm_size(outComm, &size);

        ConsumerGroup consumer(2, values);
        ReceiverGather<Item, ConsumerGroup> receiver("ReceiverGather", consumer, inComm, size);
        ScatterGroup sendGroup(3, 3, size, outComm);

        sendGroup.start();
        boost::thread receiverThread(boost::ref(receiver));
        consumer.start();

        Timeplot::Worker tworker("producer");
        for (std::size_t i = 0; i < items; i++)
        {
            boost::shared_ptr<Item> item;
            item = sendGroup.get(tworker);
            item->set(i);
            sendGroup.push(item, tworker);
        }

        sendGroup.stop();
        receiverThread.join();
        consumer.stop();

        if (values.size() != items)
            pass = false;
        else
        {
            std::sort(values.begin(), values.end());
            for (std::size_t i = 0; i < items; i++)
                if (values[i] != 2 * (int) i)
                    pass = false;
        }
    }
    slaveThread.join();

    CPPUNIT_ASSERT(pass);
}
