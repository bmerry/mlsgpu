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
#include "../testutil.h"
#include "../../src/worker_group_mpi.h"
#include <mpi.h>

class ScatterItemTest
{
private:
    int value;

public:
    ScatterItemTest() : value(-1) {}
    explicit ScatterItemTest(int value) : value(value) {}

    void set(int value) { this->value = value; }

    void send(MPI_Comm comm, int dest);
    void recv(MPI_Comm comm, int source);
};

void ScatterItemTest::send(MPI_Comm comm, int dest)
{
    MPI_Send(&value, 1, MPI_INT, dest, MLSGPU_TAG_WORK, comm);
}

void ScatterItemTest::recv(MPI_Comm comm, int source)
{
    MPI_Recv(&value, 1, MPI_INT, source, MLSGPU_TAG_WORK, comm, MPI_STATUS_IGNORE);
}

class ScatterGroupTest : public WorkerGroupScatter<ScatterItemTest, ScatterGroupTest>
{
public:
    ScatterGroupTest(std::size_t numWorkers, std::size_t spare, MPI_Comm comm)
        : WorkerGroupScatter<ScatterItemTest, ScatterGroupTest>("ScatterGroupTest", numWorkers, spare, comm)
    {
        for (std::size_t i = 0; i < numWorkers + spare; i++)
        {
            addPoolItem(boost::make_shared<ScatterItemTest>());
        }
    }
};

/**
 * Class that receives work-items and sends to one process.
 */
class ReturnWorkerTest : public WorkerBase
{
private:
    MPI_Comm comm;
    int dest;
public:
    ReturnWorkerTest(const std::string &name, int idx, MPI_Comm comm, int dest)
        : WorkerBase(name, idx), comm(comm), dest(dest)
    {
    }

    void operator()(ScatterItemTest &item)
    {
        item.send(comm, dest);
    }

    void stop()
    {
        ScatterItemTest marker(-2);
        marker.send(comm, dest);
    }
};

class ReturnGroupTest : public WorkerGroup<ScatterItemTest, ReturnWorkerTest, ReturnGroupTest>
{
public:
    ReturnGroupTest(std::size_t numWorkers, std::size_t spare, MPI_Comm comm, int dest)
        : WorkerGroup<ScatterItemTest, ReturnWorkerTest, ReturnGroupTest>("ReturnGroupTest", numWorkers, spare)
    {
        for (std::size_t i = 0; i < numWorkers; i++)
            addWorker(new ReturnWorkerTest("ReturnWorkerTest", i, comm, dest));
        for (std::size_t i = 0; i < numWorkers + spare; i++)
            addPoolItem(boost::make_shared<ScatterItemTest>());
    }
};

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
    ReturnGroupTest returnGroup(3, 1, masterComm, root);
    RequesterScatter<ScatterItemTest, ReturnGroupTest> req("scatter", comm, returnGroup, root);
    boost::thread thread(boost::ref(req));
    returnGroup.start();

    int rank;
    int size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    if (rank == root)
    {
        Timeplot::Worker tworker("test");
        const int items = 100;
        // TODO: put this into separate thread, to avoid assumption of buffering
        ScatterGroupTest sendGroup(3, 3, comm);
        sendGroup.start();
        for (int i = 0; i < items; i++)
        {
            boost::shared_ptr<ScatterItemTest> item;
            item = sendGroup.get(tworker);
            item->set(i);
            sendGroup.push(item, tworker);
        }
        sendGroup.stop();

        for (int i = 0; i < items + size; i++)
        {
            int value = -1;
            MPI_Recv(&value, 1, MPI_INT, MPI_ANY_SOURCE, MLSGPU_TAG_WORK, masterComm, MPI_STATUS_IGNORE);
            // TODO: validate it
        }
    }

    thread.join();
    returnGroup.stop();
}
