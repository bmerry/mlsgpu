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

    void send(MPI_Comm comm, int dest) const;
    void recv(MPI_Comm comm, int source);
    std::size_t size() const;
};

void Item::send(MPI_Comm comm, int dest) const
{
    MPI_Send(const_cast<int *>(&value), 1, MPI_INT, dest, MLSGPU_TAG_WORK, comm);
}

void Item::recv(MPI_Comm comm, int source)
{
    MPI_Recv(&value, 1, MPI_INT, source, MLSGPU_TAG_WORK, comm, MPI_STATUS_IGNORE);
}

std::size_t Item::size() const
{
    return 1;
}

class GatherGroup : public WorkerGroupGather<Item, GatherGroup>
{
public:
    GatherGroup(MPI_Comm comm, int root)
        : WorkerGroupGather<Item, GatherGroup>("GatherGroup", comm, root)
    {
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
    ConsumerGroup(std::vector<int> &values)
        : WorkerGroup<Item, ConsumerWorker, ConsumerGroup>("ConsumerGroup", 1)
    {
        addWorker(new ConsumerWorker(values));
    }
};

/**
 * Thread class for doing the work on each slave node. This is split into a thread
 * so that it can also be used on the master (which is actually both master and
 * a slave), in parallel to the master work.
 *
 * It sends integer items back to the master. Each slave sends those @a x for which
 * <code>x % size == rank</code>.
 */
class Slave
{
private:
    MPI_Comm comm;
    int root;
    std::size_t items;   ///< Total items to send across all slaves

public:
    Slave(MPI_Comm comm, int root, std::size_t items)
        : comm(comm), root(root), items(items)
    {
    }

    /// Thread callback
    void operator()() const
    {
        Timeplot::Worker tworker("slave");

        GatherGroup gatherGroup(comm, root);
        gatherGroup.start();

        int rank, size;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);
        for (std::size_t i = rank; i < items; i += size)
        {
            boost::shared_ptr<Item> item = gatherGroup.get(tworker, 1);
            item->set(i);
            gatherGroup.push(tworker, item);
        }

        gatherGroup.stop();
    }
};

} // anonymous namespace

class TestWorkerGroupGather : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestWorkerGroupGather);
    CPPUNIT_TEST(testStress);
    CPPUNIT_TEST_SUITE_END();

private:
    MPI_Comm comm;
    void testStress();    ///< Basic test with lots of items

public:
    virtual void setUp();
    virtual void tearDown();
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestWorkerGroupGather, TestSet::perBuild());

void TestWorkerGroupGather::setUp()
{
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Comm_dup(MPI_COMM_WORLD, &comm);
}

void TestWorkerGroupGather::tearDown()
{
    if (comm != MPI_COMM_NULL)
        MPI_Comm_free(&comm);
    MPI_Barrier(MPI_COMM_WORLD);
}

void TestWorkerGroupGather::testStress()
{
    const std::size_t items = 100000;
    const int root = 0;
    boost::thread slaveThread(Slave(comm, root, items));

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    std::vector<int> out;
    if (rank == 0)
    {

        ConsumerGroup consumer(out);
        ReceiverGather<Item, ConsumerGroup> receiver("ReceiverGather", consumer, comm, size);

        consumer.start();
        receiver();
        consumer.stop();
    }
    slaveThread.join();

    if (rank == 0)
    {
        MLSGPU_ASSERT_EQUAL(items, out.size());
        std::sort(out.begin(), out.end());
        int failed = 0;
        for (std::size_t i = 0; i < items; i++)
            if (out[i] != (int) i)
                failed++;

        CPPUNIT_ASSERT_EQUAL(0, failed);
    }
}
