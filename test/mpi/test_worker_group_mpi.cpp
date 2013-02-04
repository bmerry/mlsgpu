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
        boost::shared_ptr<Item> outItem = outGroup.get(getTimeplotWorker(), 1);
        outItem->set(2 * item.get());
        outGroup.push(getTimeplotWorker(), outItem);
    }
};

class ProcessGroup : public WorkerGroup<Item, ProcessWorker, ProcessGroup>
{
public:
    ProcessGroup(std::size_t numWorkers, GatherGroup &outGroup)
        : WorkerGroup<Item, ProcessWorker, ProcessGroup>("ProcessGroup", numWorkers)
    {
        for (std::size_t i = 0; i < numWorkers; i++)
            addWorker(new ProcessWorker("ProcessWorker", i, outGroup));
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
        GatherGroup gatherGroup(inComm, inRoot);
        ProcessGroup processGroup(3, gatherGroup);
        // TODO RequesterScatter<Item, ProcessGroup> requester("scatter", processGroup, outComm, outRoot);
        processGroup.start();
        gatherGroup.start();
        // TODO requester();
        processGroup.stop();
        gatherGroup.stop();
    }
};

} // anonymous namespace
