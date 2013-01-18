/**
 * @file
 *
 * Thread pool classes for worker/slave in MPI.
 */

#ifndef WORKER_GROUP_MPI_H
#define WORKER_GROUP_MPI_H

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <mpi.h>
#include <cassert>
#include "worker_group.h"
#include "tags.h"
#include "serialize.h"

/**
 * Transmits an item by calling its @c send member. For items that do not have this
 * member, this template can be specialized.
 */
template<typename Item>
void sendItem(const Item &item, MPI_Comm comm, int dest)
{
    item.send(comm, dest);
}

/**
 * Receives an item by calling its @c recv member. For items that do not have this
 * member, this template can be specialized.
 */
template<typename Item>
void recvItem(Item &item, MPI_Comm comm, int source)
{
    item.recv(comm, source);
}

/**
 * Determines a size for an item, that is passed to @ref WorkerGroup::get on the
 * receiver to allocate storage for the item. The default implementation is to
 * call a @a size member. For items that do not have this member, this template
 * can be specialized.
 */
template<typename Item>
std::size_t sizeItem(const Item &item)
{
    return item.size();
}

/**
 * A worker that is suitable for use with @ref WorkerGroupScatter. It processes
 * items by first waiting for a request for work, then transmitting the item to
 * the requester using @ref sendItem, passing the communicator and the rank of
 * the receiver.
 */
template<typename WorkItem>
class WorkerScatter : public WorkerBase
{
public:
    /**
     * Constructor.
     * @param name    Name for the worker.
     * @param idx     Index of the worker within the parent's workers.
     * @param comm    Communicator used to communicate with requesters.
     */
    WorkerScatter(const std::string &name, int idx, MPI_Comm comm)
        : WorkerBase(name, idx), comm(comm) {}

    void operator()(WorkItem &item)
    {
        /* Wait for a receiver to be ready */
        MPI_Status status;
        {
            Timeplot::Action action("recv", getTimeplotWorker());
            MPI_Recv(NULL, 0, MPI_INT, MPI_ANY_SOURCE, MLSGPU_TAG_SCATTER_NEED_WORK, comm, &status);
        }

        {
            Timeplot::Action action("send", getTimeplotWorker());
            // Tell it there is a work item coming
            std::size_t workSize = sizeItem(item);
            assert(workSize > 0);
            MPI_Send(&workSize, 1, Serialize::mpi_type_traits<std::size_t>::type(), status.MPI_SOURCE, MLSGPU_TAG_SCATTER_HAS_WORK, comm);
            // Send it the work item
            sendItem(item, comm, status.MPI_SOURCE);
        }
    }
private:
    MPI_Comm comm;
};

/**
 * Counterpart to @ref WorkerScatter that retrieves items from another process
 * and places them into a @ref WorkerGroup for further processing. The receiver
 * is run by calling its <code>operator()</code> (it may thus be used with @c
 * boost::thread). When there is no more data to receive it will terminate,
 * although it will not stop the group it is feeding.
 *
 * The actual receiving of data is implemented by @ref recvItem.
 */
template<typename WorkItem, typename Group>
class RequesterScatter : public boost::noncopyable
{
private:
    Group &outGroup;
    MPI_Comm comm;
    const int root;
    Timeplot::Worker tworker;
public:
    RequesterScatter(const std::string &name, Group &outGroup, MPI_Comm comm, int root)
        : outGroup(outGroup), comm(comm), root(root), tworker(name) {}

    void operator()()
    {
        Statistics::Variable &sendStat = Statistics::getStatistic<Statistics::Variable>("RequesterScatter.send");
        Statistics::Variable &waitStat = Statistics::getStatistic<Statistics::Variable>("RequesterScatter.wait");
        Statistics::Variable &recvStat = Statistics::getStatistic<Statistics::Variable>("RequesterScatter.recv");
        while (true)
        {
            std::size_t workSize;
            MPI_Status status;
            {
                Timeplot::Action action("send", tworker, sendStat);
                MPI_Send(NULL, 0, MPI_INT, root, MLSGPU_TAG_SCATTER_NEED_WORK, comm);
            }
            {
                Timeplot::Action action("wait", tworker, waitStat);
                /* We will either get some work or a request to shut down */
                MPI_Recv(&workSize, 1, Serialize::mpi_type_traits<std::size_t>::type(), MPI_ANY_SOURCE, MLSGPU_TAG_SCATTER_HAS_WORK, comm, &status);
            }
            if (workSize == 0)
                break;
            else
            {
                boost::shared_ptr<WorkItem> item = outGroup.get(tworker, workSize);
                {
                    Timeplot::Action action("recv", tworker, recvStat);
                    recvItem(*item, comm, status.MPI_SOURCE);
                }
                outGroup.push(item);
            }
        }
    }
};

/**
 * Worker group that handles distribution of work items to other MPI nodes.
 * Each @c WorkerGroupScatter is associated with multiple @ref RequesterScatter
 * instances on other nodes.
 *
 * When @ref stop is called, messages are sent to the requesters to shut them
 * down.
 */
template<typename WorkItem, typename Derived>
class WorkerGroupScatter : public WorkerGroup<WorkItem, WorkerScatter<WorkItem>, Derived>
{
private:
    MPI_Comm comm;
    const std::size_t requesters;
protected:
    /**
     * Constructor. The derived class must still generate the pool items, but it is
     * not responsible for the workers.
     *
     * @param name           Name for the threads in the pool.
     * @param numWorkers     Number of worker threads to use.
     * @param requesters     Number of requesters which need to be shut down at the end.
     * @param comm           Communicator to use. The remote group must have an associated requester per member.
     */
    WorkerGroupScatter(const std::string &name,
                       std::size_t numWorkers,
                       std::size_t requesters,
                       MPI_Comm comm)
        : WorkerGroup<WorkItem, WorkerScatter<WorkItem>, Derived>(name, numWorkers),
        comm(comm), requesters(requesters)
    {
        for (std::size_t i = 0; i < numWorkers; i++)
            this->addWorker(new WorkerScatter<WorkItem>(name + ".worker", i, comm));
    }

public:
    /**
     * Overrides the base class to send a shutdown message to all the requesters.
     */
    void stopPostJoin()
    {
        /* Shut down the receivers */
        for (std::size_t i = 0; i < requesters; i++)
        {
            MPI_Status status;
            std::size_t workSize = 0;
            MPI_Recv(NULL, 0, MPI_INT, MPI_ANY_SOURCE, MLSGPU_TAG_SCATTER_NEED_WORK, comm, &status);
            MPI_Send(&workSize, 1, Serialize::mpi_type_traits<std::size_t>::type(), status.MPI_SOURCE, MLSGPU_TAG_SCATTER_HAS_WORK, comm);
        }
    }
};


/**
 * A worker that is suitable for use with @ref WorkerGroupGather. When it pulls
 * an item from the queue, it first informs the remote that it has some work,
 * then sends it. When the queue is drained, it instead tells the remote to
 * shut down.
 */
template<typename WorkItem>
class WorkerGather : public WorkerBase
{
private:
    MPI_Comm comm;
    int root;
public:
    /**
     * Constructor.
     *
     * @param name      Name for the worker.
     * @param comm      Communicator to communicate with the remote end.
     * @param root      Target for messages.
     */
    WorkerGather(const std::string &name, MPI_Comm comm, int root)
        : WorkerBase(name, 0), comm(comm), root(root)
    {
    }

    void operator()(WorkItem &item)
    {
        Timeplot::Action action("send", getTimeplotWorker());
        std::size_t workSize = sizeItem(item);
        MPI_Send(&workSize, 1, Serialize::mpi_type_traits<std::size_t>::type(), root,
                 MLSGPU_TAG_GATHER_HAS_WORK, comm);
        sendItem(item, comm, root);
    }

    void stop()
    {
        std::size_t workSize = 0;
        MPI_Send(&workSize, 1, Serialize::mpi_type_traits<std::size_t>::type(), root,
                 MLSGPU_TAG_GATHER_HAS_WORK, comm);
    }
};

/**
 * Counterpart to @ref WorkerGather that receives the messages and
 * places them into a @ref WorkerGroup. The receiver is run by calling its
 * <code>operator()</code> (it may thus be used with @c boost::thread). When
 * there is no more data to receive it will terminate, although it will not
 * stop the group it is feeding.
 */
template<typename WorkItem, typename Group>
class ReceiverGather : public boost::noncopyable
{
private:
    Group &outGroup;
    const MPI_Comm comm;
    const std::size_t senders;
    Timeplot::Worker tworker;

public:
    ReceiverGather(const std::string &name, Group &outGroup, MPI_Comm comm, std::size_t senders)
        : outGroup(outGroup), comm(comm), senders(senders), tworker(name)
    {
    }

    void operator()()
    {
        std::size_t rem = senders;
        Statistics::Variable &waitStat = Statistics::getStatistic<Statistics::Variable>("ReceiverGather.wait");
        Statistics::Variable &recvStat = Statistics::getStatistic<Statistics::Variable>("ReceiverGather.recv");
        while (rem > 0)
        {
            std::size_t workSize;
            MPI_Status status;
            {
                Timeplot::Action action("wait", tworker, waitStat);
                MPI_Recv(&workSize, 1, Serialize::mpi_type_traits<std::size_t>::type(),
                         MPI_ANY_SOURCE, MLSGPU_TAG_GATHER_HAS_WORK, comm, &status);
            }
            if (workSize == 0)
                rem--;
            else
            {
                boost::shared_ptr<WorkItem> item = outGroup.get(tworker, workSize);
                {
                    Timeplot::Action action("recv", tworker, recvStat);
                    recvItem(*item, comm, status.MPI_SOURCE);
                }
                outGroup.push(item);
            }
        }
    }
};

/**
 * Worker group that handles sending items from a queue to a @ref
 * ReceiverGather running on another MPI process.
 */
template<typename WorkItem, typename Derived>
class WorkerGroupGather : public WorkerGroup<WorkItem, WorkerGather<WorkItem>, Derived>
{
protected:
    /**
     * Constructor. This takes care of constructing the (single) worker.
     *
     * @param name      Name for the group (also for the worker).
     * @param comm      Communicator to send the items.
     * @param root      Destination for the items within @a comm.
     */
    WorkerGroupGather(const std::string &name, MPI_Comm comm, int root)
        : WorkerGroup<WorkItem, WorkerGather<WorkItem>, Derived>(name, 1)
    {
        this->addWorker(new WorkerGather<WorkItem>(name, comm, root));
    }
};

#endif /* WORKER_GROUP_MPI_H */
