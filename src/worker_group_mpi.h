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
#include "worker_group.h"

enum
{
    MLSGPU_TAG_NEED_WORK = 0,
    MLSGPU_TAG_STOP = 1
};

/**
 * A worker that is suitable for use with @ref WorkerGroupScatter. It processes
 * items by first waiting for a request for work, then transmitting the item to
 * the requester by using its @c send method, passing the communicator and the
 * rank of the receiver.
 */
template<typename WorkItem>
class WorkerScatter : public WorkerBase
{
public:
    /**
     * Constructor.
     * @param comm    Communicator used to communicate with requesters.
     */
    explicit WorkerScatter(MPI_Comm comm) : comm(comm) {}

    void operator()(WorkItem &item)
    {
        /* Wait for a receiver to be ready */
        MPI_Status status;
        MPI_Recv(NULL, 0, MPI_INT, MPI_ANY_SOURCE, MLSGPU_TAG_NEED_WORK, comm, &status);

        // Send it the work item
        item.send(comm, status.MPI_SOURCE);
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
 * The actual receiving of data is implemented by a @c recv() method in
 * the item class, which takes the communicator and the source.
 */
template<typename WorkItem, typename Group>
class RequesterScatter
{
private:
    MPI_Comm comm;
    Group &outGroup;
    int root;
public:
    RequesterScatter(MPI_Comm comm, Group &outGroup, int root)
        : comm(comm), outGroup(outGroup), root(root) {}

    void operator()() const
    {
        while (true)
        {
            MPI_Status status;
            MPI_Send(NULL, 0, MPI_INT, root, MLSGPU_TAG_NEED_WORK, comm);
            /* We will either get some work or a request to shut down */
            MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &status);
            if (status.MPI_TAG == MLSGPU_TAG_STOP)
            {
                MPI_Recv(NULL, 0, MPI_INT, status.MPI_SOURCE, status.MPI_TAG, comm, MPI_STATUS_IGNORE);
                break;
            }
            else
            {
                boost::shared_ptr<WorkItem> item = outGroup.get();
                item->recv(comm, status.MPI_SOURCE);
                outGroup.push(item);
            }
        }
    }
};

/**
 * Worker group that handles distribution of work items to other MPI nodes.
 * Each @c WorkerGroupScatter is associated with multiple @ref ReceiverScatter
 * instances on other nodes. There must be one requester for each member of the
 * (remote) group of the provided communicator. Thus, when using an
 * intracommunicator there must also be a requester for oneself.
 *
 * When @ref stop is called, messages are sent to the requesters to shut them
 * down.
 */
template<typename WorkItem, typename Derived>
class WorkerGroupScatter : public WorkerGroup<WorkItem, WorkerScatter<WorkItem>, Derived>
{
private:
    MPI_Comm comm;
public:
    /**
     * Constructor. The derived class must still generate the pool items, but it is
     * not responsible for the workers.
     *
     * @param name           Name for the threads in the pool.
     * @param numWorkers     Number of worker threads to use.
     * @param spare          Number of work items to have available in the pool when all workers are busy.
     * @param comm           Communicator to use. The remote group must have an associated requester per member.
     */
    WorkerGroupScatter(const std::string &name,
                       std::size_t numWorkers, std::size_t spare,
                       MPI_Comm comm)
        : WorkerGroup<WorkItem, WorkerScatter<WorkItem>, Derived>(name, numWorkers, spare),
        comm(comm)
    {
        for (std::size_t i = 0; i < numWorkers; i++)
            addWorker(new WorkerScatter<WorkItem>(comm));
    }

    /**
     * Overrides the base class to send a shutdown message to all the requesters.
     */
    void stopImpl()
    {
        int size;
        int isInter;
        WorkerGroup<WorkItem, WorkerScatter<WorkItem>, Derived>::stopImpl();

        MPI_Comm_test_inter(comm, &isInter);
        if (isInter)
            MPI_Comm_remote_size(comm, &size);
        else
            MPI_Comm_size(comm, &size);
        /* Shut down the receivers */
        for (int i = 0; i < size; i++)
            MPI_Send(NULL, 0, MPI_INT, i, MLSGPU_TAG_STOP, comm);
    }
};

#endif /* WORKER_GROUP_MPI_H */
