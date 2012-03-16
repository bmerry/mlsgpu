/**
 * @file
 *
 * Collection of classes for doing specific steps from the main program.
 */

#ifndef WORKERS_H
#define WORKERS_H

#if HAVE_CONFIG_H
# include <config.h>
#endif

#include <boost/smart_ptr/shared_ptr.hpp>
#include <boost/smart_ptr/scoped_ptr.hpp>
#include <boost/thread/thread.hpp>
#include <boost/noncopyable.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <cstddef>
#include <stdexcept>
#include "splat_tree_cl.h"
#include "clip.h"
#include "marching.h"
#include "mls.h"
#include "mesh.h"
#include "mesh_filter.h"
#include "grid.h"
#include "progress.h"
#include "work_queue.h"
#include "bucket.h"
#include "splat.h"
#include "splat_set.h"
#include "clh.h"
#include "errors.h"
#include "statistics.h"

/**
 * A collection of threads operating on work-items, fed by a queue.
 *
 * @param WorkItem     A POD type describing an item of work.
 * @param Worker       Function object class that is called to process elements.
 *
 * The @a Worker class must have an @c operator() that accepts a reference to a
 * @a WorkItem. The operator does not need to be @c const.  The worker class
 * does not need to be copyable or default-constructable and may contain
 * significant state.
 *
 * The workitems may also be large objects, and the design is based on
 * recycling a fixed pool rather than having the caller construct them. Users
 * of the class will call @ref get to retrieve an item from the pool, populate
 * it, then call @ref push to enqueue the item for processing.
 *
 * At construction, the worker group is given both a number of threads and a
 * capacity. The capacity determines the number of workitems that will be
 * live. If capacity equals the number of workers, then it will only be
 * possible to populate a new work item while one of the workers is idle. If
 * capacity exceeds the number of threads, then it will be possible to
 * populate spare work items while all worker threads are busy.
 *
 * The @ref start and @ref stop functions are not thread-safe: they should
 * only be called by a single manager thread. The other functions are
 * thread-safe, allowing for multiple producers.
 */
template<typename WorkItem, typename Worker>
class WorkerGroup : public boost::noncopyable
{
public:
    /**
     * Retrieve an unused item from the pool to be populated with work. This
     * may block if all the items are currently in use.
     *
     * @warning The returned item @em must be enqueued with @ref push, even
     * if it turns out there is nothing to do. Failure to do so will lead
     * to the item being lost to the pool, and possibly deadlock.
     */
    boost::shared_ptr<WorkItem> get()
    {
        Statistics::Timer timer(getStat);
        return itemPool.pop();
    }

    /**
     * Enqueue an item of work.
     *
     * @pre @a item was obtained by @ref get.
     */
    void push(boost::shared_ptr<WorkItem> item)
    {
        Statistics::Timer timer(pushStat);
        workQueue.push(item);
    }

    /**
     * Start the worker threads running. It is not required to do this
     * before calling @ref get or @ref push, but they may block until another
     * thread calls @ref start.
     *
     * @pre The worker threads are not already running.
     */
    void start()
    {
        MLSGPU_ASSERT(threads.empty(), std::runtime_error);
        threads.reserve(workers.size());
        for (std::size_t i = 0; i < workers.size(); i++)
        {
            threads.push_back(new boost::thread(Thread(*this, workers[i])));
        }
    }

    /**
     * Shut down the worker threads.
     *
     * @warning This method is not thread-safe relative to other calls such
     * as @ref push. All producers must be shut down while this method is
     * called.
     *
     * @pre The worker threads are currently running.
     */
    void stop()
    {
        MLSGPU_ASSERT(threads.size() == workers.size(), std::runtime_error);
        for (std::size_t i = 0; i < threads.size(); i++)
            workQueue.push(boost::shared_ptr<WorkItem>());
        for (std::size_t i = 0; i < threads.size(); i++)
            threads[i].join();
        threads.clear();
    }

    /// Returns the number of workers.
    std::size_t numWorkers() const
    {
        return workers.size();
    }

protected:
    /**
     * Register a worker during construction.
     *
     * @see @ref WorkerGroup::WorkerGroup.
     */
    void addWorker(Worker *worker)
    {
        workers.push_back(worker);
    }

    /**
     * Register a work item during construction.
     *
     * @see @ref WorkerGroup::WorkerGroup.
     */
    void addPoolItem(boost::shared_ptr<WorkItem> item)
    {
        itemPool.push(item);
    }

    /// Retrieve a reference to a worker.
    Worker &getWorker(std::size_t index)
    {
        return workers.at(index);
    }

    /**
     * Constructor. The derived class must change to this, and then
     * make exactly @a numWorkers calls to @ref addWorker and @a capacity
     * calls to @ref addPoolItem to provide the constructed workers and
     * work items.
     *
     * @param numWorkers     Number of worker threads to use.
     * @param capacity       Number of work items to have in the pool.
     * @param pushStat       Statistic for time blocked in @ref push.
     * @param popStat        Statistic for time blocked in @ref WorkQueue::pop.
     * @param getStat        Statistic for time blocked in @ref get.
     *
     * @pre @a numWorkers &gt; 0.
     * @pre @a capacity &gt;= @a numWorkers.
     */
    WorkerGroup(std::size_t numWorkers, std::size_t capacity,
                Statistics::Variable &pushStat,
                Statistics::Variable &popStat,
                Statistics::Variable &getStat)
        : workQueue(capacity), itemPool(capacity),
        pushStat(pushStat), popStat(popStat), getStat(getStat)
    {
        MLSGPU_ASSERT(numWorkers > 0, std::invalid_argument);
        MLSGPU_ASSERT(capacity >= numWorkers, std::invalid_argument);
        workers.reserve(numWorkers);
    }

private:
    /// Thread object that processes items from the queue.
    class Thread
    {
        WorkerGroup<WorkItem, Worker> &owner;
        Worker &worker;

    public:
        Thread(WorkerGroup<WorkItem, Worker> &owner, Worker &worker)
            : owner(owner), worker(worker) {}

        void operator()()
        {
            while (true)
            {
                Timer timer;
                boost::shared_ptr<WorkItem> item = owner.workQueue.pop();
                if (!item.get())
                    break; // we have been asked to shut down
                owner.popStat.add(timer.getElapsed());
                worker(*item);
                owner.itemPool.push(item);
            }
        }
    };

    WorkQueue<boost::shared_ptr<WorkItem> > workQueue;
    Pool<WorkItem> itemPool;

    /**
     * Threads. This is empty when no threads are running and contains the
     * thread objects when it is running.
     */
    boost::ptr_vector<boost::thread> threads;

    /**
     * Workers. This is populated during construction (by @ref addWorker) and
     * persists until object destruction.
     */
    boost::ptr_vector<Worker> workers;

    Statistics::Variable &pushStat;
    Statistics::Variable &popStat;
    Statistics::Variable &getStat;
};

class DeviceWorkerGroup;

class DeviceWorkerGroupBase
{
public:
    /**
     * Data about a fine-grained bucket. A shared pointer to this is obtained
     * from @ref DeviceWorkerGroup::get and enqueued with @ref
     * DeviceWorkerGroup::push.
     */
    struct WorkItem
    {
        cl::Buffer splats;
        std::size_t numSplats;
        Grid grid;
        Bucket::Recursion recursionState;
    };

    class Worker : public boost::noncopyable
    {
    private:
        DeviceWorkerGroup &owner;

        const cl::CommandQueue queue;
        SplatTreeCL tree;
        MlsFunctor input;
        Marching marching;
        boost::scoped_ptr<Clip> clip;
        ScaleBiasFilter scaleBias;
        MeshFilterChain filterChain;

    public:
        typedef void result_type;

        Worker(
            DeviceWorkerGroup &owner,
            const cl::Context &context, const cl::Device &device,
            int levels, bool keepBoundary, float boundaryLimit);

        void setOutput(const Marching::OutputFunctor &output)
        {
            filterChain.setOutput(output);
        }

        void operator()(WorkItem &work);
    };
};

/**
 * Does the actual OpenCL calls necessary to compute the mesh and write
 * it to the @ref MesherBase class. It pulls chunks of work off a queue,
 * which contains pre-bucketed splats.
 */
class DeviceWorkerGroup : protected DeviceWorkerGroupBase, public WorkerGroup<DeviceWorkerGroupBase::WorkItem, DeviceWorkerGroupBase::Worker>
{
private:
    typedef WorkerGroup<DeviceWorkerGroupBase::WorkItem, DeviceWorkerGroupBase::Worker> Base;
    ProgressDisplay *progress;

    const Grid fullGrid;
    const std::size_t maxSplats;
    const Grid::size_type maxCells;
    const int subsampling;

    friend class DeviceWorkerGroupBase::Worker;

public:
    typedef DeviceWorkerGroupBase::WorkItem WorkItem;

    /**
     * Constructor.
     *
     * @param numWorkers         Number of worker threads to use (each with a separate OpenCL queue and state)
     * @param capacity           Number of workitems to use.
     * @param fullGrid           The overall bounding box grid.
     * @param context, device    OpenCL context and device to run on.
     * @param maxSplats          Space to allocate for holding splats.
     * @param maxCells           Space to allocate for the octree.
     * @param levels             Space to allocate for the octree.
     * @param subsampling        Octree subsampling level.
     * @param keepBoundary       If true, skips boundary clipping.
     * @param boundaryLimit      Tuning factor for boundary clipping.
     */
    DeviceWorkerGroup(
        std::size_t numWorkers, std::size_t capacity,
        const Grid &fullGrid,
        const cl::Context &context, const cl::Device &device,
        std::size_t maxSplats, Grid::size_type maxCells,
        int levels, int subsampling, bool keepBoundary, float boundaryLimit);

    /// Returns total resources that would be used by all workers and workitems
    static CLH::ResourceUsage resourceUsage(
        std::size_t numWorkers, std::size_t capacity,
        const cl::Device &device,
        std::size_t maxSplats, Grid::size_type maxCells,
        int levels, bool keepBoundary);

    /**
     * Sets a progress display that will be updated by the number of cells
     * processed.
     */
    void setProgress(ProgressDisplay *progress) { this->progress = progress; }

    /**
     * Sets the output functor to call with results. This must be called
     * before starting the threads.
     */
    void setOutput(const Marching::OutputFunctor &output);
};

class FineBucketGroup;

class FineBucketGroupBase
{
public:
    struct WorkItem
    {
        SplatSet::VectorSet splats;
        Grid grid;
        Bucket::Recursion recursionState;
    };

    class Worker
    {
    private:
        FineBucketGroup &owner;
        const cl::CommandQueue queue; ///< Queue for map and unmap operations

    public:
        typedef void result_type;

        Worker(FineBucketGroup &owner, const cl::Context &context, const cl::Device &device);

        /// Bucketing callback for blocks sized for device execution.
        void operator()(
            const SplatSet::Traits<SplatSet::VectorSet>::subset_type &splats,
            const Grid &grid,
            const Bucket::Recursion &recursionState);

        /// Front-end processing of one item
        void operator()(WorkItem &work);
    };
};

/**
 * A worker object that handles coarse-to-fine bucketing. It pulls work from
 * an internal queue (containing regions of splats already read from storage),
 * calls @ref Bucket::bucket to subdivide the splats into buckets suitable for
 * device execution, and passes them on to a @ref DeviceWorkerGroup.
 */
class FineBucketGroup : protected FineBucketGroupBase, public WorkerGroup<FineBucketGroupBase::WorkItem, FineBucketGroupBase::Worker>
{
public:
    typedef FineBucketGroupBase::WorkItem WorkItem;

    void setProgress(ProgressDisplay *progress) { this->progress = progress; }

    FineBucketGroup(
        std::size_t numWorkers, std::size_t capacity,
        DeviceWorkerGroup &outGroup,
        const Grid &fullGrid,
        const cl::Context &context, const cl::Device &device,
        std::size_t maxSplats,
        Grid::size_type maxCells,
        std::size_t maxSplit);

private:
    DeviceWorkerGroup &outGroup;

    const Grid fullGrid;
    std::size_t maxSplats;
    Grid::size_type maxCells;
    std::size_t maxSplit;
    ProgressDisplay *progress;

    friend class FineBucketGroupBase::Worker;
};

#endif /* !WORKERS_H */
