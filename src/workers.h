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
#include "collection.h"
#include "splat.h"
#include "splat_set.h"
#include "clh.h"
#include "errors.h"

template<typename WorkItem, typename Worker>
class WorkerGroup : public boost::noncopyable
{
public:
    boost::shared_ptr<WorkItem> get()
    {
        return itemPool.pop();
    }

    void push(boost::shared_ptr<WorkItem> item)
    {
        workQueue.push(item);
    }

    void start()
    {
        MLSGPU_ASSERT(threads.empty(), std::runtime_error);
        threads.reserve(workers.size());
        for (std::size_t i = 0; i < workers.size(); i++)
        {
            threads.push_back(new boost::thread(Thread(*this, workers[i])));
        }
    }

    void stop()
    {
        MLSGPU_ASSERT(threads.size() == workers.size(), std::runtime_error);
        for (std::size_t i = 0; i < threads.size(); i++)
            workQueue.push(boost::shared_ptr<WorkItem>());
        for (std::size_t i = 0; i < threads.size(); i++)
            threads[i].join();
        threads.clear();
    }

    std::size_t numWorkers() const
    {
        return workers.size();
    }

protected:
    void addWorker(Worker *worker)
    {
        workers.push_back(worker);
    }

    void addPoolItem(boost::shared_ptr<WorkItem> item)
    {
        itemPool.push(item);
    }

    Worker &getWorker(std::size_t index)
    {
        return workers.at(index);
    }

    WorkerGroup(std::size_t numWorkers, std::size_t capacity)
        : workQueue(capacity), itemPool(capacity)
    {
        MLSGPU_ASSERT(numWorkers > 0, std::invalid_argument);
        workers.reserve(numWorkers);
    }

private:
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
                boost::shared_ptr<WorkItem> item = owner.workQueue.pop();
                if (!item.get())
                    break; // we have been asked to shut down
                worker(*item);
                owner.itemPool.push(item);
            }
        }
    };

    WorkQueue<boost::shared_ptr<WorkItem> > workQueue;
    Pool<WorkItem> itemPool;

    boost::ptr_vector<boost::thread> threads;
    boost::ptr_vector<Worker> workers;
};

class DeviceWorkerGroup;

class DeviceWorkerGroupBase
{
public:
    /**
     * Data about a fine-grained bucket. A shared pointer to this is obtained
     * from @ref get and enqueued with @ref push.
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
    ProgressDisplay *progress;

    const Grid fullGrid;
    const std::size_t maxSplats;
    const Grid::size_type maxCells;
    const int subsampling;

    friend class DeviceWorkerGroupBase::Worker;

public:
    typedef DeviceWorkerGroupBase::WorkItem WorkItem;

    DeviceWorkerGroup(
        std::size_t numWorkers, std::size_t capacity,
        const Grid &fullGrid,
        const cl::Context &context, const cl::Device &device,
        std::size_t maxSplats, Grid::size_type maxCells,
        int levels, int subsampling, bool keepBoundary, float boundaryLimit);

    static CLH::ResourceUsage resourceUsage(
        std::size_t numWorkers, std::size_t capacity,
        const cl::Device &device,
        std::size_t maxSplats, Grid::size_type maxCells,
        int levels, bool keepBoundary);

    void setProgress(ProgressDisplay *progress) { this->progress = progress; }

    void setOutput(const Marching::OutputFunctor &output);
};

class FineBucketGroup;

class FineBucketGroupBase
{
public:
    struct WorkItem
    {
        std::vector<Splat> splats;
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
        typedef SplatSet::SimpleSet<boost::ptr_vector<StdVectorCollection<Splat> > > Set;

        Worker(FineBucketGroup &owner, const cl::Context &context, const cl::Device &device);

        /// Bucketing callback for blocks sized for device execution.
        void operator()(
            const Set &splatSet,
            Bucket::Range::index_type numSplats,
            Bucket::RangeConstIterator first,
            Bucket::RangeConstIterator last,
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
