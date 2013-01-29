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

#ifndef __CL_ENABLE_EXCEPTIONS
# define __CL_ENABLE_EXCEPTIONS
#endif

#include <boost/smart_ptr/shared_ptr.hpp>
#include <boost/smart_ptr/scoped_ptr.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/locks.hpp>
#include <boost/noncopyable.hpp>
#include <boost/foreach.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/ptr_container/ptr_map.hpp>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <CL/cl.hpp>
#include "splat_tree_cl.h"
#include "marching.h"
#include "mls.h"
#include "mesh.h"
#include "mesher.h"
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
#include "allocator.h"
#include "worker_group.h"
#include "timeplot.h"

class MesherGroup;

class MesherGroupBase
{
public:
    struct WorkItem
    {
        MesherWork work;
        CircularBuffer::Allocation alloc; ///< Allocation backing the mesh data
    };

    class Worker : public WorkerBase
    {
    private:
        MesherGroup &owner;

    public:
        typedef void result_type;

        Worker(MesherGroup &owner);
        void operator()(WorkItem &work);
    };
};

/**
 * Object for handling asynchronous meshing. It always uses one consumer thread, since
 * the operation is fundamentally not thread-safe. However, there may be multiple
 * producers.
 */
class MesherGroup : protected MesherGroupBase,
    public WorkerGroup<MesherGroupBase::WorkItem, MesherGroupBase::Worker, MesherGroup>
{
public:
    typedef MesherGroupBase::WorkItem WorkItem;

    /// Set the functor to use for processing data received from the output functor.
    void setInputFunctor(const MesherBase::InputFunctor &input) { this->input = input; }

    /**
     * Retrieve a functor that can be used in any thread to insert work into
     * the queue.
     */
    Marching::OutputFunctor getOutputFunctor(const ChunkId &chunkId, Timeplot::Worker &tworker);

    boost::shared_ptr<WorkItem> get(Timeplot::Worker &tworker, std::size_t size);

    /**
     * Constructor.
     *
     * @param memMesh Memory (in bytes) to allocate for holding queued mesh data.
     */
    explicit MesherGroup(const std::size_t memMesh);
private:
    MesherBase::InputFunctor input;
    CircularBuffer meshBuffer;

    friend class MesherGroupBase::Worker;

    void outputFunc(
        Timeplot::Worker &tworker,
        const ChunkId &chunkId,
        const cl::CommandQueue &queue,
        const DeviceKeyMesh &mesh,
        const std::vector<cl::Event> *events,
        cl::Event *event);
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
    struct SubItem
    {
        ChunkId chunkId;               ///< Chunk owning this item
        Grid grid;
        std::size_t firstSplat;        ///< Index of first splat in device buffer
        std::size_t numSplats;         ///< Number of splats in the bucket
    };

    /**
     * Data about multiple fine-grained buckets that share a single CL buffer.
     */
    struct WorkItem
    {
        /// Data for individual fine buckets. This is a linked list rather than
        Statistics::Container::vector<SubItem> subItems;
        cl::Buffer splats;             ///< Backing store for splats
        cl::Event copyEvent;           ///< Event signaled when the splats are ready to use on device

        std::size_t nextSplat() const
        {
            if (subItems.empty())
                return 0;
            else
                return subItems.back().firstSplat + subItems.back().numSplats;
        }

        WorkItem(const cl::Context &context, std::size_t maxItemSplats)
            : subItems("mem.FineBucketGroup.subItems"),
            splats(context, CL_MEM_READ_WRITE, maxItemSplats * sizeof(Splat))
        {
        }
    };

    class Worker : public WorkerBase
    {
    private:
        DeviceWorkerGroup &owner;

        const cl::CommandQueue queue;
        SplatTreeCL tree;
        MlsFunctor input;
        Marching marching;
        ScaleBiasFilter scaleBias;
        MeshFilterChain filterChain;

    public:
        typedef void result_type;

        Worker(
            DeviceWorkerGroup &owner,
            const cl::Context &context, const cl::Device &device,
            int levels, float boundaryLimit,
            MlsShape shape, int idx);

        void start();
        void operator()(WorkItem &work);
    };
};

/**
 * Does the actual OpenCL calls necessary to compute the mesh and write
 * it to the @ref MesherBase class. It pulls bins of work off a queue,
 * which contains pre-bucketed splats.
 */
class DeviceWorkerGroup :
    protected DeviceWorkerGroupBase,
    public WorkerGroup<DeviceWorkerGroupBase::WorkItem, DeviceWorkerGroupBase::Worker, DeviceWorkerGroup>
{
public:
    /**
     * Functor that generates an output function given the current chunk ID and
     * worker. This is used to abstract the downstream worker group class.
     *
     * @see @ref DeviceWorkerGroup::DeviceWorkerGroup
     */
    typedef boost::function<Marching::OutputFunctor(const ChunkId &, Timeplot::Worker &)> OutputGenerator;

private:
    typedef WorkerGroup<DeviceWorkerGroupBase::WorkItem, DeviceWorkerGroupBase::Worker, DeviceWorkerGroup> Base;

    ProgressMeter *progress;
    OutputGenerator outputGenerator;

    Grid fullGrid;
    const cl::Context context;
    const cl::Device device;
    const std::size_t maxSplats;  ///< Maximum splats in a single bucket
    const Grid::size_type maxCells;
    const std::size_t meshMemory;
    const int subsampling;

    cl::CommandQueue copyQueue;   ///< Queue for transferring data to the device

    /// Pool of unused buffers to be recycled
    WorkQueue<boost::shared_ptr<WorkItem> > itemPool;

    /// Mutex held while signaling @ref popCondition
    boost::mutex *popMutex;

    /// Condition signaled when items are added to the pool
    boost::condition_variable *popCondition;

    /// Number of spare splats in device buffers.
    std::size_t unallocated_;
    /// Mutex protecting @ref unallocated_.
    boost::mutex unallocatedMutex;

    friend class DeviceWorkerGroupBase::Worker;

public:
    typedef DeviceWorkerGroupBase::WorkItem WorkItem;
    typedef DeviceWorkerGroupBase::SubItem SubItem;
    typedef boost::shared_ptr<WorkItem> get_type;

    /**
     * Constructor.
     *
     * @param numWorkers         Number of worker threads to use (each with a separate OpenCL queue and state)
     * @param spare              Number of extra slots (beyond @a numWorkers) for items.
     * @param outputGenerator    Output handler generator. The generator is passed a chunk
     *                           ID and @ref Timeplot::Worker, and returns a @ref Marching::OutputFunctor which
     *                           which will receive the output blocks for the corresponding chunk.
     * @param context            OpenCL context to run on.
     * @param device             OpenCL device to run on.
     * @param maxSplats          Space to allocate for holding splats.
     * @param maxCells           Space to allocate for the octree.
     * @param memSplats          Device bytes to use for queued splats.
     * @param meshMemory         Maximum device bytes to use for mesh-related data.
     * @param levels             Levels to allocate for the octree.
     * @param subsampling        Octree subsampling level.
     * @param boundaryLimit      Tuning factor for boundary pruning.
     * @param shape              The shape to fit to the data
     */
    DeviceWorkerGroup(
        std::size_t numWorkers, std::size_t spare,
        OutputGenerator outputGenerator,
        const cl::Context &context, const cl::Device &device,
        std::size_t maxSplats, Grid::size_type maxCells,
        std::size_t memSplats, std::size_t meshMemory,
        int levels, int subsampling, float boundaryLimit,
        MlsShape shape);

    /// Returns total resources that would be used by all workers and workitems
    static CLH::ResourceUsage resourceUsage(
        std::size_t numWorkers,
        const cl::Device &device,
        std::size_t maxSplats, Grid::size_type maxCells,
        std::size_t memSplats, std::size_t meshMemory,
        int levels);

    /**
     * @copydoc WorkerGroup::start
     *
     * @param fullGrid  The bounding box grid.
     */
    void start(const Grid &fullGrid);

    /**
     * Sets a progress display that will be updated by the number of cells
     * processed.
     */
    void setProgress(ProgressMeter *progress) { this->progress = progress; }

    /**
     * Set a condition variable that will be signaled when space becomes
     * available in the item pool. The condition will be signaled with
     * the mutex held.
     */
    void setPopCondition(boost::mutex *mutex, boost::condition_variable *condition)
    {
        popMutex = mutex;
        popCondition = condition;
    }

    /**
     * @copydoc WorkerGroup::get
     */
    boost::shared_ptr<WorkItem> get(Timeplot::Worker &tworker, std::size_t size);

    /**
     * Determine whether @ref get will block.
     */
    bool canGet();

    /**
     * Returns the item to the pool. It is called by the base class.
     */
    void freeItem(boost::shared_ptr<WorkItem> item);

    /**
     * Estimate spare queue capacity. It takes the theoretical maximum capacity
     * and subtracts splats that are in the queue. It is not necessarily possible
     * to allocate this many.
     */
    std::size_t unallocated();

    const cl::Context &getContext() const { return context; }
    const cl::Device &getDevice() const { return device; }
    const cl::CommandQueue &getCopyQueue() const { return copyQueue; }
    Statistics::Variable &getGetStat() const { return getStat; }
};

class FineBucketGroup;

class FineBucketGroupBase
{
public:
    /// A single bin of splats
    struct WorkItem
    {
        ChunkId chunkId;
        Grid grid;
        CircularBuffer::Allocation splats;  ///< Allocation from @ref FineBucketGroup::splatBuffer
        std::size_t numSplats;              ///< Number of splats in the bin

        Splat *getSplats() const { return (Splat *) splats.get(); }
    };

    class Worker : public WorkerBase
    {
    private:
        FineBucketGroup &owner;
        CLH::PinnedMemory<Splat> pinned;  ///< Staging area for copies
        /**
         * Bins that have been saved up but not yet flushed to the device.
         */
        Statistics::Container::vector<DeviceWorkerGroup::SubItem> bufferedItems;
        std::size_t bufferedSplats;       ///< Number of splats stored in @ref pinned

    public:
        typedef void result_type;

        Worker(FineBucketGroup &owner, const cl::Context &context, const cl::Device &device);

        void flush();   ///< Flush items in @ref bufferedItems to the output
        void operator()(WorkItem &work);
        void stop() { flush(); }
    };
};

/**
 * A worker object that copies bins of data to the GPU. It receives data from
 * @ref CoarseBucket and sends it to the next available @ref DeviceWorkerGroup.
 */
class FineBucketGroup :
    protected FineBucketGroupBase,
    public WorkerGroup<FineBucketGroupBase::WorkItem, FineBucketGroupBase::Worker, FineBucketGroup>
{
public:
    typedef WorkerGroup<FineBucketGroupBase::WorkItem, FineBucketGroupBase::Worker, FineBucketGroup> BaseType;
    typedef FineBucketGroupBase::WorkItem WorkItem;
    typedef boost::shared_ptr<WorkItem> get_type;

    /**
     * Constructor.
     * @param outGroups       Target devices. The first is used for allocating pinned memory.
     * @param memSplats       Memory for splats in the internal queue.
     * @param maxDeviceSplats Maximum splats to send to a device worker in one go.
     *
     * @todo @a maxDeviceSplats should be retrieved from the output group, to avoid possible
     * mismatches.
     */
    FineBucketGroup(
        const std::vector<DeviceWorkerGroup *> &outGroups,
        std::size_t memSplats,
        std::size_t maxDeviceSplats);

    /**
     * @copydoc WorkerGroup::get
     */
    boost::shared_ptr<WorkItem> get(Timeplot::Worker &tworker, std::size_t size)
    {
        boost::shared_ptr<WorkItem> item = BaseType::get(tworker, size);
        item->splats = splatBuffer.allocate(tworker, size * sizeof(Splat), &getStat);
        item->numSplats = size;
        return item;
    }

    /// Statistic for timing @c clEnqueueWriteBuffer
    Statistics::Variable &getWriteStat() const { return writeStat; }

    // TODO: eliminate once CoarseBucket takes a singular output again
    std::size_t unallocated() const { return 1; }

private:
    const std::vector<DeviceWorkerGroup *> outGroups;
    const std::size_t maxDeviceSplats;         ///< Maximum splats to send to the device in one go
    CircularBuffer splatBuffer;                ///< Buffer holding incoming splats

    boost::mutex popMutex;                     ///< Mutex held while checking for device to target
    boost::condition_variable popCondition;    ///< Condition signalled by devices when space available

    Statistics::Variable &writeStat;           ///< See @ref getWriteStat

    friend class FineBucketGroupBase::Worker;
};

#endif /* !WORKERS_H */
