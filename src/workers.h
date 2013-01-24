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
        std::size_t firstSplat;        ///< Index of first splat in device buffer
        std::size_t numSplats;         ///< Number of splats in the bucket
        /**
         * Pointer at which to start writing splats (already offset by @a
         * firstSplat). This is only valid for producers.
         */
        Splat *splats;
        Grid grid;
        Bucket::Recursion recursionState;
    };

    /**
     * Data about multiple fine-grained buckets that share a single CL buffer.
     */
    struct WorkItem
    {
        /**
         * Data for individual fine buckets. This is a linked list rather than
         * a vector because other threads can append to the list asynchronously,
         * and this must not move the memory around.
         */
        Statistics::Container::list<SubItem> subItems;
        cl::Buffer splats;             ///< Backing store for splats
        cl::Event copyEvent;           ///< Event signaled when the splats are ready to use on device

        std::size_t nextSplat() const
        {
            if (subItems.empty())
                return 0;
            else
                return subItems.back().firstSplat + subItems.back().numSplats;
        }

        WorkItem() : subItems("mem.FineBucketGroup.subItems") {}
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
 * it to the @ref MesherBase class. It pulls chunks of work off a queue,
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
    const std::size_t maxSplats;  ///< Maximum splats in a single bucket
    const Grid::size_type maxCells;
    const std::size_t meshMemory;
    const int subsampling;
    std::size_t maxItemSplats;    ///< Maximum number of splats per work item

    cl::CommandQueue copyQueue;   ///< Queue for transferring data to the device

    /// Pool of unused buffers to be recycled
    WorkQueue<boost::shared_ptr<WorkItem> > itemPool;

    /**
     * Work item currently being filled (by all producers). This may at times
     * be @c NULL, provided that @c activeWriters is also zero. It is neither
     * in the queue nor the item pool. When it is full and once activeWriters
     * drops to zero, it is pushed onto the queue.
     */
    boost::shared_ptr<WorkItem> writeItem;

    /**
     * Pinned host memory to write splats to, prior to them being copied
     * to the GPU.
     */
    boost::scoped_ptr<CLH::PinnedMemory<Splat> > writePinned;

    /**
     * Number of writers that are currently writing through the mapped pointer.
     * It will only be safe to unmap the buffer once this hits zero.
     */
    std::size_t activeWriters;

    /**
     * Mutex protecting @ref writeItem (both the pointer value and the
     * contents, but not individual items in the list).
     */
    boost::mutex splatsMutex;

    /**
     * Mutex protecting @ref activeWriters. If held concurrently with @ref
     * splatsMutex, it must be taken afterwards.
     */
    boost::mutex activeMutex;

    /**
     * Condition that is signalled when the number of writers reaches zero,
     * making it safe to enqueue the write item.
     */
    boost::condition_variable inactiveCondition;

    /**
     * Pushes the current @ref writeItem (if any) into the queue, and resets it.
     *
     * @pre The caller holds @ref splatsMutex.
     */
    void flushWriteItem();

    friend class DeviceWorkerGroupBase::Worker;

public:
    typedef DeviceWorkerGroupBase::WorkItem WorkItem;
    typedef DeviceWorkerGroupBase::SubItem SubItem;

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
     * @copydoc WorkerGroup::stop
     */
    void stop();

    /**
     * Sets a progress display that will be updated by the number of cells
     * processed.
     */
    void setProgress(ProgressMeter *progress) { this->progress = progress; }

    /**
     * @copydoc WorkerGroup::get
     */
    SubItem &get(Timeplot::Worker &tworker, std::size_t size);

    /**
     * @copydoc WorkerGroup::push
     */
    void push();

    /**
     * Returns the item to the pool. It is called by the base class.
     */
    void freeItem(boost::shared_ptr<WorkItem> item);

    /**
     * Estimate spare queue capacity.
     *
     * @todo Fix
     */
    std::size_t unallocated() { return 1; }
};

class FineBucketGroup;

class FineBucketGroupBase
{
public:
    typedef SplatSet::FastBlobSet<SplatSet::SequenceSet<const Splat *>, std::vector<SplatSet::BlobData> > Splats;

    struct WorkItem
    {
        ChunkId chunkId;
        CircularBuffer::Allocation splats;
        std::size_t numSplats;
        Grid grid;
        Bucket::Recursion recursionState;
    };

    class Worker : public WorkerBase
    {
    private:
        FineBucketGroup &owner;

    public:
        typedef void result_type;

        Worker(FineBucketGroup &owner, int idx);

        /// Bucketing callback for blocks sized for device execution.
        void operator()(
            const ChunkId &chunkId,
            const SplatSet::Traits<Splats>::subset_type &splats,
            const Grid &grid,
            const Bucket::Recursion &recursionState);

        void operator()(WorkItem &work);
    };
};

/**
 * A worker object that handles coarse-to-fine bucketing. It pulls work from
 * an internal queue (containing regions of splats already read from storage),
 * calls @ref Bucket::bucket to subdivide the splats into buckets suitable for
 * device execution, and passes them on to a @ref DeviceWorkerGroup.
 */
class FineBucketGroup :
    protected FineBucketGroupBase,
    public WorkerGroup<FineBucketGroupBase::WorkItem, FineBucketGroupBase::Worker, FineBucketGroup>
{
public:
    typedef WorkerGroup<FineBucketGroupBase::WorkItem, FineBucketGroupBase::Worker, FineBucketGroup> BaseType;
    typedef FineBucketGroupBase::WorkItem WorkItem;

    void setProgress(ProgressMeter *progress) { this->progress = progress; }

    FineBucketGroup(
        std::size_t numWorkers,
        const std::vector<DeviceWorkerGroup *> &outGroups,
        std::size_t memCoarseSplats,
        std::size_t maxSplats,
        Grid::size_type maxCells,
        std::size_t maxSplit);

    boost::shared_ptr<WorkItem> get(Timeplot::Worker &tworker, std::size_t size)
    {
        boost::shared_ptr<WorkItem> item = BaseType::get(tworker, size);
        item->splats = splatBuffer.allocate(tworker, size * sizeof(Splat), &getStat);
        item->numSplats = size;
        return item;
    }

    void start(const Grid &fullGrid);

    Statistics::Variable &getWriteStat() const { return writeStat; }

private:
    const std::vector<DeviceWorkerGroup *> outGroups;
    CircularBuffer splatBuffer;

    Grid fullGrid;
    std::size_t maxSplats;
    Grid::size_type maxCells;
    std::size_t maxSplit;
    ProgressMeter *progress;
    Statistics::Variable &writeStat;

    friend class FineBucketGroupBase::Worker;
};

#endif /* !WORKERS_H */
