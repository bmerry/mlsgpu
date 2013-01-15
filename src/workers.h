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
#include "worker_group.h"
#include "timeplot.h"

class MesherGroup;

class MesherGroupBase
{
public:
    typedef MesherWork WorkItem;

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

    MesherGroup(std::size_t spare);
private:
    MesherBase::InputFunctor input;
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
    struct WorkItem
    {
        cl_device_id key;
        ChunkId chunkId;               ///< Chunk owning this item
        cl::CommandQueue mapQueue;     ///< Queue for mapping and unmapping the buffer
        cl::Event unmapEvent;          ///< Event signaled when the splats are ready to use

        cl::Buffer splats;
        std::size_t numSplats;
        Grid grid;
        Bucket::Recursion recursionState;

        cl_device_id getKey() const { return key; }
    };

    class Worker : public WorkerBase
    {
    private:
        DeviceWorkerGroup &owner;

        cl_device_id key;
        const cl::CommandQueue queue;
        SplatTreeCL tree;
        MlsFunctor input;
        Marching marching;
        ScaleBiasFilter scaleBias;
        MeshFilterChain filterChain;

    public:
        typedef void result_type;

        cl_device_id getKey() const { return key; }

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
    public WorkerGroupMulti<DeviceWorkerGroupBase::WorkItem, DeviceWorkerGroupBase::Worker, DeviceWorkerGroup, cl_device_id>
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
    typedef WorkerGroupMulti<DeviceWorkerGroupBase::WorkItem, DeviceWorkerGroupBase::Worker, DeviceWorkerGroup, cl_device_id> Base;
    ProgressMeter *progress;
    OutputGenerator outputGenerator;

    Grid fullGrid;
    const std::size_t maxSplats;
    const Grid::size_type maxCells;
    const std::size_t meshMemory;
    const int subsampling;

    friend class DeviceWorkerGroupBase::Worker;

public:
    typedef DeviceWorkerGroupBase::WorkItem WorkItem;

    /**
     * Constructor.
     *
     * @param numWorkers         Number of worker threads to use (each with a separate OpenCL queue and state)
     * @param spare              Number of work items to have available in the pool when all workers are busy.
     * @param outputGenerator    Output handler generator. The generator is passed a chunk
     *                           ID and @ref Timeplot::Worker, and returns a @ref Marching::OutputFunctor which
     *                           which will receive the output blocks for the corresponding chunk.
     * @param devices            OpenCL context and device to run on, with associated contexts.
     * @param maxSplats          Space to allocate for holding splats.
     * @param maxCells           Space to allocate for the octree.
     * @param meshMemory         Maximum device bytes to use for mesh-related data.
     * @param levels             Levels to allocate for the octree.
     * @param subsampling        Octree subsampling level.
     * @param boundaryLimit      Tuning factor for boundary pruning.
     * @param shape              The shape to fit to the data
     */
    DeviceWorkerGroup(
        std::size_t numWorkers, std::size_t spare,
        OutputGenerator outputGenerator,
        const std::vector<std::pair<cl::Context, cl::Device> > &devices,
        std::size_t maxSplats, Grid::size_type maxCells, std::size_t meshMemory,
        int levels, int subsampling, float boundaryLimit,
        MlsShape shape);

    /// Returns total resources that would be used by all workers and workitems
    static CLH::ResourceUsage resourceUsage(
        std::size_t numWorkers, std::size_t spare,
        const cl::Device &device,
        std::size_t maxSplats, Grid::size_type maxCells, std::size_t meshMemory,
        int levels);

    /**
     * @copydoc WorkerGroupMulti::start
     *
     * @param fullGrid  The bounding box grid.
     */
    void start(const Grid &fullGrid);

    /**
     * Sets a progress display that will be updated by the number of cells
     * processed.
     */
    void setProgress(ProgressMeter *progress) { this->progress = progress; }
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
        DeviceWorkerGroup &outGroup,
        std::size_t maxCoarseSplats,
        std::size_t maxSplats,
        Grid::size_type maxCells,
        std::size_t maxSplit);

    boost::shared_ptr<WorkItem> get(Timeplot::Worker &tworker, std::size_t size)
    {
        boost::shared_ptr<WorkItem> item = BaseType::get(tworker, size);
        item->splats = splatBuffer.allocate(size * sizeof(Splat));
        item->numSplats = size;
        return item;
    }

    void start(const Grid &fullGrid);

private:
    static const std::size_t spare;

    DeviceWorkerGroup &outGroup;
    CircularBuffer splatBuffer;

    Grid fullGrid;
    std::size_t maxSplats;
    Grid::size_type maxCells;
    std::size_t maxSplit;
    ProgressMeter *progress;

    friend class FineBucketGroupBase::Worker;
};

#endif /* !WORKERS_H */
