/**
 * @file
 *
 * Collection of classes for doing specific steps from the main program.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif

#ifndef __CL_ENABLE_EXCEPTIONS
# define __CL_ENABLE_EXCEPTIONS
#endif

#include <cstddef>
#include <vector>
#include <CL/cl.hpp>
#include <boost/smart_ptr/shared_ptr.hpp>
#include <boost/smart_ptr/make_shared.hpp>
#include <boost/smart_ptr/scoped_ptr.hpp>
#include <boost/ref.hpp>
#include <boost/bind.hpp>
#include "grid.h"
#include "workers.h"
#include "work_queue.h"
#include "splat_tree_cl.h"
#include "splat.h"
#include "splat_set.h"
#include "bucket.h"
#include "mesh.h"
#include "mesh_filter.h"
#include "statistics.h"
#include "errors.h"
#include "thread_name.h"

MesherGroupBase::Worker::Worker(MesherGroup &owner)
    : WorkerBase("mesher", 0), owner(owner) {}

void MesherGroupBase::Worker::operator()(WorkItem &work)
{
    Timeplot::Action timer("compute", getTimeplotWorker(), owner.getComputeStat());
    owner.input(work);
}

MesherGroup::MesherGroup(std::size_t spare)
    : WorkerGroup<MesherGroupBase::WorkItem, MesherGroupBase::Worker, MesherGroup>(
        "mesher",
        1, spare)
{
    for (std::size_t i = 0; i < 1 + spare; i++)
        addPoolItem(boost::make_shared<WorkItem>());
    addWorker(new Worker(*this));
}

Marching::OutputFunctor MesherGroup::getOutputFunctor(const ChunkId &chunkId, Timeplot::Worker &tworker)
{
    return boost::bind(&MesherGroup::outputFunc, this, boost::ref(tworker), chunkId, _1, _2, _3, _4);
}

void MesherGroup::outputFunc(
    Timeplot::Worker &tworker,
    const ChunkId &chunkId,
    const cl::CommandQueue &queue,
    const DeviceKeyMesh &mesh,
    const std::vector<cl::Event> *events,
    cl::Event *event)
{
    MLSGPU_ASSERT(input, std::logic_error);

    boost::shared_ptr<MesherWork> work = get(tworker);
    std::vector<cl::Event> wait(3);
    enqueueReadMesh(queue, mesh, work->mesh, events, &wait[0], &wait[1], &wait[2]);
    CLH::enqueueMarkerWithWaitList(queue, &wait, event);

    work->chunkId = chunkId;
    work->hasEvents = true;
    work->verticesEvent = wait[0];
    work->vertexKeysEvent = wait[1];
    work->trianglesEvent = wait[2];
    push(work, tworker);
}


DeviceWorkerGroup::DeviceWorkerGroup(
    std::size_t numWorkers, std::size_t spare,
    OutputGenerator outputGenerator,
    const std::vector<std::pair<cl::Context, cl::Device> > &devices,
    std::size_t maxSplats, Grid::size_type maxCells,
    int levels, int subsampling, float boundaryLimit,
    MlsShape shape)
:
    Base(
        "device",
        devices.size(), numWorkers, spare),
    progress(NULL), outputGenerator(outputGenerator),
    maxSplats(maxSplats), maxCells(maxCells), subsampling(subsampling)
{
    for (std::size_t i = 0; i < (numWorkers + spare) * devices.size(); i++)
    {
        const std::pair<cl::Context, cl::Device> &cd = devices[i % devices.size()];
        const cl::Context &context = cd.first;
        const cl::Device &device = cd.second;
        boost::shared_ptr<WorkItem> item = boost::make_shared<WorkItem>();
        item->key = device();
        item->mapQueue = cl::CommandQueue(context, device);
        item->splats = cl::Buffer(context, CL_MEM_READ_ONLY, maxSplats * sizeof(Splat));
        item->numSplats = 0;
        addPoolItem(item);
    }
    for (std::size_t i = 0; i < numWorkers * devices.size(); i++)
    {
        const std::pair<cl::Context, cl::Device> &cd = devices[i % devices.size()];
        addWorker(new Worker(*this, cd.first, cd.second, levels, boundaryLimit, shape, i));
    }
}

void DeviceWorkerGroup::start(const Grid &fullGrid)
{
    this->fullGrid = fullGrid;
    this->Base::start();
}

CLH::ResourceUsage DeviceWorkerGroup::resourceUsage(
    std::size_t numWorkers, std::size_t spare,
    const cl::Device &device,
    std::size_t maxSplats, Grid::size_type maxCells,
    int levels)
{
    Grid::size_type block = maxCells + 1;
    CLH::ResourceUsage sliceUsage =
        MlsFunctor::sliceResourceUsage(block, block);
    CLH::ResourceUsage workerUsage;
    workerUsage += Marching::resourceUsage(device, block, block, block, sliceUsage);
    workerUsage += SplatTreeCL::resourceUsage(device, levels, maxSplats);

    CLH::ResourceUsage itemUsage;
    itemUsage.addBuffer(maxSplats * sizeof(Splat));
    return workerUsage * numWorkers + itemUsage * (numWorkers + spare);
}

DeviceWorkerGroupBase::Worker::Worker(
    DeviceWorkerGroup &owner,
    const cl::Context &context, const cl::Device &device,
    int levels, float boundaryLimit,
    MlsShape shape, int idx)
:
    WorkerBase("device", idx),
    owner(owner),
    key(device()),
    queue(context, device, CL_QUEUE_PROFILING_ENABLE),
    tree(context, device, levels, owner.maxSplats),
    input(context, shape),
    marching(context, device, input, owner.maxCells + 1, owner.maxCells + 1, owner.maxCells + 1),
    scaleBias(context)
{
    input.setBoundaryLimit(boundaryLimit);
    filterChain.addFilter(boost::ref(scaleBias));
}

void DeviceWorkerGroupBase::Worker::start()
{
    scaleBias.setScaleBias(owner.fullGrid);
}

void DeviceWorkerGroupBase::Worker::operator()(WorkItem &work)
{
    cl_uint3 keyOffset;
    for (int i = 0; i < 3; i++)
        keyOffset.s[i] = work.grid.getExtent(i).first;
    // same thing, just as a different type for a different API
    Grid::difference_type offset[3] =
    {
        (Grid::difference_type) keyOffset.s[0],
        (Grid::difference_type) keyOffset.s[1],
        (Grid::difference_type) keyOffset.s[2]
    };

    Grid::size_type size[3];
    for (int i = 0; i < 3; i++)
    {
        /* Note: numVertices not numCells, because Marching does per-vertex queries.
         * So we need information about the cell that is just beyond the last vertex,
         * just to avoid special-casing it.
         */
        size[i] = work.grid.numVertices(i);
    }

    /* We need to round up the octree size to a multiple of the granularity used for MLS. */
    Grid::size_type expandedSize[3];
    for (int i = 0; i < 3; i++)
        expandedSize[i] = roundUp(size[i], MlsFunctor::wgs[i]);

    filterChain.setOutput(owner.outputGenerator(work.chunkId, getTimeplotWorker()));

    {
        Timeplot::Action timer("compute", getTimeplotWorker(), owner.getComputeStat());
        cl::Event treeBuildEvent;
        std::vector<cl::Event> wait(1);

        wait[0] = work.unmapEvent;
        tree.enqueueBuild(queue, work.splats, work.numSplats,
                          expandedSize, offset, owner.subsampling, &wait, &treeBuildEvent);
        wait[0] = treeBuildEvent;

        input.set(offset, tree, owner.subsampling);
        marching.generate(queue, input, filterChain, size, keyOffset, &wait);
    }

    if (owner.progress != NULL)
        *owner.progress += work.grid.numCells();
}


FineBucketGroup::FineBucketGroup(
    std::size_t numWorkers, std::size_t spare,
    DeviceWorkerGroup &outGroup,
    std::size_t maxCoarseSplats,
    std::size_t maxSplats,
    Grid::size_type maxCells,
    std::size_t maxSplit)
:
    WorkerGroup<FineBucketGroup::WorkItem, FineBucketGroup::Worker, FineBucketGroup>(
        "bucket.fine",
        numWorkers, spare),
    outGroup(outGroup),
    maxSplats(maxSplats),
    maxCells(maxCells),
    maxSplit(maxSplit),
    progress(NULL)
{
    for (std::size_t i = 0; i < numWorkers; i++)
    {
        addWorker(new Worker(*this, i));
    }
    for (std::size_t i = 0; i < numWorkers + spare; i++)
    {
        boost::shared_ptr<WorkItem> item = boost::make_shared<WorkItem>();
        item->splats.reserve(maxCoarseSplats);
        addPoolItem(item);
    }
}

void FineBucketGroup::start(const Grid &fullGrid)
{
    this->fullGrid = fullGrid;
    this->BaseType::start();
}

FineBucketGroupBase::Worker::Worker(FineBucketGroup &owner, int idx)
    : WorkerBase("bucket.fine", idx), owner(owner)
{
};

void FineBucketGroupBase::Worker::operator()(
    const ChunkId &chunkId,
    const SplatSet::Traits<Splats>::subset_type &splatSet,
    const Grid &grid,
    const Bucket::Recursion &recursionState)
{
    Statistics::Registry &registry = Statistics::Registry::getInstance();

    boost::shared_ptr<DeviceWorkerGroup::WorkItem> outItem = owner.outGroup.get(getTimeplotWorker());
    outItem->numSplats = splatSet.numSplats();
    outItem->grid = grid;
    outItem->recursionState = recursionState;
    outItem->chunkId = chunkId;

    {
        Timeplot::Action timer("compute", getTimeplotWorker(), owner.getComputeStat());
        CLH::BufferMapping<Splat> splats(outItem->splats, outItem->mapQueue, CL_MAP_WRITE,
                                         0, splatSet.numSplats() * sizeof(Splat));

        std::size_t pos = 0;
        boost::scoped_ptr<SplatSet::SplatStream> splatStream(splatSet.makeSplatStream());
        while (!splatStream->empty())
        {
            splats[pos] = **splatStream;
            ++*splatStream;
            ++pos;
        }
        assert(pos == splatSet.numSplats());

        registry.getStatistic<Statistics::Variable>("bucket.fine.splats").add(outItem->numSplats);
        registry.getStatistic<Statistics::Variable>("bucket.fine.ranges").add(splatSet.numRanges());
        registry.getStatistic<Statistics::Variable>("bucket.fine.size").add(grid.numCells());

        splats.reset(NULL, &outItem->unmapEvent);
        outItem->mapQueue.flush();
    }

    owner.outGroup.push(outItem, getTimeplotWorker());
}

void FineBucketGroupBase::Worker::operator()(WorkItem &work)
{
    Timeplot::Action timer("compute", getTimeplotWorker(), owner.getComputeStat());

    /* The host transformed splats from world space into fullGrid space, so we need to
     * construct a new grid for this coordinate system.
     */
    const float ref[3] = {0.0f, 0.0f, 0.0f};
    Grid grid(ref, 1.0f, 0, 1, 0, 1, 0, 1);
    for (unsigned int i = 0; i < 3; i++)
    {
        Grid::difference_type base = owner.fullGrid.getExtent(i).first;
        Grid::difference_type low = work.grid.getExtent(i).first - base;
        Grid::difference_type high = work.grid.getExtent(i).second - base;
        grid.setExtent(i, low, high);
    }

    work.splats.computeBlobs(grid.getSpacing(), owner.maxCells, NULL, false);
    Bucket::bucket(work.splats, grid, owner.maxSplats, owner.maxCells, 0, false, owner.maxSplit,
                   boost::bind<void>(boost::ref(*this), work.chunkId, _1, _2, _3),
                   owner.progress, work.recursionState);
}
