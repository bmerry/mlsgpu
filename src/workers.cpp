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
#include <boost/thread/locks.hpp>
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
#include "statistics_cl.h"
#include "errors.h"
#include "thread_name.h"
#include "misc.h"

const std::size_t DeviceWorkerGroup::spare = 64;
const std::size_t FineBucketGroup::spare = 64;
const std::size_t MesherGroup::spare = 64;

MesherGroupBase::Worker::Worker(MesherGroup &owner)
    : WorkerBase("mesher", 0), owner(owner) {}

void MesherGroupBase::Worker::operator()(WorkItem &item)
{
    Timeplot::Action timer("compute", getTimeplotWorker(), owner.getComputeStat());
    owner.input(item.work, getTimeplotWorker());
    owner.meshBuffer.free(item.alloc);
}

MesherGroup::MesherGroup(std::size_t memMesh)
    : WorkerGroup<MesherGroupBase::WorkItem, MesherGroupBase::Worker, MesherGroup>(
        "mesher", 1, spare),
    meshBuffer("mem.MesherGroup.mesh", memMesh)
{
    addWorker(new Worker(*this));
}

boost::shared_ptr<MesherGroup::WorkItem> MesherGroup::get(Timeplot::Worker &tworker, std::size_t size)
{
    boost::shared_ptr<WorkItem> item = WorkerGroup<WorkItem, Worker, MesherGroup>::get(tworker, size);
    std::size_t rounded = roundUp(size, sizeof(cl_ulong)); // to ensure alignment
    item->alloc = meshBuffer.allocate(tworker, rounded);
    return item;
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

    std::size_t bytes = mesh.getHostBytes();

    boost::shared_ptr<WorkItem> item = get(tworker, bytes);
    item->work.mesh = HostKeyMesh(item->alloc.get(), mesh);

    std::vector<cl::Event> wait(3);
    enqueueReadMesh(queue, mesh, item->work.mesh, events, &wait[0], &wait[1], &wait[2]);
    CLH::enqueueMarkerWithWaitList(queue, &wait, event);

    item->work.chunkId = chunkId;
    item->work.hasEvents = true;
    item->work.verticesEvent = wait[0];
    item->work.vertexKeysEvent = wait[1];
    item->work.trianglesEvent = wait[2];
    push(item, tworker, bytes);
}


DeviceWorkerGroup::DeviceWorkerGroup(
    std::size_t numWorkers,
    OutputGenerator outputGenerator,
    const cl::Context &context, const cl::Device &device,
    std::size_t maxSplats, Grid::size_type maxCells,
    std::size_t memSplats, std::size_t meshMemory,
    int levels, int subsampling, float boundaryLimit,
    MlsShape shape)
:
    Base("device", numWorkers, spare),
    progress(NULL), outputGenerator(outputGenerator),
    maxSplats(maxSplats), maxCells(maxCells), meshMemory(meshMemory),
    subsampling(subsampling),
    mapQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE),
    splatAlign(device.getInfo<CL_DEVICE_MEM_BASE_ADDR_ALIGN>() / 8),
    splatAllocator("mem.device.splats", roundUp(memSplats, splatAlign)),
    splatStore(context, CL_MEM_READ_WRITE, splatAllocator.size())
{
    for (std::size_t i = 0; i < numWorkers; i++)
    {
        addWorker(new Worker(*this, context, device, levels, boundaryLimit, shape, i));
    }
}

void DeviceWorkerGroup::start(const Grid &fullGrid)
{
    this->fullGrid = fullGrid;
    this->Base::start();
}

boost::shared_ptr<DeviceWorkerGroup::WorkItem> DeviceWorkerGroup::get(
    Timeplot::Worker &tworker, std::size_t size)
{
    boost::shared_ptr<DeviceWorkerGroup::WorkItem> item = Base::get(tworker, size);
    std::size_t bytes = roundUp(size * sizeof(Splat), splatAlign);
    item->alloc = splatAllocator.allocate(tworker, bytes);

    cl_buffer_region region;
    region.origin = item->alloc.get();
    region.size = size * sizeof(Splat);
    {
        boost::lock_guard<boost::mutex> createLock(splatMutex);
        item->splats = splatStore.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region);
    }
    item->numSplats = size;
    return item;
}

CLH::ResourceUsage DeviceWorkerGroup::resourceUsage(
    std::size_t numWorkers,
    const cl::Device &device,
    std::size_t maxSplats, Grid::size_type maxCells,
    std::size_t memSplats, std::size_t meshMemory,
    int levels)
{
    Grid::size_type block = maxCells + 1;

    CLH::ResourceUsage workerUsage;
    workerUsage += Marching::resourceUsage(
        device, block, block, block,
        MlsFunctor::wgs[2], meshMemory, MlsFunctor::wgs);
    workerUsage += SplatTreeCL::resourceUsage(device, levels, maxSplats);

    CLH::ResourceUsage globalUsage;
    globalUsage.addBuffer(memSplats);
    return workerUsage * numWorkers + globalUsage;
}

DeviceWorkerGroupBase::Worker::Worker(
    DeviceWorkerGroup &owner,
    const cl::Context &context, const cl::Device &device,
    int levels, float boundaryLimit,
    MlsShape shape, int idx)
:
    WorkerBase("device", idx),
    owner(owner),
    queue(context, device, Statistics::isEventTimingEnabled() ? CL_QUEUE_PROFILING_ENABLE : 0),
    tree(context, device, levels, owner.maxSplats),
    input(context, shape),
    // OpenCL requires support for images of height 8192; the complex formula below ensures we
    // do not exceed this except when maxCells is large
    marching(context, device, owner.maxCells + 1, owner.maxCells + 1, owner.maxCells + 1,
             input.alignment()[2] * std::max(1, int(divUp(8192, input.alignment()[2] * roundUp(owner.maxCells + 1, input.alignment()[1]))) - 1),
             owner.meshMemory, input.alignment()),
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

        tree.clearSplats();
        {
            boost::lock_guard<boost::mutex> releaseLock(owner.splatMutex);
            work.splats = cl::Buffer(); // free the reference to the sub-buffer
        }
        owner.splatAllocator.free(work.alloc);
    }

    if (owner.progress != NULL)
        *owner.progress += work.grid.numCells();
}

FineBucketGroup::FineBucketGroup(
    std::size_t numWorkers,
    const std::vector<DeviceWorkerGroup *> &outGroups,
    std::size_t memCoarseSplats,
    std::size_t maxSplats,
    Grid::size_type maxCells,
    std::size_t maxSplit)
:
    WorkerGroup<FineBucketGroup::WorkItem, FineBucketGroup::Worker, FineBucketGroup>(
        "bucket.fine",
        numWorkers, spare),
    outGroups(outGroups),
    splatBuffer("mem.FineBucketGroup.splats", memCoarseSplats),
    maxSplats(maxSplats),
    maxCells(maxCells),
    maxSplit(maxSplit),
    progress(NULL)
{
    for (std::size_t i = 0; i < numWorkers; i++)
    {
        addWorker(new Worker(*this, i));
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

    const std::size_t bytes = splatSet.numSplats() * sizeof(Splat);

    /* Select the least-busy device to target */
    DeviceWorkerGroup *outGroup = NULL;
    std::size_t bestSpare = 0;
    BOOST_FOREACH(DeviceWorkerGroup *w, owner.outGroups)
    {
        std::size_t spare = w->unallocated();
        if (spare >= bestSpare)
        {
            // Note: >= above so that we always get a non-NULL result
            outGroup = w;
            bestSpare = spare;
        }
    }

    boost::shared_ptr<DeviceWorkerGroup::WorkItem> outItem =
        outGroup->get(getTimeplotWorker(), splatSet.numSplats());
    outItem->grid = grid;
    outItem->recursionState = recursionState;
    outItem->chunkId = chunkId;

    {
        Timeplot::Action timer("compute", getTimeplotWorker(), owner.getComputeStat());

        cl::Event mapEvent;
        boost::unique_lock<boost::mutex> mapLock(outGroup->getSplatMutex());
        CLH::BufferMapping<Splat> splats(outItem->splats, outGroup->getMapQueue(), CL_MAP_WRITE,
                                         0, bytes, &mapEvent);
        mapLock.release()->unlock();
        outGroup->getMapQueue().flush();
        mapEvent.wait();

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

        {
            boost::lock_guard<boost::mutex> unmapLock(outGroup->getSplatMutex());
            splats.reset(NULL, &outItem->unmapEvent);
        }
        outGroup->getMapQueue().flush();
    }

    outGroup->push(outItem, getTimeplotWorker(), splatSet.numSplats());
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

    const Splat *splatsPtr = (const Splat *) work.splats.get();
    Splats splats;
    splats.reset(splatsPtr, splatsPtr + work.numSplats);
    splats.computeBlobs(grid.getSpacing(), owner.maxCells, NULL, false);
    Bucket::bucket(splats, grid, owner.maxSplats, owner.maxCells, 0, false, owner.maxSplit,
                   boost::bind<void>(boost::ref(*this), work.chunkId, _1, _2, _3),
                   owner.progress, work.recursionState);
    owner.splatBuffer.free(work.splats);
}
