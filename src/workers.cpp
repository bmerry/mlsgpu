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
#include <boost/foreach.hpp>
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
        "mesher", 1),
    meshBuffer("mem.MesherGroup.mesh", memMesh)
{
    addWorker(new Worker(*this));
}

boost::shared_ptr<MesherGroup::WorkItem> MesherGroup::get(Timeplot::Worker &tworker, std::size_t size)
{
    boost::shared_ptr<WorkItem> item = WorkerGroup<WorkItem, Worker, MesherGroup>::get(tworker, size);
    std::size_t rounded = roundUp(size, sizeof(cl_ulong)); // to ensure alignment
    item->alloc = meshBuffer.allocate(tworker, rounded, &getStat);
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
    push(item);
}


DeviceWorkerGroup::DeviceWorkerGroup(
    std::size_t numWorkers, std::size_t spare,
    OutputGenerator outputGenerator,
    const cl::Context &context, const cl::Device &device,
    std::size_t maxSplats, Grid::size_type maxCells,
    std::size_t memSplats, std::size_t meshMemory,
    int levels, int subsampling, float boundaryLimit,
    MlsShape shape)
:
    Base("device", numWorkers),
    progress(NULL), outputGenerator(outputGenerator),
    maxSplats(maxSplats), maxCells(maxCells), meshMemory(meshMemory),
    subsampling(subsampling),
    getFlushStat(Statistics::getStatistic<Statistics::Variable>("device.get.flush")),
    copyQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE),
    itemPool(),
    activeWriters(0)
{
    for (std::size_t i = 0; i < numWorkers; i++)
    {
        addWorker(new Worker(*this, context, device, levels, boundaryLimit, shape, i));
    }
    const std::size_t items = numWorkers + spare;
    maxItemSplats = memSplats / (items * sizeof(Splat));
    MLSGPU_ASSERT(maxItemSplats >= maxSplats, std::invalid_argument);
    writePinned.reset(new CLH::PinnedMemory<Splat>("mem.DeviceWorkerGroup.writePinned", context, device, maxItemSplats));
    for (std::size_t i = 0; i < items; i++)
    {
        boost::shared_ptr<WorkItem> item = boost::make_shared<WorkItem>();
        item->splats = cl::Buffer(context, CL_MEM_READ_WRITE, maxItemSplats * sizeof(Splat));
        itemPool.push(item);
    }
}

void DeviceWorkerGroup::start(const Grid &fullGrid)
{
    this->fullGrid = fullGrid;
    Base::start();
}

void DeviceWorkerGroup::stop()
{
    // Note: we can't just override the stopPreJoin hook, because that still runs after
    // the work queue is stopped
    {
        boost::lock_guard<boost::mutex> lock(splatsMutex);
        flushWriteItem();
    }
    Base::stop();
}

void DeviceWorkerGroup::flushWriteItem()
{
    if (!writeItem)
        return;

    boost::unique_lock<boost::mutex> activeLock(activeMutex);
    while (activeWriters > 0)
    {
        inactiveCondition.wait(activeLock);
    }

    copyQueue.enqueueWriteBuffer(
        writeItem->splats,
        CL_FALSE,
        0, writeItem->nextSplat() * sizeof(Splat),
        writePinned->get(),
        NULL, &writeItem->copyEvent);
    copyQueue.flush();
    Base::push(writeItem);
    writeItem.reset();
}

DeviceWorkerGroup::SubItem &DeviceWorkerGroup::get(
    Timeplot::Worker &tworker, std::size_t numSplats)
{
    Timeplot::Action timer("get", tworker, getStat);
    boost::lock_guard<boost::mutex> splatsLock(splatsMutex);

retry:
    if (!writeItem)
    {
        writeItem = itemPool.pop();
    }
    std::size_t spare = maxItemSplats - writeItem->nextSplat();
    if (numSplats > spare)
    {
        Timeplot::Action timer("get.flush", tworker, getFlushStat);
        flushWriteItem();
        goto retry;
    }

    // Now writeItem is safe to use
    std::size_t start = writeItem->nextSplat();
    SubItem sub;
    sub.firstSplat = start;
    sub.numSplats = numSplats;
    sub.splats = writePinned->get() + start;
    writeItem->subItems.push_back(sub);

    {
        boost::lock_guard<boost::mutex> activeLock(activeMutex);
        activeWriters++;
    }

    return writeItem->subItems.back();
}

void DeviceWorkerGroup::push()
{
    boost::lock_guard<boost::mutex> activeLock(activeMutex);
    activeWriters--;
    if (activeWriters == 0)
        inactiveCondition.notify_one();
}

void DeviceWorkerGroup::freeItem(boost::shared_ptr<WorkItem> item)
{
    item->subItems.clear();
    item->copyEvent = cl::Event(); // release the reference
    itemPool.push(item);
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
    Timeplot::Action timer("compute", getTimeplotWorker(), owner.getComputeStat());
    BOOST_FOREACH(const SubItem &sub, work.subItems)
    {
        cl_uint3 keyOffset;
        for (int i = 0; i < 3; i++)
            keyOffset.s[i] = sub.grid.getExtent(i).first;
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
            size[i] = sub.grid.numVertices(i);
        }

        /* We need to round up the octree size to a multiple of the granularity used for MLS. */
        Grid::size_type expandedSize[3];
        for (int i = 0; i < 3; i++)
            expandedSize[i] = roundUp(size[i], MlsFunctor::wgs[i]);

        filterChain.setOutput(owner.outputGenerator(sub.chunkId, getTimeplotWorker()));

        cl::Event treeBuildEvent;
        std::vector<cl::Event> wait(1);

        wait[0] = work.copyEvent;
        tree.enqueueBuild(queue, work.splats, sub.firstSplat, sub.numSplats,
                          expandedSize, offset, owner.subsampling, &wait, &treeBuildEvent);
        wait[0] = treeBuildEvent;

        input.set(offset, tree, owner.subsampling);
        marching.generate(queue, input, filterChain, size, keyOffset, &wait);

        tree.clearSplats();

        if (owner.progress != NULL)
            *owner.progress += sub.grid.numCells();
    }
}

FineBucketGroup::FineBucketGroup(
    std::size_t numWorkers,
    const std::vector<DeviceWorkerGroup *> &outGroups,
    std::size_t memCoarseSplats,
    std::size_t maxSplats,
    Grid::size_type maxCells,
    Grid::size_type microCells,
    std::size_t maxSplit)
:
    WorkerGroup<FineBucketGroup::WorkItem, FineBucketGroup::Worker, FineBucketGroup>(
        "bucket.fine",
        numWorkers),
    outGroups(outGroups),
    splatBuffer("mem.FineBucketGroup.splats", memCoarseSplats),
    maxSplats(maxSplats),
    maxCells(maxCells),
    microCells(microCells),
    maxSplit(maxSplit),
    progress(NULL),
    writeStat(Statistics::getStatistic<Statistics::Variable>("bucket.fine.write"))
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
}

void FineBucketGroupBase::Worker::operator()(WorkItem &work)
{
    Timeplot::Action timer("compute", getTimeplotWorker(), owner.getComputeStat());
    Statistics::Registry &registry = Statistics::Registry::getInstance();

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

    DeviceWorkerGroup::SubItem &outItem =
        outGroup->get(getTimeplotWorker(), work.numSplats);

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
    outItem.grid = grid;
    outItem.recursionState = work.recursionState;
    outItem.chunkId = work.chunkId;

    {
        Timeplot::Action writeTimer("write", getTimeplotWorker(), owner.getWriteStat());
        std::memcpy(outItem.splats, work.splats.get(), work.numSplats * sizeof(Splat));
    }

    registry.getStatistic<Statistics::Variable>("bucket.fine.splats").add(outItem.numSplats);
    registry.getStatistic<Statistics::Variable>("bucket.fine.size").add(grid.numCells());

    outGroup->push();
    owner.splatBuffer.free(work.splats);
}
