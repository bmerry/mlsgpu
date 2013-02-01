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
    push(tworker, item);
}


DeviceWorkerGroup::DeviceWorkerGroup(
    std::size_t numWorkers, std::size_t spare,
    OutputGenerator outputGenerator,
    const cl::Context &context, const cl::Device &device,
    std::size_t maxBucketSplats, Grid::size_type maxCells,
    std::size_t meshMemory,
    int levels, int subsampling, float boundaryLimit,
    MlsShape shape)
:
    Base("device", numWorkers),
    progress(NULL), outputGenerator(outputGenerator),
    context(context), device(device),
    maxBucketSplats(maxBucketSplats), maxCells(maxCells), meshMemory(meshMemory),
    subsampling(subsampling),
    copyQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE),
    itemPool(),
    popMutex(NULL),
    popCondition(NULL)
{
    for (std::size_t i = 0; i < numWorkers; i++)
    {
        addWorker(new Worker(*this, context, device, levels, boundaryLimit, shape, i));
    }
    const std::size_t items = numWorkers + spare;
    const std::size_t maxItemSplats = maxBucketSplats; // the same thing for now
    for (std::size_t i = 0; i < items; i++)
    {
        boost::shared_ptr<WorkItem> item = boost::make_shared<WorkItem>(context, maxItemSplats);
        itemPool.push(item);
    }
    unallocated_ = maxItemSplats * items;
}

void DeviceWorkerGroup::start(const Grid &fullGrid)
{
    this->fullGrid = fullGrid;
    Base::start();
}

bool DeviceWorkerGroup::canGet()
{
    return !itemPool.empty();
}

boost::shared_ptr<DeviceWorkerGroup::WorkItem> DeviceWorkerGroup::get(
    Timeplot::Worker &tworker, std::size_t numSplats)
{
    Timeplot::Action timer("get", tworker, getStat);
    timer.setValue(numSplats * sizeof(Splat));
    boost::shared_ptr<DeviceWorkerGroup::WorkItem> item = itemPool.pop();

    boost::lock_guard<boost::mutex> unallocatedLock(unallocatedMutex);
    unallocated_ -= numSplats;
    return item;
}

void DeviceWorkerGroup::freeItem(boost::shared_ptr<WorkItem> item)
{
    item->subItems.clear();
    item->copyEvent = cl::Event(); // release the reference

    if (popCondition != NULL)
    {
        boost::lock_guard<boost::mutex> popLock(*popMutex);
        itemPool.push(item);
        popCondition->notify_one();
    }
    else
        itemPool.push(item);
}

std::size_t DeviceWorkerGroup::unallocated()
{
    boost::lock_guard<boost::mutex> unallocatedLock(unallocatedMutex);
    return unallocated_;
}

CLH::ResourceUsage DeviceWorkerGroup::resourceUsage(
    std::size_t numWorkers, std::size_t spare,
    const cl::Device &device,
    std::size_t maxBucketSplats, Grid::size_type maxCells,
    std::size_t meshMemory,
    int levels)
{
    Grid::size_type block = maxCells + 1;

    CLH::ResourceUsage workerUsage;
    workerUsage += Marching::resourceUsage(
        device, block, block, block,
        MlsFunctor::wgs[2], meshMemory, MlsFunctor::wgs);
    workerUsage += SplatTreeCL::resourceUsage(device, levels, maxBucketSplats);

    const std::size_t maxItemSplats = maxBucketSplats; // the same thing for now
    CLH::ResourceUsage itemUsage;
    itemUsage.addBuffer(maxItemSplats * sizeof(Splat));
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
    queue(context, device, Statistics::isEventTimingEnabled() ? CL_QUEUE_PROFILING_ENABLE : 0),
    tree(context, device, levels, owner.maxBucketSplats),
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

        {
            boost::lock_guard<boost::mutex> unallocatedLock(owner.unallocatedMutex);
            owner.unallocated_ += sub.numSplats;
        }
    }
}

FineBucketGroup::FineBucketGroup(
    const std::vector<DeviceWorkerGroup *> &outGroups,
    std::size_t maxQueueSplats)
:
    WorkerGroup<FineBucketGroup::WorkItem, FineBucketGroup::Worker, FineBucketGroup>(
        "bucket.fine", 1),
    outGroups(outGroups),
    maxDeviceItemSplats(outGroups[0]->getMaxItemSplats()),
    splatBuffer("mem.FineBucketGroup.splats", maxQueueSplats * sizeof(Splat)),
    writeStat(Statistics::getStatistic<Statistics::Variable>("bucket.fine.write")),
    splatsStat(Statistics::getStatistic<Statistics::Variable>("bucket.fine.splats")),
    sizeStat(Statistics::getStatistic<Statistics::Variable>("bucket.fine.size"))
{
    addWorker(new Worker(*this, outGroups[0]->getContext(), outGroups[0]->getDevice()));
    BOOST_FOREACH(DeviceWorkerGroup *g, outGroups)
        g->setPopCondition(&popMutex, &popCondition);
}

FineBucketGroupBase::Worker::Worker(
    FineBucketGroup &owner, const cl::Context &context, const cl::Device &device)
    : WorkerBase("bucket.fine", 0), owner(owner),
    pinned("mem.FineBucketGroup.pinned", context, device, owner.maxDeviceItemSplats),
    bufferedItems("mem.FineBucketGroup.bufferedItems"),
    bufferedSplats(0)
{
}

void FineBucketGroupBase::Worker::flush()
{
    if (bufferedItems.empty())
        return;

    boost::unique_lock<boost::mutex> popLock(owner.popMutex);
    DeviceWorkerGroup *outGroup = NULL;
    while (true)
    {
        /* Try all devices for which we can pop immediately, and take the one that
         * seems likely to run out the soonest. It's a poor guess, but does at
         * least make sure that we always service totally idle devices before ones
         * that still have work queued.
         */
        std::size_t best = 0;
        BOOST_FOREACH(DeviceWorkerGroup *g, owner.outGroups)
        {
            if (g->canGet())
            {
                std::size_t u = g->unallocated();
                if (u >= best)
                {
                    best = u;
                    outGroup = g;
                }
            }
        }
        if (outGroup != NULL)
            break;

        // No spare slots. Wait until there is one
        {
            Timeplot::Action timer("get", getTimeplotWorker(), owner.outGroups[0]->getGetStat());
            owner.popCondition.wait(popLock);
        }
    }
    popLock.release()->unlock();

    // This should now never block
    boost::shared_ptr<DeviceWorkerGroup::WorkItem> item = outGroup->get(getTimeplotWorker(), bufferedSplats);
    item->subItems.swap(bufferedItems);
    outGroup->getCopyQueue().enqueueWriteBuffer(
        item->splats,
        CL_FALSE,
        0, bufferedSplats * sizeof(Splat),
        pinned.get(),
        NULL, &item->copyEvent);
    cl::Event copyEvent = item->copyEvent;
    outGroup->push(getTimeplotWorker(), item);

    /* Ensures that we can start refilling the pinned memory right away. Note
     * that this is not the same as doing a synchronous transfer, because we
     * are still overlapping the transfer with enqueuing the item.
     */
    {
        Timeplot::Action writeTimer("write", getTimeplotWorker(), owner.getWriteStat());
        writeTimer.setValue(bufferedSplats * sizeof(Splat));
        copyEvent.wait();
    }
    bufferedSplats = 0;
}

void FineBucketGroupBase::Worker::operator()(WorkItem &work)
{
    Timeplot::Action timer("compute", getTimeplotWorker(), owner.getComputeStat());
    timer.setValue(work.numSplats * sizeof(Splat));

    if (bufferedSplats + work.numSplats > owner.maxDeviceItemSplats)
        flush();

    std::memcpy(pinned.get() + bufferedSplats, work.getSplats(),
                work.numSplats * sizeof(Splat));
    DeviceWorkerGroup::SubItem subItem;
    subItem.chunkId = work.chunkId;
    subItem.grid = work.grid;
    subItem.numSplats = work.numSplats;
    subItem.firstSplat = bufferedSplats;
    bufferedItems.push_back(subItem);
    bufferedSplats += work.numSplats;

    owner.splatsStat.add(work.numSplats);
    owner.sizeStat.add(work.grid.numCells());

    owner.splatBuffer.free(work.splats);
}
