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

DeviceWorkerGroup::DeviceWorkerGroup(
    std::size_t numWorkers, std::size_t capacity,
    const Grid &fullGrid,
    const cl::Context &context, const cl::Device &device,
    std::size_t maxSplats, Grid::size_type maxCells,
    int levels, int subsampling, bool keepBoundary, float boundaryLimit)
:
    WorkerGroup<DeviceWorkerGroup::WorkItem, DeviceWorkerGroup::Worker>(
        numWorkers, capacity,
        Statistics::getStatistic<Statistics::Variable>("device.worker.push"),
        Statistics::getStatistic<Statistics::Variable>("device.worker.pop"),
        Statistics::getStatistic<Statistics::Variable>("device.worker.get")),
    progress(NULL), fullGrid(fullGrid),
    maxSplats(maxSplats), maxCells(maxCells), subsampling(subsampling)
{
    for (std::size_t i = 0; i < capacity; i++)
    {
        boost::shared_ptr<WorkItem> item = boost::make_shared<WorkItem>();
        item->splats = cl::Buffer(context, CL_MEM_READ_ONLY, maxSplats * sizeof(Splat));
        item->numSplats = 0;
        addPoolItem(item);
    }
    for (std::size_t i = 0; i < numWorkers; i++)
    {
        addWorker(new Worker(*this, context, device, levels, keepBoundary, boundaryLimit));
    }
}

CLH::ResourceUsage DeviceWorkerGroup::resourceUsage(
    std::size_t numWorkers, std::size_t capacity,
    const cl::Device &device,
    std::size_t maxSplats, Grid::size_type maxCells,
    int levels, bool keepBoundary)
{
    Grid::size_type block = maxCells + 1;
    std::size_t maxVertices = Marching::getMaxVertices(block, block);
    std::size_t maxTriangles = Marching::getMaxTriangles(block, block);
    CLH::ResourceUsage workerUsage;
    workerUsage += Marching::resourceUsage(device, block, block);
    workerUsage += SplatTreeCL::resourceUsage(device, levels, maxSplats);
    if (!keepBoundary)
        workerUsage += Clip::resourceUsage(device, maxVertices, maxTriangles);

    CLH::ResourceUsage itemUsage;
    itemUsage.addBuffer(maxSplats * sizeof(Splat));
    return workerUsage * numWorkers + itemUsage * capacity;
}

void DeviceWorkerGroup::setOutput(const Marching::OutputFunctor &output)
{
    for (std::size_t i = 0; i < numWorkers(); i++)
        getWorker(i).setOutput(output);
}

DeviceWorkerGroupBase::Worker::Worker(
    DeviceWorkerGroup &owner,
    const cl::Context &context, const cl::Device &device,
    int levels, bool keepBoundary, float boundaryLimit)
:
    owner(owner),
    queue(context, device),
    tree(context, levels, owner.maxSplats),
    input(context),
    marching(context, device, owner.maxCells + 1, owner.maxCells + 1),
    scaleBias(context)
{
    if (!keepBoundary)
    {
        input.setBoundaryLimit(boundaryLimit);
        clip.reset(new Clip(context, device,
                            marching.getMaxVertices(owner.maxCells + 1, owner.maxCells + 1),
                            marching.getMaxTriangles(owner.maxCells + 1, owner.maxCells + 1)));
        clip->setDistanceFunctor(input);
        filterChain.addFilter(boost::ref(*clip));
    }

    filterChain.addFilter(boost::ref(scaleBias));
    scaleBias.setScaleBias(owner.fullGrid);
}

void DeviceWorkerGroupBase::Worker::operator()(WorkItem &work)
{
    cl_uint3 keyOffset; 
    for (int i = 0; i < 3; i++)
        keyOffset.s[i] = work.grid.getExtent(i).first;
    // same thing, just as a different type for a different API
    Grid::difference_type offset[3] = { keyOffset.s[0], keyOffset.s[1], keyOffset.s[2] };

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
    for (int i = 0; i < 2; i++)
        expandedSize[i] = roundUp(size[i], MlsFunctor::wgs[i]);
    expandedSize[2] = size[2];

    // TODO: use mapping to transfer the data directly into a buffer

    {
        Statistics::Timer timer("device.worker.time");
        cl::Event treeBuildEvent;
        std::vector<cl::Event> wait(1);
        tree.enqueueBuild(queue, work.splats, work.numSplats,
                          expandedSize, offset, owner.subsampling, NULL, &treeBuildEvent);
        wait[0] = treeBuildEvent;

        input.set(expandedSize, offset, tree, owner.subsampling);
        marching.generate(queue, input, filterChain, size, keyOffset, &wait);
    }

    if (owner.progress != NULL)
        *owner.progress += work.grid.numCells();
}


FineBucketGroup::FineBucketGroup(
    std::size_t numWorkers, std::size_t capacity,
    DeviceWorkerGroup &outGroup,
    const Grid &fullGrid,
    const cl::Context &context, const cl::Device &device,
    std::size_t maxCoarseSplats,
    std::size_t maxSplats,
    Grid::size_type maxCells,
    std::size_t maxSplit)
:
    WorkerGroup<FineBucketGroup::WorkItem, FineBucketGroup::Worker>(
        numWorkers, capacity,
        Statistics::getStatistic<Statistics::Variable>("bucket.fine.push"),
        Statistics::getStatistic<Statistics::Variable>("bucket.fine.pop"),
        Statistics::getStatistic<Statistics::Variable>("bucket.fine.get")),
    outGroup(outGroup),
    fullGrid(fullGrid),
    maxSplats(maxSplats),
    maxCells(maxCells),
    maxSplit(maxSplit),
    progress(NULL)
{
    for (std::size_t i = 0; i < numWorkers; i++)
    {
        addWorker(new Worker(*this, context, device));
    }
    for (std::size_t i = 0; i < capacity; i++)
    {
        boost::shared_ptr<WorkItem> item = boost::make_shared<WorkItem>();
        item->splats.reserve(maxCoarseSplats);
        addPoolItem(item);
    }
}

FineBucketGroupBase::Worker::Worker(FineBucketGroup &owner, const cl::Context &context, const cl::Device &device)
    : owner(owner), queue(context, device)
{
};

void FineBucketGroupBase::Worker::operator()(
    const SplatSet::Traits<SplatSet::VectorSet>::subset_type &splatSet,
    const Grid &grid,
    const Bucket::Recursion &recursionState)
{
    Statistics::Registry &registry = Statistics::Registry::getInstance();

    boost::shared_ptr<DeviceWorkerGroup::WorkItem> outItem = owner.outGroup.get();
    outItem->numSplats = splatSet.numSplats();
    outItem->grid = grid;
    outItem->recursionState = recursionState;
    Splat *splats = static_cast<Splat *>(
        queue.enqueueMapBuffer(outItem->splats, CL_TRUE, CL_MAP_WRITE,
                               0, splatSet.numSplats() * sizeof(Splat)));


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

    queue.enqueueUnmapMemObject(outItem->splats, splats);
    queue.finish(); // TODO: see if this can be made asynchronous

    owner.outGroup.push(outItem);
}

void FineBucketGroup::Worker::operator()(WorkItem &work)
{
    Statistics::Timer timer("bucket.fine.exec");

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
    Bucket::bucket(work.splats, grid, owner.maxSplats, owner.maxCells, false, owner.maxSplit,
                   boost::ref(*this), owner.progress, work.recursionState);
}
