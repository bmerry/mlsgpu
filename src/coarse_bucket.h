/**
 * @file
 *
 * Handles coarse-grained bucketing for external storage.
 */

#ifndef COARSE_BUCKET_H
#define COARSE_BUCKET_H

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <boost/smart_ptr/shared_ptr.hpp>
#include <boost/smart_ptr/scoped_ptr.hpp>
#include <boost/noncopyable.hpp>
#include <vector>
#include <list>
#include <cassert>
#include <cstring>
#include "clh.h"
#include "splat_set.h"
#include "grid.h"
#include "bucket.h"
#include "timeplot.h"
#include "mesher.h"
#include "statistics.h"

/**
 * Handles coarse-level bucketing from external storage. Unlike @ref
 * DeviceWorkerGroupBase::Worker and @ref FineBucketGroupBase::Worker, there
 * is only expected to be one of these, and it does not run in a separate
 * thread. It produces coarse buckets, read the splats into memory and pushes
 * the results to a @ref FineBucketGroup.
 */

template<typename Splats, typename OutGroup>
class CoarseBucket : public boost::noncopyable
{
private:
    typedef typename SplatSet::Traits<Splats>::subset_type subset_type;
    typedef std::pair<SplatSet::splat_id, SplatSet::splat_id> range_type;

public:
    void operator()(
        const subset_type &splatSet,
        const Grid &grid,
        const Bucket::Recursion &recursionState);

    CoarseBucket(
        const cl::Context &context, const cl::Device &device,
        std::size_t maxSplats,
        const std::vector<OutGroup *> &outGroups, Timeplot::Worker &tworker);

    /// Prepares for a pass
    void start(const Splats &super, const Grid &fullGrid);

    /// Ends a pass
    void stop();
private:
    struct Bin
    {
        SplatSet::SubsetBase ranges;
        ChunkId chunkId;
        Grid grid;
    };

    const std::size_t maxSplats;
    const std::vector<OutGroup *> outGroups;
    ChunkId curChunkId;
    Grid fullGrid;
    Timeplot::Worker &tworker;

    const Splats *super;
    std::size_t outSplats;    ///< Number of splats contained in bins (including duplicates)
    Statistics::Container::list<Bin> bins;
    Statistics::Container::vector<range_type> ranges;
    Statistics::Container::PODBuffer<Splat> splatBuffer;
    CLH::PinnedMemory<Splat> splatPinned;
    cl::Event copyEvent;      ///< Event signaled when copy from @ref splatPinned completes

    /// Pick the least-busy output group
    OutGroup *getOutGroup() const;

    /// Flush the data in @ref bins and @ref ranges to the output group
    void flush();
};

template<typename Splats, typename OutGroup>
CoarseBucket<Splats, OutGroup>::CoarseBucket(
    const cl::Context &context, const cl::Device &device,
    std::size_t maxSplats,
    const std::vector<OutGroup *> &outGroups,
    Timeplot::Worker &tworker)
    :
    maxSplats(maxSplats),
    outGroups(outGroups),
    tworker(tworker),
    super(NULL),
    outSplats(0),
    bins("mem.CoarseBucket.bins"),
    ranges("mem.CoarseBucket.ranges"),
    splatBuffer("mem.CoarseBucket.splatBuffer"),
    splatPinned("mem.CoarseBucket.splatPinned", context, device, maxSplats)
{
    splatBuffer.reserve(maxSplats);
}

template<typename Splats, typename OutGroup>
void CoarseBucket<Splats, OutGroup>::operator()(
    const typename SplatSet::Traits<Splats>::subset_type &splats,
    const Grid &grid, const Bucket::Recursion &recursionState)
{
    assert(splats.numSplats() <= maxSplats);
    if (outSplats + splats.numSplats() > maxSplats)
        flush();
    outSplats += splats.numSplats();

    if (recursionState.chunk != curChunkId.coords)
    {
        curChunkId.gen++;
        curChunkId.coords = recursionState.chunk;
    }

    /* The host transformed splats from world space into fullGrid space, so we need to
     * construct a new grid for this coordinate system.
     */
    const float ref[3] = {0.0f, 0.0f, 0.0f};
    Grid subGrid(ref, 1.0f, 0, 1, 0, 1, 0, 1);
    for (unsigned int i = 0; i < 3; i++)
    {
        Grid::difference_type base = fullGrid.getExtent(i).first;
        Grid::difference_type low = grid.getExtent(i).first - base;
        Grid::difference_type high = grid.getExtent(i).second - base;
        subGrid.setExtent(i, low, high);
    }

    bins.push_back(Bin());
    Bin &bin = bins.back();
    bin.ranges = splats;
    bin.grid = subGrid;
    bin.chunkId = curChunkId;

    Statistics::Container::vector<range_type> oldRanges("mem.CoarseBucket.ranges");
    oldRanges.swap(ranges);
    SplatSet::merge(oldRanges.begin(), oldRanges.end(),
                    splats.begin(), splats.end(),
                    std::back_inserter(ranges));

    Statistics::Registry &registry = Statistics::Registry::getInstance();
    registry.getStatistic<Statistics::Variable>("bucket.coarse.splats").add(splats.numSplats());
    registry.getStatistic<Statistics::Variable>("bucket.coarse.ranges").add(splats.numRanges());
    registry.getStatistic<Statistics::Variable>("bucket.coarse.size").add
        (double(grid.numCells(0)) * grid.numCells(1) * grid.numCells(2));
}

template<typename Splats, typename OutGroup>
OutGroup *CoarseBucket<Splats, OutGroup>::getOutGroup() const
{
    OutGroup *outGroup = NULL;
    std::size_t bestSpare = 0;
    BOOST_FOREACH(OutGroup *w, outGroups)
    {
        std::size_t spare = w->unallocated();
        if (spare >= bestSpare)
        {
            // Note: >= above so that we always get a non-NULL result
            outGroup = w;
            bestSpare = spare;
        }
    }
    assert(outGroup != NULL);
    return outGroup;
}

template<typename Splats, typename OutGroup>
void CoarseBucket<Splats, OutGroup>::flush()
{
    if (bins.empty())
        return;

    {
        // TODO: cache the statistics references passed to timeplot
        Timeplot::Action timer("load", tworker, "bucket.coarse.load");
        std::size_t pos = 0;
        boost::scoped_ptr<SplatSet::SplatStream> splatStream(super->makeSplatStream(ranges.begin(), ranges.end()));
        float invSpacing = 1.0f / fullGrid.getSpacing();
        while (!splatStream->empty())
        {
            Splat splat = **splatStream;
            /* Transform the splats into the grid's coordinate system */
            fullGrid.worldToVertex(splat.position, splat.position);
            splat.radius *= invSpacing;
            splatBuffer[pos++] = splat;
            ++*splatStream;
            assert(pos <= maxSplats);
        }
    }

    // Make sure any previous use of the pinned memory has completed
    if (copyEvent())
        copyEvent.wait();
    // Now process each bin, copying the relevant subset to the pinned memory
    {
        Timeplot::Action timer("write", tworker, "bucket.coarse.write");
        Splat *splatPtr = splatPinned.get();
        BOOST_FOREACH(const Bin &bin, bins)
        {
            std::size_t pos = 0;
            Statistics::Container::vector<range_type>::const_iterator p = ranges.begin();
            for (SplatSet::SubsetBase::const_iterator q = bin.ranges.begin(); q != bin.ranges.end(); ++q)
            {
                while (p->second < q->second)
                {
                    pos += p->second - p->first;
                    ++p;
                }
                assert(p->first <= q->first && p->second >= q->second);
                std::memcpy(splatPtr, &splatBuffer[pos + (q->first - p->first)],
                            (q->second - q->first) * sizeof(Splat));
                splatPtr += q->second - q->first;
            }
        }
    }

    // Select the least-busy output group to target, and fill in its item
    OutGroup *outGroup = getOutGroup();
    boost::shared_ptr<typename OutGroup::WorkItem> item = outGroup->get(tworker, outSplats);
    outGroup->getCopyQueue().enqueueWriteBuffer(
        item->splats,
        CL_FALSE,
        0, outSplats * sizeof(Splat),
        splatPinned.get(),
        NULL, &copyEvent);
    outGroup->getCopyQueue().flush();
    item->copyEvent = copyEvent;

    std::size_t pos = 0;
    BOOST_FOREACH(const Bin &bin, bins)
    {
        typename OutGroup::SubItem sub;
        sub.chunkId = bin.chunkId;
        sub.grid = bin.grid;
        sub.firstSplat = pos;
        sub.numSplats = bin.ranges.numSplats();
        pos += sub.numSplats;
        item->subItems.push_back(sub);
    }
    assert(pos == outSplats);
    outGroup->push(item);

    ranges.clear();
    bins.clear();
    outSplats = 0;
}

template<typename Splats, typename OutGroup>
void CoarseBucket<Splats, OutGroup>::start(const Splats &super, const Grid &fullGrid)
{
    this->fullGrid = fullGrid;
    this->super = &super;
}

template<typename Splats, typename OutGroup>
void CoarseBucket<Splats, OutGroup>::stop()
{
    flush();
}

#endif /* !COARSE_BUCKET_H */
