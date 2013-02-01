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
        std::size_t maxItemSplats,
        OutGroup &outGroup, Timeplot::Worker &tworker);

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

    const std::size_t maxItemSplats;
    OutGroup &outGroup;
    ChunkId curChunkId;
    Grid fullGrid;
    Timeplot::Worker &tworker;

    const Splats *super;
    /// Bins that have been received from bucketing but not yet loaded
    Statistics::Container::list<Bin> bins;
    /// Union of the ranges stored in @ref bins
    Statistics::Container::vector<range_type> ranges;
    /// Temporary storage for loading @ref ranges before turning back into individual buckets
    Statistics::Container::PODBuffer<Splat> splatBuffer;

    Statistics::Variable &loadStat;
    Statistics::Variable &writeStat;
    Statistics::Variable &binsStat;

    /// Number of splats buffers in @ref bins
    std::size_t numSplats() const;

    /// Flush the data in @ref bins and @ref ranges to the output group
    void flush();
};

template<typename Splats, typename OutGroup>
CoarseBucket<Splats, OutGroup>::CoarseBucket(
    std::size_t maxItemSplats,
    OutGroup &outGroup,
    Timeplot::Worker &tworker)
    :
    maxItemSplats(maxItemSplats),
    outGroup(outGroup),
    tworker(tworker),
    super(NULL),
    bins("mem.CoarseBucket.bins"),
    ranges("mem.CoarseBucket.ranges"),
    splatBuffer("mem.CoarseBucket.splatBuffer"),
    loadStat(Statistics::getStatistic<Statistics::Variable>("bucket.coarse.load")),
    writeStat(Statistics::getStatistic<Statistics::Variable>("bucket.coarse.write")),
    binsStat(Statistics::getStatistic<Statistics::Variable>("bucket.coarse.bins"))
{
    splatBuffer.reserve(maxItemSplats);
}

template<typename Splats, typename OutGroup>
void CoarseBucket<Splats, OutGroup>::operator()(
    const typename SplatSet::Traits<Splats>::subset_type &splats,
    const Grid &grid, const Bucket::Recursion &recursionState)
{
    if (numSplats() + splats.numSplats() > maxItemSplats)
        flush();

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
}

template<typename Splats, typename OutGroup>
std::size_t CoarseBucket<Splats, OutGroup>::numSplats() const
{
    std::size_t totalSplats = 0;
    BOOST_FOREACH(const range_type &range, ranges)
    {
        totalSplats += range.second - range.first;
    }
    return totalSplats;
}

template<typename Splats, typename OutGroup>
void CoarseBucket<Splats, OutGroup>::flush()
{
    if (bins.empty())
        return;

    binsStat.add(bins.size());

    {
        Timeplot::Action timer("load", tworker, loadStat);
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
        }
    }

    // Now process each bin, copying the relevant subset to the device
    BOOST_FOREACH(const Bin &bin, bins)
    {
        boost::shared_ptr<typename OutGroup::WorkItem> item = outGroup.get(tworker, bin.ranges.numSplats());
        item->chunkId = bin.chunkId;
        item->grid = bin.grid;

        Timeplot::Action timer("write", tworker, writeStat);
        timer.setValue(bin.ranges.numSplats() * sizeof(Splat));

        Statistics::Container::vector<range_type>::const_iterator p = ranges.begin();
        std::size_t pos = 0;
        Splat *splatPtr = (Splat *) item->getSplats();
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
        outGroup.push(tworker, item);
    }

    ranges.clear();
    bins.clear();
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
