/**
 * @file
 *
 * Loads buckets of data from a SplatSet and passes it to a worker group.
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
#include "bucket_collector.h"
#include "timeplot.h"
#include "mesher.h"
#include "statistics.h"

/**
 * Load buckets from disk and pass to the device. It is expected to be fed by a
 * @ref BucketCollector, either directly or over a network.
 */
template<typename Splats, typename OutGroup>
class BucketLoader : public boost::noncopyable
{
private:
    typedef std::pair<SplatSet::splat_id, SplatSet::splat_id> range_type;

public:
    typedef void result_type;

    BucketLoader(std::size_t maxItemSplats, OutGroup &outGroup, Timeplot::Worker &tworker);

    /// Prepares for a pass
    void start(const Splats &super, const Grid &fullGrid);

    /// Callback for @ref BucketCollector
    void operator()(const Statistics::Container::vector<BucketCollector::Bin> &bins);
private:
    OutGroup &outGroup;
    Grid fullGrid;
    Timeplot::Worker &tworker;

    const Splats *super;
    /// Temporary storage for loading @ref ranges before turning back into individual buckets
    Statistics::Container::PODBuffer<Splat> splatBuffer;

    Statistics::Variable &computeStat;
    Statistics::Variable &loadStat;
    Statistics::Variable &writeStat;
};

template<typename Splats, typename OutGroup>
BucketLoader<Splats, OutGroup>::BucketLoader(
    std::size_t maxItemSplats, OutGroup &outGroup, Timeplot::Worker &tworker)
    :
    outGroup(outGroup),
    tworker(tworker),
    super(NULL),
    splatBuffer("mem.BucketLoader.splatBuffer"),
    computeStat(Statistics::getStatistic<Statistics::Variable>("bucket.loader.compute")),
    loadStat(Statistics::getStatistic<Statistics::Variable>("bucket.loader.load")),
    writeStat(Statistics::getStatistic<Statistics::Variable>("bucket.loader.write"))
{
    splatBuffer.reserve(maxItemSplats);
}

template<typename Splats, typename OutGroup>
void BucketLoader<Splats, OutGroup>::operator()(const Statistics::Container::vector<BucketCollector::Bin> &bins)
{
    if (bins.empty())
        return;

    Statistics::Container::vector<range_type> ranges("mem.BucketLoader.ranges");
    {
        Timeplot::Action timer("compute", tworker, computeStat);
        /* Compute merged ranges */
        BOOST_FOREACH(const BucketCollector::Bin &bin, bins)
        {
            Statistics::Container::vector<range_type> tmp("mem.BucketLoader.ranges");
            SplatSet::merge(bin.ranges.begin(), bin.ranges.end(),
                            ranges.begin(), ranges.end(), std::back_inserter(tmp));
            tmp.swap(ranges);
        }
    }

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
    BOOST_FOREACH(const BucketCollector::Bin &bin, bins)
    {
        /* We transformed splats from world space into fullGrid space, so we need to
         * construct a new grid for this coordinate system.
         */
        const float ref[3] = {0.0f, 0.0f, 0.0f};
        Grid subGrid(ref, 1.0f, 0, 1, 0, 1, 0, 1);
        for (unsigned int i = 0; i < 3; i++)
        {
            Grid::difference_type base = fullGrid.getExtent(i).first;
            Grid::difference_type low = bin.grid.getExtent(i).first - base;
            Grid::difference_type high = bin.grid.getExtent(i).second - base;
            subGrid.setExtent(i, low, high);
        }

        boost::shared_ptr<typename OutGroup::WorkItem> item = outGroup.get(tworker, bin.ranges.numSplats());
        item->chunkId = bin.chunkId;
        item->grid = subGrid;

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
}

template<typename Splats, typename OutGroup>
void BucketLoader<Splats, OutGroup>::start(const Splats &super, const Grid &fullGrid)
{
    this->fullGrid = fullGrid;
    this->super = &super;
}

#endif /* !COARSE_BUCKET_H */
