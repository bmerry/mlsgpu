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
public:
    void operator()(
        const typename SplatSet::Traits<Splats>::subset_type &splatSet,
        const Grid &grid,
        const Bucket::Recursion &recursionState);

    CoarseBucket(const std::vector<OutGroup *> &outGroups, Timeplot::Worker &tworker);

    /// Prepares for a pass
    void start(const Grid &fullGrid);

    /// Ends a pass
    void stop();
private:
    const std::vector<OutGroup *> outGroups;
    ChunkId curChunkId;
    Grid fullGrid;
    Timeplot::Worker &tworker;
};

template<typename Splats, typename OutGroup>
CoarseBucket<Splats, OutGroup>::CoarseBucket(const std::vector<OutGroup *> &outGroups, Timeplot::Worker &tworker)
: outGroups(outGroups), tworker(tworker)
{
}

template<typename Splats, typename OutGroup>
void CoarseBucket<Splats, OutGroup>::operator()(
    const typename SplatSet::Traits<Splats>::subset_type &splats,
    const Grid &grid, const Bucket::Recursion &recursionState)
{
    if (recursionState.chunk != curChunkId.coords)
    {
        curChunkId.gen++;
        curChunkId.coords = recursionState.chunk;
    }

    Statistics::Registry &registry = Statistics::Registry::getInstance();

    /* Select the least-busy output group to target */
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

    typename OutGroup::get_type item = outGroup->get(tworker, splats.numSplats());
    item->chunkId = curChunkId;
    item->grid = subGrid;
    float invSpacing = 1.0f / fullGrid.getSpacing();

    {
        Timeplot::Action timer("load", tworker, "bucket.coarse.load");

        boost::scoped_ptr<SplatSet::SplatStream> splatStream(splats.makeSplatStream());
        Splat *splatPtr = (Splat *) item->getSplats();
        while (!splatStream->empty())
        {
            Splat splat = **splatStream;
            /* Transform the splats into the grid's coordinate system */
            fullGrid.worldToVertex(splat.position, splat.position);
            splat.radius *= invSpacing;
            *splatPtr++ = splat;
            ++*splatStream;
        }

        registry.getStatistic<Statistics::Variable>("bucket.coarse.splats").add(splats.numSplats());
        registry.getStatistic<Statistics::Variable>("bucket.coarse.ranges").add(splats.numRanges());
        registry.getStatistic<Statistics::Variable>("bucket.coarse.size").add
            (double(grid.numCells(0)) * grid.numCells(1) * grid.numCells(2));
    }

    outGroup->push(item);
}

template<typename Splats, typename OutGroup>
void CoarseBucket<Splats, OutGroup>::start(const Grid &fullGrid)
{
    this->fullGrid = fullGrid;
}

template<typename Splats, typename OutGroup>
void CoarseBucket<Splats, OutGroup>::stop()
{
}

#endif /* !COARSE_BUCKET_H */
