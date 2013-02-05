/**
 * @file
 *
 * Accumulate buckets to process as a batch.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <boost/function.hpp>
#include <boost/ref.hpp>
#include "splat_set.h"
#include "statistics.h"
#include "allocator.h"
#include "timeplot.h"
#include "chunk_id.h"
#include "bucket.h"
#include "bucket_collector.h"

BucketCollector::BucketCollector(SplatSet::splat_id maxSplats, Functor functor)
    : maxSplats(maxSplats), functor(functor),
    bins("mem.BucketCollector.bins"), numSplats(0),
    binsStat(Statistics::getStatistic<Statistics::Variable>("bucket.collector.bins")),
    splatsStat(Statistics::getStatistic<Statistics::Variable>("bucket.collector.splats"))
{
}

void BucketCollector::operator()(
    const SplatSet::SubsetBase &splats,
    const Grid &grid,
    const Bucket::Recursion &recursionState)
{
    if (numSplats + splats.numSplats() > maxSplats)
        flush();

    if (recursionState.chunk != curChunkId.coords)
    {
        curChunkId.gen++;
        curChunkId.coords = recursionState.chunk;
    }

    bins.push_back(Bin());
    Bin &bin = bins.back();
    bin.ranges = splats;
    bin.grid = grid;
    bin.chunkId = curChunkId;

    numSplats += splats.numSplats();
}

void BucketCollector::flush()
{
    if (bins.empty())
        return;

    binsStat.add(bins.size());
    splatsStat.add(numSplats);

    boost::unwrap_ref(functor)(bins);

    bins.clear();
    numSplats = 0;
}
