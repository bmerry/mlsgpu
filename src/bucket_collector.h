/**
 * @file
 *
 * Accumulate buckets to process as a batch.
 */

#ifndef BUCKET_COLLECTOR_H
#define BUCKET_COLLECTOR_H

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <boost/function.hpp>
#include "splat_set.h"
#include "statistics.h"
#include "allocator.h"
#include "timeplot.h"
#include "mesher.h"
#include "bucket.h"

/**
 * Receives multiple buckets from @ref Bucket::bucket and accumulates
 * them until the total number of splats reaches a threshold. It then
 * makes a callback with the collected results.
 *
 * It also assigns generation numbers to chunk IDs.
 */
class BucketCollector : public boost::noncopyable
{
public:
    struct Bin
    {
        SplatSet::SubsetBase ranges;
        ChunkId chunkId;
        Grid grid;
    };

    typedef boost::function<void(const Statistics::Container::vector<Bin> &bins)> Functor;

    void operator()(
        const SplatSet::SubsetBase &splats,
        const Grid &grid,
        const Bucket::Recursion &recursionState);

    /**
     * Constructor.
     *
     * @param maxSplats    Maximum total number of splats to pass to the functor
     * @param functor      Functor called with collected ranges
     */
    BucketCollector(SplatSet::splat_id maxSplats, Functor functor);

    void flush(); ///< Flush any partial bins to the output

private:
    ChunkId curChunkId;           ///< Last-seen chunk ID
    SplatSet::splat_id maxSplats; ///< Limit on splats to pass to @ref functor
    Functor functor;              ///< Callback function
    Statistics::Container::vector<Bin> bins;  ///< Buffer of splat ranges
    SplatSet::splat_id numSplats; ///< Splats collected in @ref bins

    Statistics::Variable &binsStat;   ///< Number of bins per flush
    Statistics::Variable &splatsStat; ///< Number of splats per flush
};

#endif /* !BUCKET_COLLECTOR_H */
