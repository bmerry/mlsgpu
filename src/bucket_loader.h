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
#include <boost/noncopyable.hpp>
#include <utility>
#include <cstring>
#include <cstddef>
#include "grid.h"
#include "bucket_collector.h"
#include "allocator.h"

class CopyGroup;
namespace SplatSet { class FileSet; }
namespace Statistics { class Variable; }
namespace Timeplot { class Worker; }

/**
 * Load buckets from disk and pass to the device. It is expected to be fed by a
 * @ref BucketCollector, either directly or over a network.
 */
class BucketLoader : public boost::noncopyable
{
private:
    typedef SplatSet::FileSet Splats;
    typedef std::pair<SplatSet::splat_id, SplatSet::splat_id> range_type;

public:
    typedef void result_type;

    BucketLoader(std::size_t maxItemSplats, CopyGroup &outGroup, Timeplot::Worker &tworker);

    /// Prepares for a pass
    void start(const Splats &super, const Grid &fullGrid);

    /// Callback for @ref BucketCollector
    void operator()(const Statistics::Container::vector<BucketCollector::Bin> &bins);
private:
    CopyGroup &outGroup;
    Grid fullGrid;
    Timeplot::Worker &tworker;

    const Splats *super;
    /// Temporary storage for loading combined ranges before turning back into individual buckets
    Statistics::Container::PODBuffer<Splat> splatBuffer;

    Statistics::Variable &computeStat;
    Statistics::Variable &loadStat;
    Statistics::Variable &writeStat;
};

#endif /* !COARSE_BUCKET_H */
