/**
 * @file
 *
 * Bucketing of splats into sufficiently small buckets.
 */

#ifndef BUCKET_H
#define BUCKET_H

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <vector>
#include "tr1_cstdint.h"
#include <stdexcept>
#include <boost/function.hpp>
#include <boost/array.hpp>
#include "splat.h"
#include "grid.h"
#include "fast_ply.h"
#include "progress.h"
#include "splat_set.h"

/**
 * Bucketing of large numbers of splats into blocks.
 */
namespace Bucket
{

/**
 * Error that is thrown if too many splats cover a single cell, making it
 * impossible to satisfy the splat limit.
 */
class DensityError : public std::runtime_error
{
private:
    std::tr1::uint64_t cellSplats;   ///< Number of splats covering the affected cell

public:
    DensityError(std::tr1::uint64_t cellSplats) :
        std::runtime_error("Too many splats covering one cell"),
        cellSplats(cellSplats) {}

    std::tr1::uint64_t getCellSplats() const { return cellSplats; }
};

/**
 * Tracking of state across recursive calls.
 * This class has no impact on the algorithm, and exists for tracking metrics
 * and progress. It is used in several places:
 *  -# It is passed between calls to @ref detail::bucketRecurse to track
 *     statistics;
 *  -# It is passed to the processing callback so that it can update a
 *     progress meter if desired;
 *  -# It may be passed into @ref bucket in which case it is used as the
 *     initial state. The intended use is when the processor function
 *     makes a recursive call back into @ref bucket.
 */
struct Recursion
{
    unsigned int depth;              ///< Current depth of recursion.
    std::size_t totalRanges;         ///< Blob ranges held in memory at all levels.
    boost::array<Grid::size_type, 3> chunk; ///< Output file chunk.

    Recursion() : depth(0), totalRanges(0)
    {
        for (unsigned int i = 0; i < 3; i++)
            chunk[i] = 0;
    }
};

/**
 * Type-class for callback function called by @ref bucket. The parameters are:
 *  -# The splat collection.
 *  -# A grid covering the spatial extent of the bucket.
 *  -# A count of the number of grid cells already processed (before this one).
 *     The intended use is for progress meters.
 * It is guaranteed that the number of splats will be non-zero (empty buckets
 * are skipped). All splats that intersect the bucket will be passed, but
 * the intersection test is conservative so there may be extras.
 */
template<typename Splats>
class ProcessorType
{
public:
    /// The actual type.
    typedef boost::function<void(
        const typename SplatSet::Traits<Splats>::subset_type &,
        const Grid &,
        const Recursion &recursionState)> type;
};

/**
 * Subdivide a grid and the splats it contains into buckets with a maximum size
 * and splat count, and call a user callback function for each. This function
 * is designed to operate out-of-core and so very large inputs can be used.
 *
 * @param splats     The backing store of splats.
 * @param region     The region to process.
 * @param maxSplats  The maximum number of splats that may occur in a bucket.
 * @param maxCells   The maximum side length of a bucket, in grid cells.
 * @param chunkCells The output regions will be aligned on a regular grid of
 *                   approximately this granularity ("approximately" because it
 *                   will be rounded up to a multiple of the microblock size).
 *                   To disable this chunking, pass 0.
 * @param microCells The side length of a microblock. The implementation may
 *                   use coarser microCells if forced by @a maxSplit, but if so
 *                   it will scale up by a power of 2. This may also be zero to
 *                   fall back to a heuristic choice.
 * @param maxSplit   Maximum recursion fan-out. Larger values will usually
 *                   give higher performance by reducing recursion depth,
 *                   but at the cost of more memory.
 * @param process    Processing function called for each non-empty bucket.
 * @param progress   If specified, a progress display that will be incremented
 *                   for each empty cell that is skipped. It does not increment
 *                   for returned buckets, as that should be done as the final
 *                   processing on the bucket is done.
 * @param recursionState Optional parameter indicating recursion statistics
 *                   on entry. This is intended for use when the processing
 *                   callback calls this function again.
 *
 * @throw DensityError If any single grid cell conservatively intersects more
 *                     than @a maxSplats splats.
 *
 * @note If any splat falls completely outside of @a region, it is undefined
 * whether it will be passed to the processing function at all.
 *
 * @see @ref ProcessorType.
 *
 * @internal
 * The algorithm works recursively. First, some terminology:
 *  - @b Cell: A cube whose side length is defined by the spacing of @ref Grid.
 *  All grids used at all layers of recursion use the same spacing, so this is
 *  a universal unit.
 *  - @b Region: A cuboid aligned to the cell grid passed to @ref bucketRecurse.
 *  Each call to bucketRecurse either passes its region directly to the processing
 *  callback, or splits its region into a number of subregions for recursive
 *  processing.
 *  - @b Microblock: The smallest unit into which a single level of recursion will
 *  subdivide its region. The final subregions chosen are formed from collections
 *  of microblocks (specifically, nodes - see below). A microblock is a cube of
 *  cells.
 *  - @b Node: an octree node from an octree in which the leaves are microblocks.
 *  - @b Bucket: a region that forms a leaf of the recursion.
 *
 * At each level of recursion, it takes the current region and subdivides it into
 * microblocks. Microblocks are chosen to be as small as possible (subject
 * to @a maxSplit), but not smaller than determined by @a maxCells unless
 * the region is already that small. Of course, if on entry to the
 * recursion the region is suitable for processing this is done immediately.
 *
 * The microblocks are arranged in an implicit, dense octree. The splats
 * are then processed in several passes:
 *  -# Each splat is accumulated into per-node counters. A delta encoding is
 *     used so that small splats (those that intersect only one microblock)
 *     only require one modification to the data structure, instead of one per
 *     level.
 *  -# The octree is walked top-down to identify subregions.  A node is chosen
 *     as a subregion if it satisfies @a maxCells and @a maxSplats, or if it is a
 *     microblock. Otherwise it is subdivided.
 *  -# The splats are processed again to enter them into per-subregion buckets.
 *     A single splat can be placed into multiple buckets if it straddles
 *     subregion borders.
 * The subregions are then processed recursively.
 */
template<typename Splats>
void bucket(const Splats &splats,
            const Grid &region,
            std::tr1::uint64_t maxSplats,
            Grid::size_type maxCells,
            Grid::size_type chunkCells,
            Grid::size_type microCells,
            std::size_t maxSplit,
            const typename ProcessorType<Splats>::type &process,
            ProgressMeter *progress = NULL,
            const Recursion &recursionState = Recursion());

} // namespace Bucket

#include "bucket_impl.h"

#endif /* BUCKET_H */
