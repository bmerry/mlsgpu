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
#include <tr1/cstdint>
#include <stdexcept>
#include <boost/function.hpp>
#if HAVE_STXXL
# include <stxxl.h>
#endif
#include "splat.h"
#include "grid.h"
#include "collection.h"
#include "fast_ply.h"

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
 * Indexes a sequential range of splats from an input file.
 *
 * This is intended to be POD that can be put in a @c stxxl::vector.
 *
 * @invariant @ref start + @ref size - 1 does not overflow @ref index_type.
 * (maintained by constructor and by @ref append).
 */
struct Range
{
    /// Type used to index the list of files
    typedef std::tr1::uint32_t scan_type;
    /// Type used to specify the length of a range
    typedef std::tr1::uint32_t size_type;
    /// Type used to index a splat within a file
    typedef std::tr1::uint64_t index_type;

    /* Note: the order of these is carefully chosen for alignment */
    scan_type scan;    ///< Index of the originating file
    size_type size;    ///< Size of the range
    index_type start;  ///< Splat index in the file

    /**
     * Constructs an empty scan range.
     */
    Range();

    /**
     * Constructs a splat range with one splat.
     */
    Range(scan_type scan, index_type splat);

    /**
     * Constructs a splat range with multiple splats.
     *
     * @pre @a start + @a size - 1 must fit within @ref index_type.
     */
    Range(scan_type scan, index_type start, size_type size);

    /**
     * Attempts to extend this range with a new element.
     * @param scan, splat     The new element
     * @retval true if the element was successfully appended
     * @retval false otherwise.
     */
    bool append(scan_type scan, index_type splat);
};

/**
 * Type passed to @ref ProcessorType<CollectionSet>::type to delimit a range of ranges.
 */
typedef std::vector<Range>::const_iterator RangeConstIterator;

/**
 * Type-class for callback function called by @ref bucket. The parameters are:
 *  -# The backing store of splats.
 *  -# The number of splats in the bucket.
 *  -# A [first, last) pair indicating a range of splat ranges in the bucket
 *  -# A grid covering the spatial extent of the bucket.
 *  -# A count of the number of grid cells already processed (before this one).
 *     The intended use is for progress meters.
 * It is guaranteed that the number of splats will be non-zero (empty buckets
 * are skipped). All splats that intersect the bucket will be passed, but
 * the intersection test is conservative so there may be extras. The ranges
 * will be ordered by scan so all splats from one scan are contiguous.
 */
template<typename CollectionSet>
class ProcessorType
{
public:
    /// The actual type.
    typedef boost::function<void(
        const CollectionSet &,
        Range::index_type,
        RangeConstIterator,
        RangeConstIterator,
        const Grid &,
        std::tr1::uint64_t done)> type;
};

/**
 * Subdivide a grid and the splats it contains into buckets with a maximum size
 * and splat count, and call a user callback function for each. This function
 * is designed to operate out-of-core and so very large inputs can be used.
 *
 * @param splats     The backing store of splats. All splats are used.
 * @param bbox       A grid which must completely enclose all the splats.
 * @param maxSplats  The maximum number of splats that may occur in a bucket.
 * @param maxCells   The maximum side length of a bucket, in grid cells.
 * @param maxSplit   Maximum recursion fan-out. Larger values will usually
 *                   give higher performance by reducing recursion depth,
 *                   but at the cost of more memory.
 * @param process    Processing function called for each non-empty bucket.
 *
 * @throw DensityError If any single grid cell conservatively intersects more
 *                     than @a maxSplats splats.
 *
 * @note If any splat falls partially or completely outside of @a bbox, it
 * is undefined whether it will be passed to the processing functions.
 *
 * @see @ref ProcessorType.
 *
 * @internal
 * The algorithm works recursively. At each level of recursion, it takes the
 * current "cell" (which is a cuboid of grid cells), and subdivides it into
 * "microblocks". Microblocks are chosen to be as small as possible (subject
 * to @a maxSplit), but not smaller than determined by @a maxCells. Of course,
 * if on entry to the recursion the cell is suitable for processing this
 * is done immediately.
 *
 * The microblocks are arranged in an implicit, dense octree. The splats
 * are then processed in several passes:
 *  -# Each splat is accumulated into a counter for all octree nodes
 *     it intersects, to determine the size of the node should it be
 *     turned into a bucket.
 *  -# The octree is walked top-down to identify buckets for passing to
 *     the next level. A node is chosen if it satisfies @a maxCells and
 *     @a maxSplats, or if it is a microblock. Otherwise it is subdivided.
 *  -# Storage is allocated to hold all the splat ranges for the next
 *     level of the octree.
 *  -# The splats are reprocessed to place them into the storage (a single
 *     splat may be stored in multiple buckets, if it crosses boundaries).
 * The buckets are then processed recursively.
 */
template<typename CollectionSet>
void bucket(const CollectionSet &splats,
            const Grid &bbox,
            Range::index_type maxSplats,
            int maxCells,
            std::size_t maxSplit,
            const typename ProcessorType<CollectionSet>::type &process);

#if HAVE_STXXL
/**
 * Transfer splats into an @c stxxl::vector (optionally sorting) and
 * simultaneously compute a bounding grid. The resulting grid is suitable for
 * passing to @ref bucket.
 *
 * The grid is constructed as follows:
 *  -# The bounding box of the sample points is found, ignoring influence regions.
 *  -# The lower bound is used as the grid reference point.
 *  -# The grid extends are set to cover the full bounding box.
 *
 * @param[in]  files         PLY input files (already opened)
 * @param      spacing       The spacing between grid vertices.
 * @param[out] splats        Vector of loaded splats
 * @param[out] grid          Bounding grid.
 * @param      sort          Whether to sort the splats using @ref CompareSplatsMorton.
 *
 * @throw std::length_error if the files contain no splats.
 */
template<typename CollectionSet>
void loadSplats(const CollectionSet &files,
                float spacing,
                bool sort,
                StxxlVectorCollection<Splat>::vector_type &splats,
                Grid &grid);
#endif

/**
 * Compute a bounding grid for the splats. It uses the same
 * algorithm as @ref sortSplats, but does not make a copy of the splats.
 */
template<typename CollectionSet>
void makeGrid(const CollectionSet &files,
              float spacing,
              Grid &grid);

} // namespace Bucket

#include "bucket_impl.h"

#endif /* BUCKET_H */
