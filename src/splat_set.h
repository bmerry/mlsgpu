/**
 * @file
 *
 * Data structures for convenient and efficient iterations of collections of
 * collections of splats.
 */

#ifndef SPLAT_SET_H
#define SPLAT_SET_H

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <boost/noncopyable.hpp>
#include <boost/array.hpp>
#include <tr1/cstdint>
#include <ostream>
#include <cstddef>
#include <iterator>
#include "grid.h"
#include "splat.h"
#include "progress.h"

/**
 * Data structures for convenient and efficient iterations of collections of
 * collections of splats.
 */
namespace SplatSet
{

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

    /**
     * Attempts to merge this range with another range.
     * @param scan            The scan for the new range.
     * @param first,last      The new range.
     * @retval true if the ranges were successfully merged
     * @retval false otherwise (no change is made)
     */
    bool append(scan_type scan, index_type first, index_type last);
};

namespace detail
{

void splatToBuckets(const Splat &splat,
                    const Grid &grid, Grid::size_type bucketSize,
                    boost::array<Grid::difference_type, 3> &lower,
                    boost::array<Grid::difference_type, 3> &upper);

} // namespace detail

/**
 * A collection of collections of splats. Each collection is called a @em scan.
 * This is both a concrete class and a concept that is reimplemented by subclasses.
 * It provides mechanisms for iterating over all splats using an interface designed
 * for efficient bucketing.
 */
template<typename SplatCollectionSet>
class SimpleSet : public boost::noncopyable
{
public:
    typedef SplatCollectionSet value_type;
    typedef typename SplatCollectionSet::size_type scan_type;
    typedef typename std::iterator_traits<typename SplatCollectionSet::iterator>::value_type Collection;
    typedef typename Collection::size_type index_type;

    explicit SimpleSet(const SplatCollectionSet &splats);

    const SplatCollectionSet &getSplats() const { return splats; }

    /**
     * Call a function for each contiguous range of splats occupying the
     * same bucket. Note that it is not guaranteed that the contiguous
     * ranges are maximal; it is simply an interface that allows more efficient
     * computation if the callee can process batches faster than individual
     * elements.
     *
     * The function signature should be
     * <code>void(scan_type scan, size_type first, size_type last,
     *            const boost::array<Grid::difference_type, 3> &lower,
     *            const boost::array<Grid::difference_type, 3> &upper);</code>
     * The lower and upper bounds are inclusive and are in units of the
     * buckets given on input. The function will be called sequentially (i.e.,
     * in order) on the splats.
     *
     * Splats that do not intersect the grid might or might not be passed to
     * the callback. The same applies to non-finite splats.
     *
     * @param func        The function to call.
     * @param bucketSize  The number of grid cells per bucket (in each dimension).
     */
    template<typename Func>
    void forEach(const Grid &grid, Grid::size_type bucketSize, const Func &func) const;

    template<typename RangeIterator, typename Func>
    void forEachRange(RangeIterator first, RangeIterator last,
                      const Grid &grid, Grid::size_type bucketSize, const Func &func) const;

protected:
    /**
     * The raw splats themselves.
     */
    const SplatCollectionSet &splats;

private:
    template<typename Func>
    void forEachOne(
        scan_type scan, index_type index,
        const Grid &grid, Grid::size_type bucketSize,
        const Splat &splat, const Func &func) const;
};

/**
 * Entry in a list representing splat positions in a compacted form.
 * Contiguous ranges of splats from the scans are placed into buckets
 * (which size is stored in @ref SplatSetBlobbed). Each contiguous range must
 * intersect the same cuboid of buckets. There are two cases:
 *  -# They intersect a single bucket. The coordinates of the bucket
 *     (in units of buckets) are stored in @ref coords, and the number
 *     of splats is stored in @ref size.
 *  -# They intersect multiple buckets. Two consecutive blobs are used
 *     to encode the range. The first has the lower-bound (inclusive)
 *     coordinates, and a @ref size of zero. The second has the
 *     upper-bound (inclusive) coordinates and the actual size of the
 *     range.
 * The coordinates are relative to the grid reference point rather than
 * the edge of the grid, and so need correcting before they can be useful.
 * The lower grid extent is deliberately chosen to be a multiple of the
 * bucket size so that alignment is not an issue.
 */
struct Blob
{
    typedef std::tr1::uint32_t size_type;

    boost::array<Grid::difference_type, 3> coords;
    std::tr1::uint32_t size;
};

/**
 * Contains a list of splat collections, together with metadata
 * that allows the splats to be bucketed more efficiently. It also computes
 * a bounding box during construction.
 *
 * For @ref forEach to be efficient, the provided grid must
 *  - have the same reference point and spacing as the grid given by @ref getBounding;
 *  - have a lower extent which is a multiple of the bucket size in each dimension.
 *
 * @param SplatCollectionSet A random access container of @ref Collection of @ref Splat.
 * @param BlobCollection A forward container of @ref Blob.
 */
template<typename SplatCollectionSet, typename BlobCollection>
class BlobSet : public SimpleSet<SplatCollectionSet>
{
public:
    typedef typename SimpleSet<SplatCollectionSet>::scan_type scan_type;
    typedef typename SimpleSet<SplatCollectionSet>::index_type index_type;
    typedef typename SimpleSet<SplatCollectionSet>::Collection Collection;

    /**
     * Constructor. This will make a pass over all the splats to compute
     * the bounding box grid and populate the blob bucket.
     *
     * @param splats          The input splats.
     * @param spacing         The grid spacing for the final reconstruction.
     * @param blobBucket      The number of grid cells per bucket for blobs.
     * @param progressStream  If non-NULL, will be used to log progress through the splats.
     */
    BlobSet(const SplatCollectionSet &splats, float spacing, Grid::size_type blobBucket,
             std::ostream *progressStream = NULL);

    template<typename Func>
    void forEach(const Grid &grid, Grid::size_type bucketSize, const Func &func) const;

    template<typename RangeIterator, typename Func>
    void forEachRange(RangeIterator first, RangeIterator last,
                      const Grid &grid, Grid::size_type bucketSize, const Func &func) const;

    const Grid &getBoundingGrid() const { return boundingGrid; }

private:
    /// Data only needed during construction
    struct Build
    {
        typename SplatCollectionSet::size_type nonFinite;

        float bboxMin[3];
        float bboxMax[3];

        boost::array<Grid::difference_type, 3> blobLower, blobUpper;
        Blob::size_type blobSize;
    };

    Grid boundingGrid;

    /**
     * The number of grid cells per bucket for decoding the blobs in @ref blobs.
     */
    Grid::size_type blobBucket;

    /**
     * A list of blobs. The blobs can only be sensibly processed sequentially,
     * because this is the only way to maintain which splats each blob refers
     * to. Each blob refers to a range of splats immediately following the
     * previous blob. The blob collection will cover all scans in @ref splats,
     * but an individual blob does not span scan boundaries.
     */
    BlobCollection blobs;

    void flushBlob(Build &build);

    void processSplat(const Splat &splat, Build &build, ProgressDisplay *progress);

    bool fastGrid(const Grid &grid, Grid::size_type bucketSize) const;
};

} // namespace SplatSet

#include "splat_set_impl.h"

#endif /* !SPLAT_SET_H */
