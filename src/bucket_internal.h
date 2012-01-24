/**
 * @file
 *
 * Classes and functions in @ref bucket.cpp that are declared in a header so that
 * they can be tested. The declarations in this header should not be used
 * other than in @ref bucket.cpp and in test code.
 */

#ifndef BUCKET_INTERNAL_H
#define BUCKET_INTERNAL_H

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <tr1/cstdint>
#include <cstddef>
#include <boost/ref.hpp>
#include <boost/array.hpp>
#include "bucket.h"
#include "errors.h"
#include "fast_ply.h"
#include "splat.h"
#include "grid.h"

namespace Bucket
{

/**
 * Internal classes and functions for bucketing.
 */
namespace internal
{

/**
 * A cube with power-of-two side lengths.
 */
class Cell
{
public:
    /**
     * Type used to represent cell coordinates. Note that there is no point
     * in this being wider than the type used by @ref Grid.
     */
    typedef unsigned int size_type;

    /// Default constructor: does not initialize anything.
    Cell() {}

    /**
     * Constructor from an array.
     * @param base     Coordinates of minimum corner.
     * @param level    Log base two of the side length.
     * @pre The coordinates of the maximum corner do not overflow @ref size_type.
     */

    Cell(const size_type base[3], int level);

    /**
     * Constructor from scalars.
     * @param x,y,z    Coordinates of minimum corner.
     * @param level    Log base two of the side length.
     * @pre The coordinates of the maximum corner do not overflow @ref size_type.
     */
    Cell(size_type x, size_type y, size_type z, int level);

    /// Equality comparison operator
    bool operator==(const Cell &c) const;

    /// Get the log base 2 of the side length
    int getLevel() const { return level; }

    /// Get the side length
    size_type getSize() const { return size_type(1) << level; }

    /// Get the lower corner
    const size_type *getBase() const { return base; }

    /// Get the bounding corners
    void getCorners(size_type lower[3], size_type upper[3]) const;

private:
    size_type base[3];         ///< Coordinates of lower-left-bottom corner
    int level;                 ///< Log base two of the side length
};

/**
 * Tracks how many ranges are needed to encode a list of splats and
 * how many splats are in the list. It will match the actual number
 * written by @ref RangeCollector.
 */
class RangeCounter
{
private:
    std::tr1::uint64_t ranges;
    std::tr1::uint64_t splats;
    Range current;

public:
    /// Constructor
    RangeCounter();

    /// Adds a new splat to the virtual list.
    void append(Range::scan_type scan, Range::index_type splat);

    /// Returns the number of ranges that would be required to encode the provided splats.
    std::tr1::uint64_t countRanges() const;

    /// Returns the number of splats given to @ref append
    std::tr1::uint64_t countSplats() const;
};

/**
 * Accepts a list of splat IDs and merges them into ranges which are then
 * output.
 * @param OutputIterator an output iterator that accepts assignments of @ref Range.
 */
template<typename OutputIterator>
class RangeCollector
{
public:
    typedef OutputIterator iterator_type;

private:
    Range current;
    iterator_type out;

public:
    /**
     * Constructor.
     *
     * @param out     Output iterator to which completed ranges will be written.
     */
    RangeCollector(iterator_type out);

    /**
     * Destructor. It will flush any buffered ranges.
     */
    ~RangeCollector();

    /**
     * Adds a new splat to the list.
     *
     * @return The output iterator after the update.
     */
    iterator_type append(Range::scan_type scan, Range::index_type splat);

    /**
     * Force any buffered ranges to be emitted. This is done implicitly
     * by the destructor, so it is only necessary if there are more
     * ranges to be written later with the same object, or if lifetime
     * management makes it inconvenient to destroy the object.
     *
     * @return The output iterator after the update.
     */
    iterator_type flush();
};


template<typename OutputIterator>
RangeCollector<OutputIterator>::RangeCollector(iterator_type out)
    : current(), out(out)
{
}

template<typename OutputIterator>
RangeCollector<OutputIterator>::~RangeCollector()
{
    flush();
}

template<typename OutputIterator>
OutputIterator RangeCollector<OutputIterator>::append(Range::scan_type scan, Range::index_type splat)
{
    if (!current.append(scan, splat))
    {
        *out++ = current;
        current = Range(scan, splat);
    }
    return out;
}

template<typename OutputIterator>
OutputIterator RangeCollector<OutputIterator>::flush()
{
    if (current.size > 0)
    {
        *out++ = current;
        current = Range();
    }
    return out;
}

/**
 * Implementation detail of @ref forEachCell. Do not call this directly.
 *
 * @param dims      See @ref forEachCell.
 * @param cell      Current cell to process recursively.
 * @param func      See @ref forEachCell.
 */
template<typename Func>
void forEachCell_r(const Cell::size_type dims[3], const Cell &cell, const Func &func)
{
    if (func(cell))
    {
        if (cell.getLevel() > 0)
        {
            const Cell::size_type half = cell.getSize() / 2;
            for (int i = 0; i < 8; i++)
            {
                const Cell::size_type base[3] =
                {
                    cell.getBase()[0] + (i & 1 ? half : 0),
                    cell.getBase()[1] + (i & 2 ? half : 0),
                    cell.getBase()[2] + (i & 4 ? half : 0)
                };
                if (base[0] < dims[0] && base[1] < dims[1] && base[2] < dims[2])
                    forEachCell_r(dims, Cell(base, cell.getLevel() - 1), func);
            }
        }
    }
}

/**
 * Recursively walk an octree, calling a user-defined function on each cell.
 * The function takes a @ref Cell and returns a boolean indicating whether
 * the children should be visited in turn (ignored for leaves).
 *
 * The most refined level does not need to be power-of-two in size. Any octree
 * cells that do not at least partially intersect the rectangle defined by @a
 * dims will be skipped.
 *
 * @param dims     Dimensions of the most refined layer.
 * @param levels   Number of levels in the virtual octree.
 * @param func     User-provided functor.
 *
 * @pre
 * - @a levels is at least 1 and at most the number of bits in @c Cell::size_type
 * - The dimensions are each at most 2<sup>levels - 1</sup>.
 */
template<typename Func>
void forEachCell(const Cell::size_type dims[3], int levels, const Func &func)
{
    MLSGPU_ASSERT(levels >= 1 && levels <= std::numeric_limits<Cell::size_type>::digits, std::invalid_argument);
    int level = levels - 1;
    MLSGPU_ASSERT((Cell::size_type(1) << level) >= dims[0], std::invalid_argument);
    MLSGPU_ASSERT((Cell::size_type(1) << level) >= dims[1], std::invalid_argument);
    MLSGPU_ASSERT((Cell::size_type(1) << level) >= dims[2], std::invalid_argument);

    forEachCell_r(dims, Cell(0, 0, 0, level), func);
}


/**
 * Iterate over all splats given be a collection of @ref Range, calling
 * a user-provided function for each.
 *
 * @param files        Files references by the ranges.
 * @param first, last  %Range of @ref Range objects.
 * @param func         User-provided callback.
 */
template<typename Func>
void forEachSplat(
    const boost::ptr_vector<FastPly::Reader> &files,
    RangeConstIterator first,
    RangeConstIterator last,
    const Func &func)
{
    // Maximum number of splats we read from a range.
    static const std::size_t splatBufferSize = 8192;

    for (RangeConstIterator it = first; it != last; ++it)
    {
        const Range &range = *it;
        Splat buffer[splatBufferSize];
        Range::size_type size = range.size;
        Range::index_type start = range.start;
        while (size != 0)
        {
            Range::size_type chunkSize = size;
            if (splatBufferSize < size)
                chunkSize = splatBufferSize;
            files[range.scan].readVertices(start, chunkSize, buffer);
            for (std::size_t j = 0; j < chunkSize; j++)
            {
                boost::unwrap_ref(func)(range.scan, start + j, buffer[j]);
            }
            size -= chunkSize;
            start += chunkSize;
        }
    }
}

bool splatCellIntersect(const Splat &splat, const internal::Cell &cell, const Grid &grid);

} // namespace internal
} // namespace Bucket

#endif /* !BUCKET_INTERNAL_H */
