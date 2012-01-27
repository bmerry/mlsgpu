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
#include <cstring>
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
 * Implementation of the STXXL stream concept that reads all splats
 * from a collection of PLY files.
 *
 * @param FileIterator an iterator type whose value type is @ref FastPly::Reader.
 */
template<typename FileIterator>
class FilesStream
{
private:
    FileIterator first, last;
    FileIterator current;
    FastPly::Reader::size_type position;
public:
    typedef Splat value_type;

    FilesStream();
    FilesStream(FileIterator first, FileIterator last);

    Splat operator*() const;
    FilesStream<FileIterator> &operator++();
    bool empty() const;
};

template<typename FileIterator>
FilesStream<FileIterator>::FilesStream() : first(), last(), current(last), position(0) {}

template<typename FileIterator>
FilesStream<FileIterator>::FilesStream(FileIterator first, FileIterator last)
    : first(first), last(last), current(first), position(0)
{
    while (current != last && current->numVertices() == 0)
        ++current;
}

template<typename FileIterator>
Splat FilesStream<FileIterator>::operator*() const
{
    assert(!empty());
    Splat ans;
    current->readVertices(position, 1, &ans);
    return ans;
}

template<typename FileIterator>
FilesStream<FileIterator> &FilesStream<FileIterator>::operator++()
{
    assert(!empty());
    ++position;
    while (current != last && position == current->numVertices())
    {
        position = 0;
        ++current;
    }
    return *this;
}

template<typename FileIterator>
bool FilesStream<FileIterator>::empty() const
{
    return current == last;
}

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
     * Constructor.
     *
     * @param lower    Coordinates of minimum corner.
     * @param upper    Coordinates of maximum corner.
     * @param level    Octree level.
     * @pre The cell has positive side lengths.
     */
    Cell(const size_type lower[3], const size_type upper[3], unsigned int level);

    /**
     * Constructor which is more convenient for building literals.
     *
     * @param lowerX,lowerY,lowerZ    Coordinates of minimum corner.
     * @param upperX,upperY,upperZ    Coordinates of maximum corner.
     * @param level                   Octree level.
     * @pre The cell has positive side lengths.
     */
    Cell(size_type lowerX, size_type lowerY, size_type lowerZ,
         size_type upperX, size_type upperY, size_type upperZ,
         unsigned int level);

    /// Equality comparison operator
    bool operator==(const Cell &c) const;

    /// Get the octree level
    unsigned int getLevel() const { return level; }

    /// Get the side length on an axis
    size_type getSize(int axis) const { return upper[axis] - lower[axis]; }

    /// Get the lower corner
    const size_type *getLower() const { return lower; }

    /// Get the upper corner
    const size_type *getUpper() const { return upper; }

    /**
     * Create a child cell in the octree.
     * @pre level > 0 and idx < 8.
     */
    Cell child(unsigned int idx) const;

private:
    size_type lower[3];         ///< Coordinates of lower-left-bottom corner
    size_type upper[3];         ///< Coordinates of upper-right-top corner
    /**
     * Octree level. This has no direct effect on the spatial extent of the
     * cell, but is used in finding corresponding indices in the octree. The
     * level is zero for the finest (microblock) level of the octree.
     */
    unsigned int level;
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
    void append(Range::index_type splat);

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
    iterator_type append(Range::index_type splat);

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
OutputIterator RangeCollector<OutputIterator>::append(Range::index_type splat)
{
    if (!current.append(splat))
    {
        *out++ = current;
        current = Range(splat);
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
            for (unsigned int i = 0; i < 8; i++)
            {
                Cell child = cell.child(i);
                if (child.getLower()[0] < dims[0]
                    && child.getLower()[1] < dims[1]
                    && child.getLower()[2] < dims[2])
                    forEachCell_r(dims, child, func);
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
 * @param dims      Dimensions of the most refined layer, in grid cells.
 * @param microSize Dimensions of the cubic cells in the most refined layer, in grid cells.
 * @param levels    Number of levels in the virtual octree.
 * @param func      User-provided functor.
 *
 * @pre
 * - @a levels is at least 1 and at most the number of bits in @c Cell::size_type
 * - The dimensions are each at most @a microSize * 2<sup>levels - 1</sup>.
 */
template<typename Func>
void forEachCell(const Cell::size_type dims[3], Cell::size_type microSize, unsigned int levels, const Func &func)
{
    MLSGPU_ASSERT(levels >= 1U
                  && levels <= (unsigned int) std::numeric_limits<Cell::size_type>::digits, std::invalid_argument);
    int level = levels - 1;
    MLSGPU_ASSERT((dims[0] - 1) >> level < microSize, std::invalid_argument);
    MLSGPU_ASSERT((dims[1] - 1) >> level < microSize, std::invalid_argument);
    MLSGPU_ASSERT((dims[2] - 1) >> level < microSize, std::invalid_argument);

    Cell::size_type size = microSize << level;
    forEachCell_r(dims, Cell(0, 0, 0, size, size, size, level), func);
}

/**
 * Overload that takes a grid instead of explicit dimensions. The dimensions are taken from
 * the number of cells along each dimension of the grid.
 */
template<typename Func>
void forEachCell(const Grid &grid, Cell::size_type microSize, unsigned int levels, const Func &func)
{
    const Cell::size_type dims[3];
    for (int i = 0; i < 3; i++)
        dims[i] = grid.numCells(i);
    forEachCell(dims, microSize, levels, func);
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
    const SplatVector &splats,
    RangeConstIterator first,
    RangeConstIterator last,
    const Func &func)
{
    for (RangeConstIterator it = first; it != last; ++it)
    {
        const Range &range = *it;
        SplatVector::const_iterator cur = splats.begin() + range.start;
        for (Range::size_type i = 0; i < range.size; ++i, ++cur)
        {
            boost::unwrap_ref(func)(range.start + i, *cur);
        }
    }
}

template<typename Func>
class ForEachSplatCell
{
private:
    const Grid &grid;
    Cell::size_type dims[3];
    Cell::size_type microSize;
    unsigned int levels;
    const Func &func;

    class PerSplat
    {
    private:
        Range::index_type id;
        const Splat &splat;
        const Func &func;
        float lower[3];      ///< Splat lower bound converted to grid coordinates
        float upper[3];      ///< Splat upper bound converted to grid coordinates

    public:
        PerSplat(const Grid &grid, Range::index_type id, const Splat &splat, const Func &func)
            : id(id), splat(splat), func(func)
        {
            float lo[3], hi[3];
            for (int i = 0; i < 3; i++)
            {
                lo[i] = splat.position[i] - splat.radius;
                hi[i] = splat.position[i] + splat.radius;
            }
            grid.worldToVertex(lo, lower);
            grid.worldToVertex(hi, upper);
        }

        bool operator()(const Cell &cell) const
        {
            for (int i = 0; i < 3; i++)
                if (upper[i] < cell.getLower()[i] || lower[i] > cell.getUpper()[i])
                    return false;
            return func(id, splat, cell);
        }
    };

public:
    ForEachSplatCell(const Grid &grid, Cell::size_type microSize, unsigned int levels, const Func &func)
        : grid(grid), microSize(microSize), levels(levels), func(func)
    {
        for (int i = 0; i < 3; i++)
            dims[i] = grid.numCells(i);
    }

    void operator()(Range::index_type id, const Splat &splat) const
    {
        PerSplat p(grid, id, splat, func);
        forEachCell(dims, microSize, levels, p);
    }
};

/**
 * Combination of @ref forEachSplat and @ref forEachCell that calls the user
 * function for cells that are intersected by the splat. As for @ref forEachCell,
 * the function may return @c false to prevent recursion into child cells.
 *
 * The function takes the arg
 *
 * @param splats       Splats backing the ranges.
 * @param first, last  %Range of @ref Range objects.
 * @param grid         %Grid for transforming splat coordinates into grid coordinates.
 * @param microSize    Dimensions of the cubic cells in the most refined layer, in grid cells.
 * @param levels       Number of levels in the virtual octree.
 * @param func         User-provided functor.
 */
template<typename Func>
void forEachSplatCell(
    const SplatVector &splats,
    RangeConstIterator first,
    RangeConstIterator last,
    const Grid &grid, Cell::size_type microSize, unsigned int levels,
    const Func &func)
{
    ForEachSplatCell<Func> f(grid, microSize, levels, func);
    forEachSplat(splats, first, last, f);
}

} // namespace internal
} // namespace Bucket

#endif /* !BUCKET_INTERNAL_H */
