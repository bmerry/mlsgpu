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
void forEachCell(const Cell::size_type dims[3], Cell::size_type microSize, unsigned int levels, const Func &func);

/**
 * Overload that takes a grid instead of explicit dimensions. The dimensions are taken from
 * the number of cells along each dimension of the grid.
 */
template<typename Func>
void forEachCell(const Grid &grid, Cell::size_type microSize, unsigned int levels, const Func &func);

/**
 * Iterate over all splats given be a collection of @ref Range, calling
 * a user-provided function for each.
 *
 * @param collections  %Random access container of collections to walk.
 * @param first, last  %Range of @ref Range objects.
 * @param func         User-provided callback.
 */
template<typename CollectionSet, typename Func>
void forEachSplat(
    const CollectionSet &splats,
    RangeConstIterator first,
    RangeConstIterator last,
    const Func &func);

/**
 * Combination of @ref forEachSplat and @ref forEachCell that calls the user
 * function for cells that are intersected by the splat. As for @ref forEachCell,
 * the function may return @c false to prevent recursion into child cells.
 *
 * The function takes the arg
 *
 * @param splats       %Random access container of collections to walk.
 * @param first, last  %Range of @ref Range objects.
 * @param grid         %Grid for transforming splat coordinates into grid coordinates.
 * @param microSize    Dimensions of the cubic cells in the most refined layer, in grid cells.
 * @param levels       Number of levels in the virtual octree.
 * @param func         User-provided functor.
 */
template<typename CollectionSet, typename Func>
void forEachSplatCell(
    const CollectionSet &splats,
    RangeConstIterator first,
    RangeConstIterator last,
    const Grid &grid, Cell::size_type microSize, unsigned int levels,
    const Func &func);

} // namespace internal
} // namespace Bucket

#include "bucket_impl.h"

#endif /* !BUCKET_INTERNAL_H */
