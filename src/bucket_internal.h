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
 * A cuboid consisting of a number of microblocks.
 * However, the coordinates are currently represented in cells, not
 * microblocks.
 */
class Node
{
public:
    /**
     * Type used to represent cell coordinates. Note that there is no point
     * in this being wider than the type used by @ref Grid.
     */
    typedef Grid::size_type size_type;

    /// Default constructor: does not initialize anything.
    Node() {}

    /**
     * Constructor.
     *
     * @param lower    Coordinates of minimum corner.
     * @param upper    Coordinates of maximum corner.
     * @param level    Octree level.
     * @pre The node has positive side lengths.
     */
    Node(const size_type lower[3], const size_type upper[3], unsigned int level);

    /**
     * Constructor which is more convenient for building literals.
     *
     * @param lowerX,lowerY,lowerZ    Coordinates of minimum corner.
     * @param upperX,upperY,upperZ    Coordinates of maximum corner.
     * @param level                   Octree level.
     * @pre The node has positive side lengths.
     */
    Node(size_type lowerX, size_type lowerY, size_type lowerZ,
         size_type upperX, size_type upperY, size_type upperZ,
         unsigned int level);

    /// Equality comparison operator
    bool operator==(const Node &c) const;

    /// Get the octree level
    unsigned int getLevel() const { return level; }

    /// Get the side length on an axis
    size_type getSize(int axis) const { return upper[axis] - lower[axis]; }

    /// Get the lower corner
    const size_type *getLower() const { return lower; }

    /// Get the upper corner
    const size_type *getUpper() const { return upper; }

    /**
     * Create a child node in the octree.
     * @pre level > 0 and idx < 8.
     */
    Node child(unsigned int idx) const;

private:
    size_type lower[3];         ///< Coordinates of lower-left-bottom corner
    size_type upper[3];         ///< Coordinates of upper-right-top corner
    /**
     * Octree level. This has no direct effect on the spatial extent of the
     * node, but is used in finding corresponding indices in the octree. The
     * level is zero for the finest (microblock) level of the octree.
     */
    unsigned int level;
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
 * Recursively walk an octree, calling a user-defined function on each node.
 * The function takes a @ref Node and returns a boolean indicating whether
 * the children should be visited in turn (ignored for leaves).
 *
 * The most refined level does not need to be power-of-two in size. Any octree
 * nodes that do not at least partially intersect the rectangle defined by @a
 * dims will be skipped.
 *
 * @param dims      Dimensions of the most refined layer, in cells.
 * @param microSize Dimensions of the nodes in the most refined layer, in cells.
 * @param levels    Number of levels in the virtual octree.
 * @param func      User-provided functor.
 *
 * @pre
 * - @a levels is at least 1 and at most the number of bits in @c Node::size_type
 * - The dimensions are each at most @a microSize * 2<sup>levels - 1</sup>.
 */
template<typename Func>
void forEachNode(const Node::size_type dims[3], Node::size_type microSize, unsigned int levels, const Func &func);

/**
 * Overload that takes a grid instead of explicit dimensions. The dimensions are taken from
 * the number of cells along each dimension of the grid.
 */
template<typename Func>
void forEachNode(const Grid &grid, Node::size_type microSize, unsigned int levels, const Func &func);

/**
 * Iterate over all splats given be a collection of @ref Range, calling
 * a user-provided function for each.
 *
 * @param splats       %Random access container of collections to walk.
 * @param first, last  %Range of @ref Range objects.
 * @param func         User-provided callback.
 */
template<typename CollectionSet, typename Func>
void forEachSplat(
    const CollectionSet &splats,
    RangeConstIterator first,
    RangeConstIterator last,
    const Func &func);

} // namespace internal
} // namespace Bucket

#include "bucket_impl.h"

#endif /* !BUCKET_INTERNAL_H */
