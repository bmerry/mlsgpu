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
#include <iosfwd>
#include <cstddef>
#include <cstring>
#include <boost/ref.hpp>
#include <boost/array.hpp>
#include <boost/noncopyable.hpp>
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
     * @param coords   Coordinates of the node (in units of its own size)
     * @param level    Octree level (zero being the finest).
     */
    Node(const size_type coords[3], unsigned int level);

    /**
     * Constructor which is more convenient for building literals.
     *
     * @param x,y,z    Coordinates of node (in units of its own size)
     * @param level    Octree level (zero being the finest).
     */
    Node(size_type x, size_type y, size_type z, unsigned int level);

    /// Get the octree level
    unsigned int getLevel() const { return level; }

    /// Get the side length in microblocks
    size_type size() const { return size_type(1) << level; }

    const boost::array<size_type, 3> &getCoords() const { return coords; }

    /**
     * Convert to coordinate range in microblocks.
     *
     * @param[out] lower     Lower coordinate (inclusive).
     * @param[out] upper     Upper coordinate (exclusive).
     */
    void toMicro(size_type lower[3], size_type upper[3]) const;

    /**
     * Convert to coordinate range in microblocks and clamp.
     *
     * @param[out] lower     Lower coordinate (inclusive).
     * @param[out] upper     Upper coordinate (exclusive).
     * @param      limit     Maximum values that will be returned.
     */
    void toMicro(size_type lower[3], size_type upper[3], const size_type limit[3]) const;

    /**
     * Convert to coordinate range measured in grid cells.
     *
     * @param      microSize Number of grid cells per microblock.
     * @param[out] lower     Lower coordinate (inclusive).
     * @param[out] upper     Upper coordinate (exclusive).
     */
    void toCells(Grid::size_type microSize, Grid::size_type lower[3], Grid::size_type upper[3]) const;

    /**
     * Convert to coordinate range measured in grid cells, and clamp to grid.
     * The coordinates will be limited to the number of cells in each
     * dimension in the grid.
     *
     * @param      microSize Number of grid cells per microblock.
     * @param[out] lower     Lower coordinate (inclusive).
     * @param[out] upper     Upper coordinate (exclusive).
     * @param      grid      Clamping grid.
     */
    void toCells(Grid::size_type microSize, Grid::size_type lower[3], Grid::size_type upper[3],
                 const Grid &grid) const;

    /**
     * Create a child node in the octree.
     * @pre level > 0 and idx < 8.
     */
    Node child(unsigned int idx) const;

    /// Equality comparison (used by test code)
    bool operator==(const Node &n) const;

private:
    boost::array<size_type, 3> coords;         ///< Coordinates (in units of its own size)
    unsigned int level;                        ///< Octree level
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
     * Add a contiguous range of new splats to the list.
     *
     * @return The output iterator after the update.
     */
    iterator_type append(Range::scan_type scan, Range::index_type first, Range::index_type last);

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
 * @param dims      Dimensions of the most refined layer, in microblocks.
 * @param levels    Number of levels in the virtual octree.
 * @param func      User-provided functor.
 *
 * @pre
 * - @a levels is at least 1 and at most the number of bits in @c Node::size_type
 * - The dimensions are each at most @a 2<sup>levels - 1</sup>.
 */
template<typename Func>
void forEachNode(const Node::size_type dims[3], unsigned int levels, const Func &func);

} // namespace internal
} // namespace Bucket

#include "bucket_impl.h"

#endif /* !BUCKET_INTERNAL_H */
