/**
 * @file
 *
 * Declaration of @ref SplatTree.
 */

#ifndef SPLAT_TREE_H
#define SPLAT_TREE_H

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <vector>
#include <tr1/cstdint>

struct Splat;
class Grid;
class TestSplatTree;

/**
 * An octree containing splats.
 *
 * This is an abstract base class, which subclasses override to provide the
 * backing storage (e.g. in host memory, or in something like OpenCL buffer
 * memory).
 *
 * Cells in the octree are described by a @em level and a @em code. The level counts
 * from 0 (top-level single cell) to a maximum, while the code consists of the
 * coordinates of the cell bit-interleaved into a single Morton code (see @ref
 * makeCode). These are sometimes combined into a @em position, which is a flattened
 * iteration over the cells, level-by-level from level 0 downwards.
 *
 * The octree is a complete octree (i.e. has information about all cells at
 * all levels of the hierarchy). It is encoded in several layers of indirection:
 * -# The splats themselves.
 * -# The splat IDs array. This is a flat store of indices into the splats. Each
 *    cell corresponds to a contiguous slice of this array. Splats are entered into
 *    possibly multiple cells, so the IDs array might have up to 8 entries per splat.
 * -# The start array. The cell with position @em pos is described by elements
 *    [<code>start[pos]</code>, <code>start[pos + 1]</code>) in the IDs array. The
 *    start array has an extra trailing element to allow this to be used for the final
 *    cell as well.
 * -# The levelStart array. This is a convenience array to indicate the first position
 *    for each level. It always has the same elements: 0, 1, 1+8, 1+8+64 etc.
 */
struct SplatTree
{
    friend class TestSplatTree;
public:
    /**
     * Type used to represent offsets in the internal data structures.
     * This type must have enough bits to represent numbers up to 8 times
     * the number of splats.
     */
    typedef std::tr1::uint32_t size_type;
private:
    unsigned int numLevels;

protected:
    const std::vector<Splat> &splats;
    const Grid &grid;

    /**
     * Build the octree data from the splats and grid.
     *
     * This is called by subclasses to implement the building
     * algorithm. It will in turn call the pure virtual methods
     * to manage data allocation.
     *
     * It is guaranteed that this function will only write data
     * into the allocated arrays, not read from them. Thus, it is
     * suitable for use with non-standard memory types with poor
     * read performance.
     */
    void initialize();

    /**
     * Allocate the IDs array.
     */
    virtual size_type *allocateIds(size_type size) = 0;

    /**
     * Allocate the start array.
     */
    virtual size_type *allocateStart(size_type size) = 0;

    /**
     * Allocate the levelStart array.
     */
    virtual size_type *allocateLevelStart(size_type size) = 0;

    /**
     * Compute a Morton code by interleaving the bits of @a x, @a y, @a z.
     *
     * @pre x, y and z contain at most one third of the bits in @ref size_type (e.g.
     * 10 if @c size_type is 32-bit).
     */
    static size_type makeCode(size_type x, size_type y, size_type z);

    /**
     * Constructor.
     *
     * The tree holds references to the splats and the grid. They must not be
     * deleted or changed until the SplatTree has been destroyed.
     *
     * @param splats        Underlying splats (holds a reference, not a copy).
     * @param grid          Grid overlaying the splats (holds a reference, not a copy).
     *
     * This does not initialize the octree. The subclass must first make preparation to
     * implement the allocators, then call @ref initialize.
     *
     * @pre
     * - The number of splats must be at most 1/8th the number that can be held
     *   in @ref size_type.
     * - All the splats must be entirely contained within the grid cells.
     * - The grid must support @ref Grid::worldtoVertex.
     */
    SplatTree(const std::vector<Splat> &splats, const Grid &grid);

public:
    unsigned int getNumLevels() const { return numLevels; }
};

#endif /* !SPLATTREE_H */
