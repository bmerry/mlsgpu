/**
 * @file
 *
 * Declaration of @ref SplatTree.
 */

#ifndef SPLATTREE_H
#define SPLATTREE_H

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
    struct Level
    {
        /**
         * Index into @ref SplatTree::ids of the first
         * splat index for each cell. The cells are indexed
         * by their Morton codes.
         */
        std::vector<size_type> start;
    };

    const std::vector<Splat> &splats;
    const Grid &grid;

    /**
     * List of splats in each cell, all concatenated together.
     * They are organised first by level, then by cell code.
     */
    std::vector<size_type> ids;
    std::vector<Level> levels;

    /**
     * Compute a Morton code by interleaving the bits of @a x, @a y, @a z.
     *
     * @pre x, y and z contain at most one third of the bits in @ref size_type (e.g.
     * 10 if @c size_type is 32-bit).
     */
    static size_type makeCode(size_type x, size_type y, size_type z);
public:
    /**
     * Constructor.
     *
     * The tree holds references to the splats and the grid. They must not be
     * deleted or changed until the SplatTree has been destroyed.
     *
     * @param splats        Underlying splats (holds a reference, not a copy).
     * @param grid          Grid overlaying the splats (holds a reference, not a copy).
     *
     * @pre
     * - The number of splats must be at most 1/8th the number that can be held
     *   in @ref size_type.
     * - All the splats must be entirely contained within the grid cells.
     * - The grid must support @ref Grid::worldtoVertex.
     */
    SplatTree(const std::vector<Splat> &splats, const Grid &grid);
};

#endif /* !SPLATTREE_H */
