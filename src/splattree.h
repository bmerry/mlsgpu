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

struct SplatTree
{
public:
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
     */
    size_type makeCode(size_type x, size_type y, size_type z);
public:
    SplatTree(const std::vector<Splat> &splats, const Grid &grid);
};

#endif /* !SPLATTREE_H */
