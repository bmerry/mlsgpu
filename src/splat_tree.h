/*
 * mlsgpu: surface reconstruction from point clouds
 * Copyright (C) 2013  University of Cape Town
 *
 * This file is part of mlsgpu.
 *
 * mlsgpu is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

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
#include "tr1_cstdint.h"
#include "grid.h"

struct Splat;
class TestSplatTree;

/**
 * A dense octree containing splats.
 *
 * This is an abstract base class, which subclasses override to provide the
 * backing storage (e.g. in host memory, or in something like OpenCL buffer
 * memory).
 *
 * The octree is a complete octree (i.e. has information about all cells at
 * all levels of the hierarchy). It is encoded in several layers of indirection:
 * -# The splats themselves.
 * -# A <em>command buffer</em>, consisting of a mix of splat indices and metadata.
 *    The buffer is grouped into <em>ranges</em>. The last entry in the range is
 *    either non-negative (indicating the absolute offset to the start of another
 *    range), or is -1 to indicate a terminal range. The first entry in each range
 *    contains the absolute index of the last entry in the range (which is the
 *    jump/terminate). All other entries in the range are splat IDs. Ranges always
 *    contain at least one splat ID.
 * -# A <em>start array</em>, indicating where to start in the command buffer for
 *    each leaf. If the leaf has no overlapping splats (at any level of the hierarchy)
 *    the start value will be -1.
 *
 * To find all splats overlapping a leaf, look up the start position in the start
 * array, then process the linked list of ranges (linked by the jump index).
 *
 * The octree structure is thus hidden, but internally each cell in the octree
 * corresponds to a contiguous slice of the command buffer, which ends with a
 * jump to an ancestor.
 *
 * The leaves correspond to grid cells. Thus, a splat will be found if it intersects
 * any part of that cell. The boundary conditions are not formally defined, but
 * in the current implementation they are half-open in each dimension.
 *
 * Note that splats can be found just by walking up (or down) the tree, without
 * visiting any neighbors.
 */
struct SplatTree
{
    friend class TestSplatTree;
public:
    /**
     * Maximum number of cells in which any splat will be entered.
     */
    static const unsigned int maxAmplify = 8;

    /**
     * Type used to represent values in the command table.
     * It needs enough bits to represent both splat values, and to
     * represent jump values.
     */
    typedef std::tr1::int32_t command_type;

    /**
     * Type used to represent indices into the cells, and also for
     * sort keys that are a Morton code with a leading 1 bit.
     */
    typedef std::tr1::uint32_t code_type;

private:
    unsigned int numLevels; ///< Number of levels in the octree.

protected:
    /**
     * The backing store of splats. These should not be changed after
     * constructing the octree (other than their normals or quality),
     * as this would invalidate the octree.
     */
    const std::vector<Splat> &splats;

    /**
     * Size passed to the constructor.
     */
    Grid::size_type size[3];

    /**
     * Offset passed to the constructor.
     */
    Grid::difference_type offset[3];

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
     * Allocate the command array.
     */
    virtual command_type *allocateCommands(std::size_t size) = 0;

    /**
     * Allocate the start array.
     */
    virtual command_type *allocateStart(std::size_t size) = 0;

    /**
     * Constructor.
     *
     * The tree holds references to the splats. They must not be deleted or
     * changed until the SplatTree has been destroyed. The splats must already
     * have been transformed into the grid coordinate system, although this class
     * allows for biasing by integer amounts to support building octrees on
     * subgrids.
     *
     * @param splats        Underlying splats (holds a reference, not a copy).
     * @param size          Number of grid cells in each dimension.
     * @param offset        First grid cell to use in each dimension.
     *
     * This does not initialize the octree. The subclass must first make preparation to
     * implement the allocators, then call @ref initialize.
     *
     * @pre
     * - The number of splats must be strictly less than 1/(2*maxAmplify) the number that
     *   can be held in @ref command_type.
     * - The number of grid cells (after padding out to a power of 2) must satisfy
     *   the range limits in @ref makeCode.
     *
     * @todo Implement checks for the pre-conditions.
     */
    SplatTree(const std::vector<Splat> &splats,
              const Grid::size_type size[3],
              const Grid::difference_type offset[3]);

public:
    unsigned int getNumLevels() const { return numLevels; }

    /**
     * Compute a Morton code by interleaving the bits of @a x, @a y, @a z.
     *
     * @pre x, y and z contain less than one third of the bits in @ref
     * code_type (e.g. at most 10 bits if @c code_type is 32-bit, 21 if
     * 64-bit).
     */
    static code_type makeCode(code_type x, code_type y, code_type z);
};

#endif /* !SPLAT_TREE_H */
