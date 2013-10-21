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
 * Declaration of @ref Grid.
 */

#ifndef GRID_H
#define GRID_H

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <utility>
#include "tr1_cstdint.h"
#include <boost/numeric/conversion/converter.hpp>

/**
 * Class representing a regular grid of points in 3D space.
 *
 * A grid is specified by:
 *  - a reference point
 *  - a sample spacing
 *  - range of steps to take along each axis
 * The ranges are specified by two endpoints, rather than zero to some endpoint.
 * Although mathematically this is redundant (the reference could just be
 * moved), this is done for reasons of invariance: it allows the grid to be grown or shrunk in
 * either direction while keeping the common grid points in @em exactly the same place.
 *
 * However, this is an internal detail, only visible to the code which constructs the grid.
 * Users of the constructed grid use coordinates that always zero-based.
 */
class Grid
{
public:
    typedef int difference_type;
    typedef unsigned int size_type;
    typedef std::pair<difference_type, difference_type> extent_type;

    typedef boost::numeric::converter<
        difference_type,
        float,
        boost::numeric::conversion_traits<difference_type, float>,
        boost::numeric::def_overflow_handler,
        boost::numeric::Ceil<float> > RoundUp;
    typedef boost::numeric::converter<
        difference_type,
        float,
        boost::numeric::conversion_traits<difference_type, float>,
        boost::numeric::def_overflow_handler,
        boost::numeric::Floor<float> > RoundDown;

    Grid();
    Grid(const float ref[3], float spacing,
         difference_type xLow, difference_type xHigh,
         difference_type yLow, difference_type yHigh,
         difference_type zLow, difference_type zHigh);

    /// Set the reference point
    void setReference(const float ref[3]);

    /**
     * Set the grid spacing.
     */
    void setSpacing(float spacing);

    /**
     * Set the number of steps for vertices relative to one axis.
     *
     * The vertices along this axis range from
     * <code>reference + low * spacing * unit(axis)</code> to
     * <code>reference + high * spacing * unit(axis)</code> inclusive.
     * @pre
     * - @a axis is 0, 1 or 2.
     * - @a low < @a high.
     */
    void setExtent(unsigned int axis, difference_type low, difference_type high);

    /**
     * Retrieve the reference point.
     * @see @ref setReference.
     */
    const float *getReference() const;
    /**
     * Retrieve sample spacing.
     * @see @ref setSpacing.
     */
    float getSpacing() const;
    /**
     * Retrieve the extent range for one axis.
     * @pre @a axis is 0, 1 or 2.
     * @see @ref setExtent.
     */
    const extent_type &getExtent(unsigned int axis) const;

    /**
     * Turn a grid-indexed vertex position into world coordinates.
     * Note that (0, 0, 0) need not correspond to the reference point.
     * It corresponds to the low extents.
     *
     * The coordinates need not fall inside the grid itself.
     */
    void getVertex(difference_type x, difference_type y, difference_type z, float vertex[3]) const;

    /**
     * Retrieves the number of vertices along the specified axis.
     * @pre @a axis is 0, 1 or 2.
     */
    size_type numVertices(unsigned int axis) const;

    /**
     * Retrieves the number of cells along the specified axis. This is
     * simply one less than the number of vertices.
     * @pre @a axis is 0, 1 or 2.
     */
    size_type numCells(unsigned int axis) const;

    /**
     * Retrieves the number of cells in the grid.
     */
    std::tr1::uint64_t numCells() const;

    /**
     * Inverse of @ref getVertex. It is legal for @a world and @a out to be the
     * same.
     */
    void worldToVertex(const float world[3], float out[3]) const;

    /**
     * Rounds result of @ref worldToVertex down to the next integer.  It is
     * implemented in a way that is invariant to changes in the extents.  In
     * other words, adding X to the base extent will cause the result to
     * decrease by exactly X, which a naive implementation would not do
     * due to different rounding.
     *
     * If an exception occurs, @a out will contain undefined values.
     *
     * @param world        World coordinates
     * @param[out] out     Cell containing the world coordinates.
     * @throw boost::bad_numeric_conversion on overflow or non-finite value. This
     *        can happen because the cell coordinates are out of range either
     *        before or after the extent bias is applied.
     */
    void worldToCell(const float world[3], difference_type out[3]) const;

    /**
     * Create a new grid that has the same reference point and spacing
     * as this one, but different extents. The extents are specified
     * relative to this grid, so passing 0 and @ref numVertices) for
     * each dimension would give back the original grid.
     *
     * Note that although this is called @c subGrid, you can go outside
     * the extents of the original grid.
     *
     * @pre @a x0 <= @a x1, @a y0 <= @a y1, @a z0 <= @a z1.
     */
    Grid subGrid(difference_type x0, difference_type x1,
                 difference_type y0, difference_type y1,
                 difference_type z0, difference_type z1) const;

private:
    float reference[3];              ///< Reference point
    float spacing;                   ///< Spacing between samples
    extent_type extents[3];          ///< Axis extents
};

#endif /* !GRID_H */
