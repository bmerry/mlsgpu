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

/**
 * Class representing a regular grid of points in 3D space.
 *
 * A grid is specified by:
 *  - a reference point
 *  - three sampling steps (typically axis-aligned, but this is not required)
 *  - range of steps to take along each direction
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
    Grid();
    Grid(const float ref[3], const float xDir[3], const float yDir[3], const float zDir[3],
         int xLow, int xHigh, int yLow, int yHigh, int zLow, int zHigh);

    /// Set the reference point
    void setReference(const float ref[3]);

    /**
     * Set the reference direction for one axis.
     * @pre @a axis is 0, 1 or 2.
     */
    void setDirection(int axis, const float dir[3]);

    /**
     * Set the number of steps for vertices relative to one axis.
     *
     * The vertices along this axis range from
     * <code>reference + low * dir[axis]</code> to
     * <code>reference + high * dir[axis]</code> inclusive.
     * @pre
     * - @a axis is 0, 1 or 2.
     * - @a low < @a high.
     */
    void setExtent(int axis, int low, int high);

    /**
     * Retrieve the reference point.
     * @see @ref setReference.
     */
    const float *getReference() const;
    /**
     * Retrieve the step direction for one axis.
     * @pre @a axis is 0, 1 or 2.
     * @see @ref setDirection.
     */
    const float *getDirection(int axis) const;
    /**
     * Retrieve the extent range for one axis.
     * @pre @a axis is 0, 1 or 2.
     * @see @ref setExtent.
     */
    const std::pair<int, int> &getExtent(int axis) const;

    /**
     * Turn a grid-indexed vertex position into world coordinates.
     * Note that (0, 0, 0) need not correspond to the reference point.
     * It corresponds to the low extents.
     *
     * The coordinates need not fall inside the grid itself.
     */
    void getVertex(int x, int y, int z, float vertex[3]) const;

    /**
     * Retrieves the number of vertices along the specified axis.
     * @pre @a axis is 0, 1 or 2.
     */
    int numVertices(int axis) const;

    /**
     * Retrieves the number of cells along the specified axis. This is
     * simply one less than the number of vertices.
     * @pre @a axis is 0, 1 or 2.
     */
    int numCells(int axis) const;

    /**
     * Inverse of @ref getVertex.
     *
     * @pre The directions are axially aligned.
     */
    void worldToVertex(const float world[3], float out[3]) const;

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
    Grid subGrid(int x0, int x1, int y0, int y1, int z0, int z1) const;

private:
    float reference[3];              ///< Reference point
    float directions[3][3];          ///< <code>directions[i]</code> is the step along the ith axis
    std::pair<int, int> extents[3];  ///< Axis extents
};

#endif /* !GRID_H */
