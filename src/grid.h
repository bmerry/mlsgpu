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

    void setReference(const float ref[3]);
    void setDirection(int axis, const float dir[3]);
    void setExtent(int axis, int low, int high);

    const float *getReference() const;
    const float *getDirection(int axis) const;
    const std::pair<int, int> &getExtent(int axis) const;

    void getVertex(int x, int y, int z, float vertex[3]) const;
    int numVertices(int axis) const;
    int numCells(int axis) const;
private:
    float reference[3];
    float directions[3][3];
    std::pair<int, int> extents[3];
};

#endif /* !GRID_H */
