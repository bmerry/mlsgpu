/**
 * @file
 *
 * Implementation of @ref Grid.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <stdexcept>
#include <algorithm>
#include "errors.h"
#include "grid.h"

Grid::Grid()
{
    for (unsigned int i = 0; i < 3; i++)
    {
        reference[i] = 0.0f;
        extents[i] = std::pair<int, int>(0, 1);
    }
    for (unsigned int i = 0; i < 3; i++)
    {
        for (unsigned int j = 0; j < 3; j++)
            directions[i][j] = 0.0f;
    }
}

Grid::Grid(const float ref[3],
           const float xDir[3],
           const float yDir[3],
           const float zDir[3],
           int xLow, int xHigh, int yLow, int yHigh, int zLow, int zHigh)
{
    setReference(ref);
    setDirection(0, xDir);
    setDirection(1, yDir);
    setDirection(2, zDir);
    setExtent(0, xLow, xHigh);
    setExtent(1, yLow, yHigh);
    setExtent(2, zLow, zHigh);
}

void Grid::setReference(const float ref[3])
{
    std::copy(ref, ref + 3, reference);
}

void Grid::setDirection(int axis, const float dir[3])
{
    MLSGPU_ASSERT(axis >= 0 && axis < 3, std::out_of_range);
    std::copy(dir, dir + 3, directions[axis]);
}

void Grid::setExtent(int axis, int low, int high)
{
    MLSGPU_ASSERT(axis >= 0 && axis < 3, std::out_of_range);
    MLSGPU_ASSERT(low < high, std::invalid_argument);
    extents[axis] = std::pair<int, int>(low, high);
}

const float *Grid::getReference() const
{
    return reference;
}

const float *Grid::getDirection(int axis) const
{
    MLSGPU_ASSERT(axis >= 0 && axis < 3, std::out_of_range);
    return directions[axis];
}

const std::pair<int, int> &Grid::getExtent(int axis) const
{
    MLSGPU_ASSERT(axis >= 0 && axis < 3, std::out_of_range);
    return extents[axis];
}

void Grid::getVertex(int x, int y, int z, float vertex[3]) const
{
    for (unsigned int i = 0; i < 3; i++)
    {
        vertex[i] = reference[i]
            + (x + extents[0].first) * directions[0][i]
            + (y + extents[1].first) * directions[1][i]
            + (z + extents[2].first) * directions[2][i];
    }
}

void Grid::worldToVertex(const float world[3], float out[3]) const
{
    // Check that the directions are axial
    MLSGPU_ASSERT(directions[0][1] == 0.0f && directions[0][2] == 0.0f, std::invalid_argument);
    MLSGPU_ASSERT(directions[1][0] == 0.0f && directions[1][2] == 0.0f, std::invalid_argument);
    MLSGPU_ASSERT(directions[2][0] == 0.0f && directions[2][1] == 0.0f, std::invalid_argument);

    for (unsigned int i = 0; i < 3; i++)
    {
        out[i] = (world[i] - reference[i]) / directions[i][i] - extents[i].first;
    }
}

int Grid::numVertices(int axis) const
{
    return numCells(axis) + 1;
}

int Grid::numCells(int axis) const
{
    MLSGPU_ASSERT(axis >= 0 && axis < 3, std::out_of_range);
    return extents[axis].second - extents[axis].first;
}

Grid Grid::subGrid(int x0, int x1, int y0, int y1, int z0, int z1) const
{
    MLSGPU_ASSERT(x0 <= x1, std::invalid_argument);
    MLSGPU_ASSERT(y0 <= y1, std::invalid_argument);
    MLSGPU_ASSERT(z0 <= z1, std::invalid_argument);
    Grid g = *this;
    g.extents[0] = std::make_pair(extents[0].first + x0, extents[0].first + x1);
    g.extents[1] = std::make_pair(extents[1].first + y0, extents[1].first + y1);
    g.extents[2] = std::make_pair(extents[2].first + z0, extents[2].first + z1);
    return g;
}
