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

Grid::Grid(const float bboxMin[3], const float bboxMax[3], int nx, int ny, int nz)
{
    MLSGPU_ASSERT(nx > 0 && ny > 0 && nz > 0, std::length_error);
    setReference(bboxMin);

    float dir[3];
    for (unsigned int i = 0; i < 3; i++)
    {
        dir[0] = dir[1] = dir[2] = 0.0f;
        dir[i] = bboxMax[i] - bboxMin[i];
        setDirection(i, dir);
    }
    setExtent(0, 0, nx);
    setExtent(1, 0, ny);
    setExtent(2, 0, nz);
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
    MLSGPU_ASSERT(x >= 0 && x <= extents[0].second - extents[0].first, std::out_of_range);
    MLSGPU_ASSERT(y >= 0 && y <= extents[1].second - extents[1].first, std::out_of_range);
    MLSGPU_ASSERT(z >= 0 && z <= extents[2].second - extents[2].first, std::out_of_range);
    for (unsigned int i = 0; i < 3; i++)
    {
        vertex[i] = reference[i]
            + (x + extents[0].first) * directions[0][i]
            + (y + extents[1].first) * directions[1][i]
            + (z + extents[2].first) * directions[2][i];
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

