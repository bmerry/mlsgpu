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
#include <tr1/cstdint>
#include "errors.h"
#include "grid.h"

Grid::Grid()
{
    for (unsigned int i = 0; i < 3; i++)
    {
        reference[i] = 0.0f;
        extents[i] = std::pair<int, int>(0, 1);
    }
    spacing = 0.0f;
}

Grid::Grid(const float ref[3], const float spacing,
           int xLow, int xHigh, int yLow, int yHigh, int zLow, int zHigh)
{
    setReference(ref);
    setSpacing(spacing);
    setExtent(0, xLow, xHigh);
    setExtent(1, yLow, yHigh);
    setExtent(2, zLow, zHigh);
}

void Grid::setReference(const float ref[3])
{
    std::copy(ref, ref + 3, reference);
}

void Grid::setSpacing(float spacing)
{
    this->spacing = spacing;
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

float Grid::getSpacing() const
{
    return spacing;
}

const std::pair<int, int> &Grid::getExtent(int axis) const
{
    MLSGPU_ASSERT(axis >= 0 && axis < 3, std::out_of_range);
    return extents[axis];
}

void Grid::getVertex(int x, int y, int z, float vertex[3]) const
{
    vertex[0] = reference[0] + spacing * (x + extents[0].first);
    vertex[1] = reference[1] + spacing * (y + extents[1].first);
    vertex[2] = reference[2] + spacing * (z + extents[2].first);
}

void Grid::worldToVertex(const float world[3], float out[3]) const
{
    for (unsigned int i = 0; i < 3; i++)
    {
        out[i] = (world[i] - reference[i]) / spacing - extents[i].first;
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

std::tr1::uint64_t Grid::numCells() const
{
    std::tr1::uint64_t ans = 1;
    for (int axis = 0; axis < 3; axis++)
        ans *= numCells(axis);
    return ans;
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
