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
#include <boost/tr1/cmath.hpp>
#include "errors.h"
#include "grid.h"

Grid::Grid()
{
    for (unsigned int i = 0; i < 3; i++)
    {
        reference[i] = 0.0f;
        extents[i] = extent_type(0, 1);
    }
    spacing = 0.0f;
}

Grid::Grid(const float ref[3], const float spacing,
           difference_type xLow, difference_type xHigh,
           difference_type yLow, difference_type yHigh,
           difference_type zLow, difference_type zHigh)
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

void Grid::setExtent(unsigned int axis, difference_type low, difference_type high)
{
    MLSGPU_ASSERT(axis < 3, std::out_of_range);
    MLSGPU_ASSERT(low < high, std::invalid_argument);
    extents[axis] = extent_type(low, high);
}

const float *Grid::getReference() const
{
    return reference;
}

float Grid::getSpacing() const
{
    return spacing;
}

const Grid::extent_type &Grid::getExtent(unsigned int axis) const
{
    MLSGPU_ASSERT(axis < 3, std::out_of_range);
    return extents[axis];
}

void Grid::getVertex(difference_type x, difference_type y, difference_type z, float vertex[3]) const
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

void Grid::worldToCell(const float world[3], difference_type out[3]) const
{
    for (unsigned int i = 0; i < 3; i++)
    {
        float raw = (world[i] - reference[i]) / spacing;
        // boost::numeric_cast doesn't catch NaNs
        if (!(std::tr1::isfinite(raw)))
            throw boost::numeric::bad_numeric_cast();
        Grid::difference_type d = RoundDown::convert(raw);
        if (extents[i].first >= 0 &&
            d < extents[i].first + std::numeric_limits<difference_type>::min())
        {
            throw boost::numeric::negative_overflow();
        }
        else if (extents[i].first <= 0 && d > extents[i].first + std::numeric_limits<difference_type>::max())
        {
            throw boost::numeric::positive_overflow();
        }
        out[i] = d - extents[i].first;
    }
}

Grid::size_type Grid::numVertices(unsigned int axis) const
{
    return numCells(axis) + 1;
}

Grid::size_type Grid::numCells(unsigned int axis) const
{
    MLSGPU_ASSERT(axis < 3, std::out_of_range);
    return extents[axis].second - extents[axis].first;
}

std::tr1::uint64_t Grid::numCells() const
{
    std::tr1::uint64_t ans = 1;
    for (int axis = 0; axis < 3; axis++)
        ans *= numCells(axis);
    return ans;
}

Grid Grid::subGrid(difference_type x0, difference_type x1,
                   difference_type y0, difference_type y1,
                   difference_type z0, difference_type z1) const
{
    MLSGPU_ASSERT(x0 <= x1, std::invalid_argument);
    MLSGPU_ASSERT(y0 <= y1, std::invalid_argument);
    MLSGPU_ASSERT(z0 <= z1, std::invalid_argument);
    Grid g = *this;
    g.extents[0] = extent_type(extents[0].first + x0, extents[0].first + x1);
    g.extents[1] = extent_type(extents[1].first + y0, extents[1].first + y1);
    g.extents[2] = extent_type(extents[2].first + z0, extents[2].first + z1);
    return g;
}
