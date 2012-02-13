/**
 * @file
 *
 * Bucketing of splats into sufficiently small buckets.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <vector>
#include <tr1/cstdint>
#include <limits>
#include <stdexcept>
#include <algorithm>
#include <utility>
#include <cassert>
#include <functional>
#include <tr1/cmath>
#include <boost/array.hpp>
#include <boost/multi_array.hpp>
#include <boost/foreach.hpp>
#include <boost/ref.hpp>
#include <boost/numeric/conversion/converter.hpp>
#include <stxxl.h>
#include "splat.h"
#include "bucket.h"
#include "bucket_internal.h"
#include "bucket_impl.h"
#include "errors.h"
#include "statistics.h"
#include "timer.h"
#include "misc.h"
#include "logging.h"

namespace Bucket
{

namespace internal
{

Node::Node(const size_type coords[3], unsigned int level) : level(level)
{
    for (unsigned int i = 0; i < 3; i++)
    {
        this->coords[i] = coords[i];
    }
}

Node::Node(size_type x, size_type y, size_type z, unsigned int level) : level(level)
{
    coords[0] = x;
    coords[1] = y;
    coords[2] = z;
}

Node Node::child(unsigned int idx) const
{
    MLSGPU_ASSERT(level > 0, std::invalid_argument);
    MLSGPU_ASSERT(idx < 8, std::invalid_argument);
    return Node(
        coords[0] * 2 + (idx & 1),
        coords[1] * 2 + ((idx >> 1) & 1),
        coords[2] * 2 + (idx >> 2),
        level - 1);
}

void Node::toMicro(size_type lower[3], size_type upper[3]) const
{
    for (unsigned int i = 0; i < 3; i++)
    {
        lower[i] = coords[i] << level;
        upper[i] = lower[i] + (size_type(1) << level);
    }
}

void Node::toMicro(size_type lower[3], size_type upper[3], const size_type limit[3]) const
{
    toMicro(lower, upper);
    for (unsigned int i = 0; i < 3; i++)
    {
        lower[i] = std::min(lower[i], limit[i]);
        upper[i] = std::min(upper[i], limit[i]);
    }
}

void Node::toCells(Grid::size_type microSize, Grid::size_type lower[3], Grid::size_type upper[3]) const
{
    for (unsigned int i = 0; i < 3; i++)
    {
        lower[i] = (microSize * coords[i]) << level;
        upper[i] = lower[i] + (microSize << level);
    }
}

void Node::toCells(Grid::size_type microSize, Grid::size_type lower[3], Grid::size_type upper[3],
                   const Grid &grid) const
{
    toCells(microSize, lower, upper);
    for (unsigned int i = 0; i < 3; i++)
    {
        lower[i] = std::min(lower[i], grid.numCells(i));
        upper[i] = std::min(upper[i], grid.numCells(i));
    }
}

bool Node::operator==(const Node &b) const
{
    return coords == b.coords && level == b.level;
}

const std::size_t BucketState::BAD_REGION;

BucketState::BucketState(
    const BucketParameters &params, const Grid &grid,
    Grid::size_type microSize, int macroLevels)
    : params(params), grid(grid), microSize(microSize), macroLevels(macroLevels),
    nodeCounts(macroLevels)
{
    for (int i = 0; i < 3; i++)
        dims[i] = divUp(grid.numCells(i), microSize);

    for (int level = 0; level < macroLevels; level++)
    {
        /* We don't need a full power-of-two allocation for a level of the octree,
         * just enough to completely cover the original dimensions.
         */
        boost::array<Node::size_type, 3> s;
        for (int i = 0; i < 3; i++)
        {
            s[i] = divUp(dims[i], Grid::size_type(1) << level);
            assert(level != macroLevels - 1 || s[i] == 1);
        }
        nodeCounts[level].resize(s);
        if (level == 0)
        {
            microRegions.resize(s);
            for (Node::size_type x = 0; x < s[0]; x++)
                for (Node::size_type y = 0; y < s[1]; y++)
                    for (Node::size_type z = 0; z < s[2]; z++)
                        microRegions[x][y][z] = BAD_REGION;
        }
    }
}

void BucketState::upsweepCounts()
{
    for (int level = 0; level + 1 < macroLevels; level++)
    {
        for (Node::size_type x = 0; x < nodeCounts[level].shape()[0]; x++)
            for (Node::size_type y = 0; y < nodeCounts[level].shape()[1]; y++)
                for (Node::size_type z = 0; z < nodeCounts[level].shape()[2]; z++)
                    nodeCounts[level + 1][x >> 1][y >> 1][z >> 1] += nodeCounts[level][x][y][z];
    }
}

std::tr1::int64_t BucketState::getNodeCount(const Node &node) const
{
    assert(node.getLevel() < nodeCounts.size());
    return nodeCounts[node.getLevel()](node.getCoords());
}

void BucketState::getSplatMicro(const Splat &splat, Node::size_type lo[3], Node::size_type hi[3])
{
    float worldLow[3], worldHigh[3];
    float gridLow[3], gridHigh[3];

    if (!splat.isFinite())
    {
        /* Return an empty region */
        lo[0] = lo[1] = lo[2] = 1;
        hi[0] = hi[1] = hi[2] = 0;
        return;
    }

    for (unsigned int i = 0; i < 3; i++)
    {
        worldLow[i] = splat.position[i] - splat.radius;
        worldHigh[i] = splat.position[i] + splat.radius;
    }
    grid.worldToVertex(worldLow, gridLow);
    grid.worldToVertex(worldHigh, gridHigh);
    for (int i = 0; i < 3; i++)
    {
        Grid::size_type top = nodeCounts[0].shape()[i] - 1;
        lo[i] = std::max(RoundUp()(gridLow[i] / microSize), 1) - 1;
        hi[i] = std::min(Grid::size_type(RoundDown()(gridHigh[i] / microSize)), top);
    }
}

void CountSplat::operator()(Range::scan_type scan, Range::index_type id,
                            const Splat &splat) const
{
    (void) scan;
    (void) id;

    int level = 0;
    Node::size_type lo[3], hi[3];
    state.getSplatMicro(splat, lo, hi);
    for (Node::size_type x = lo[0]; x <= hi[0]; x++)
        for (Node::size_type y = lo[1]; y <= hi[1]; y++)
            for (Node::size_type z = lo[2]; z <= hi[2]; z++)
            {
                ++state.nodeCounts[level][x][y][z];
            }
    while (level < state.macroLevels && (lo[0] < hi[0] || lo[1] < hi[1] || lo[2] < hi[2]))
    {
        level++;
        for (Node::size_type x = lo[0] >> 1; x <= (hi[0] >> 1); x++)
            for (Node::size_type y = lo[1] >> 1; y <= (hi[1] >> 1); y++)
                for (Node::size_type z = lo[2] >> 1; z <= (hi[2] >> 1); z++)
                {
                    unsigned int hits = 1;
                    if (lo[0] <= 2 * x && 2 * x < hi[0])
                        hits *= 2;
                    if (lo[1] <= 2 * y && 2 * y < hi[1])
                        hits *= 2;
                    if (lo[2] <= 2 * z && 2 * z < hi[2])
                        hits *= 2;
                    state.nodeCounts[level][x][y][z] -= hits - 1;
                }
        for (unsigned int i = 0; i < 3; i++)
        {
            lo[i] >>= 1;
            hi[i] >>= 1;
        }
    }
}

bool PickNodes::operator()(const Node &node) const
{
    std::tr1::uint64_t count = state.getNodeCount(node);

    // Skip completely empty regions, but record the fact
    // for progress meters
    if (count == 0)
    {
        if (state.params.progress != NULL)
        {
            // Intersect with grid
            std::tr1::uint64_t skipped = 1;
            Grid::size_type lower[3], upper[3];
            node.toCells(state.microSize, lower, upper, state.grid);
            for (int i = 0; i < 3; i++)
            {
                skipped *= upper[i] - lower[i];
            }
            *state.params.progress += skipped;
        }
        return false;
    }

    if (node.getLevel() == 0
        || (state.microSize * node.size() <= state.params.maxCells
            && count <= state.params.maxSplats))
    {
        std::size_t id = state.subregions.size();
        Node::size_type lo[3], hi[3];
        node.toMicro(lo, hi, state.dims);
        for (Node::size_type x = lo[0]; x < hi[0]; x++)
            for (Node::size_type y = lo[1]; y < hi[1]; y++)
                for (Node::size_type z = lo[2]; z < hi[2]; z++)
                    state.microRegions[x][y][z] = id;
        state.subregions.push_back(BucketState::Subregion(node, count));
        return false; // no more recursion required
    }
    else
        return true;
}

/**
 * Pick the smallest possible power of 2 size for a microblock.
 * The limitation is thta there must be at most @a maxSplit microblocks.
 */
Grid::size_type chooseMicroSize(
    const Grid::size_type dims[3], std::size_t maxSplit)
{
    Grid::size_type microSize = 1;
    std::size_t microBlocks = 1;
    for (int i = 0; i < 3; i++)
        microBlocks = mulSat(microBlocks, std::size_t(divUp(dims[i], microSize)));
    while (microBlocks > maxSplit)
    {
        microSize *= 2;
        microBlocks = 1;
        for (int i = 0; i < 3; i++)
            microBlocks = mulSat(microBlocks, std::size_t(divUp(dims[i], microSize)));
    }
    return microSize;
}

const Splat &MakeGrid::operator()(const Splat &splat)
{
    if (!splat.isFinite())
    {
        nonFinite++;
        return splat;
    }

    const float radius = splat.radius;
    if (first)
    {
        for (unsigned int i = 0; i < 3; i++)
        {
            low[i] = splat.position[i];
            bboxMin[i] = low[i] - radius;
            bboxMax[i] = low[i] + radius;
        }
        first = false;
    }
    else
    {
        for (unsigned int j = 0; j < 3; j++)
        {
            float p = splat.position[j];
            low[j] = std::min(low[j], p);
            bboxMin[j] = std::min(bboxMin[j], p - radius);
            bboxMax[j] = std::max(bboxMax[j], p + radius);
        }
    }
    return splat;
}

void MakeGrid::operator()(unsigned int, const Splat &splat)
{
    operator()(splat);
}

Grid MakeGrid::makeGrid(float spacing) const
{
    if (nonFinite > 0)
        Log::log[Log::warn] << "Input contains " << nonFinite << " splat(s) with non-finite values\n";
    if (first)
        throw std::length_error("Must be at least one splat");

    int extents[3][2];
    for (unsigned int i = 0; i < 3; i++)
    {
        float l = (bboxMin[i] - low[i]) / spacing;
        float h = (bboxMax[i] - low[i]) / spacing;
        extents[i][0] = RoundDown::convert(l);
        extents[i][1] = RoundUp::convert(h);
    }

    return Grid(low, spacing,
                extents[0][0], extents[0][1], extents[1][0], extents[1][1], extents[2][0], extents[2][1]);
}

} // namespace internal

} // namespace Bucket
