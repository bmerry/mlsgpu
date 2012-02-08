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

namespace Bucket
{

Range::Range() :
    scan(std::numeric_limits<scan_type>::max()),
    size(0),
    start(std::numeric_limits<index_type>::max())
{
}

Range::Range(scan_type scan, index_type splat) :
    scan(scan),
    size(1),
    start(splat)
{
}

Range::Range(scan_type scan, index_type start, size_type size)
    : scan(scan), size(size), start(start)
{
    MLSGPU_ASSERT(size == 0 || start <= std::numeric_limits<index_type>::max() - size + 1, std::out_of_range);
}

bool Range::append(scan_type scan, index_type splat)
{
    if (size == 0)
    {
        /* An empty range can always be extended. */
        this->scan = scan;
        size = 1;
        start = splat;
    }
    else if (this->scan == scan && splat >= start && splat - start <= size)
    {
        if (splat - start == size)
        {
            if (size == std::numeric_limits<size_type>::max())
                return false; // would overflow
            size++;
        }
    }
    else
        return false;
    return true;
}

namespace internal
{

Node::Node(const size_type lower[3], const size_type upper[3], unsigned int level) : level(level)
{
    for (unsigned int i = 0; i < 3; i++)
    {
        MLSGPU_ASSERT(lower[i] < upper[i], std::invalid_argument);
        this->lower[i] = lower[i];
        this->upper[i] = upper[i];
    }
}

Node::Node(
    size_type lowerX, size_type lowerY, size_type lowerZ,
    size_type upperX, size_type upperY, size_type upperZ,
    unsigned int level) : level(level)
{
    lower[0] = lowerX;
    lower[1] = lowerY;
    lower[2] = lowerZ;
    upper[0] = upperX;
    upper[1] = upperY;
    upper[2] = upperZ;
    for (unsigned int i = 0; i < 3; i++)
        MLSGPU_ASSERT(lower[i] < upper[i], std::invalid_argument);
}

bool Node::operator==(const Node &c) const
{
    return std::equal(lower, lower + 3, c.lower)
        && std::equal(upper, upper + 3, c.upper)
        && level == c.level;
}

Node Node::child(unsigned int idx) const
{
    MLSGPU_ASSERT(level > 0, std::invalid_argument);
    MLSGPU_ASSERT(idx < 8, std::invalid_argument);
    Node c = *this;
    c.level--;
    for (int i = 0; i < 3; i++)
    {
        size_type mid = (lower[i] + upper[i]) / 2;
        if ((idx >> i) & 1)
            c.lower[i] = mid;
        else
            c.upper[i] = mid;
    }
    return c;
}

const std::size_t BucketState::BAD_BLOCK;

BucketState::BucketState(
    const BucketParameters &params, const Grid &grid,
    Node::size_type microSize, int macroLevels)
    : params(params), grid(grid), microSize(microSize), macroLevels(macroLevels),
    nodeCounts(macroLevels), skippedCells(0)
{
    for (int i = 0; i < 3; i++)
        this->dims[i] = grid.numCells(i);

    for (int level = 0; level < macroLevels; level++)
    {
        /* We don't need a full power-of-two allocation for a level of the octree,
         * just enough to completely cover the original dimensions.
         */
        boost::array<Node::size_type, 3> s;
        for (int i = 0; i < 3; i++)
        {
            s[i] = divUp(dims[i], microSize << level);
            assert(level != macroLevels - 1 || s[i] == 1);
        }
        nodeCounts[level].resize(s);
        if (level == 0)
        {
            microRegions.resize(s);
            for (Node::size_type x = 0; x < s[0]; x++)
                for (Node::size_type y = 0; y < s[1]; y++)
                    for (Node::size_type z = 0; z < s[2]; z++)
                        microRegions[x][y][z] = BAD_BLOCK;
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
    boost::array<Node::size_type, 3> coords;
    Node::size_type factor = microSize << node.getLevel();
    for (int i = 0; i < 3; i++)
    {
        assert(node.getLower()[i] % factor == 0);
        coords[i] = node.getLower()[i] / factor;
    }
    assert(node.getLevel() < nodeCounts.size());
    return nodeCounts[node.getLevel()](coords);
}

void BucketState::getSplatMicro(const Splat &splat, Node::size_type lo[3], Node::size_type hi[3])
{
    float worldLow[3], worldHigh[3];
    float gridLow[3], gridHigh[3];

    for (unsigned int i = 0; i < 3; i++)
    {
        worldLow[i] = splat.position[i] - splat.radius;
        worldHigh[i] = splat.position[i] + splat.radius;
    }
    grid.worldToVertex(worldLow, gridLow);
    grid.worldToVertex(worldHigh, gridHigh);
    for (int i = 0; i < 3; i++)
    {
        int top = nodeCounts[0].shape()[i] - 1;
        lo[i] = std::min(std::max(RoundUp()(gridLow[i] / microSize) - 1, 0), top);
        hi[i] = std::min(std::max(RoundDown()(gridHigh[i] / microSize), 0), top);
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
    while (level < state.macroLevels - 1 && (lo[0] < hi[0] || lo[1] < hi[1] || lo[2] < hi[2]))
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
        // Intersect with grid
        std::tr1::uint64_t skipped = 1;
        for (int i = 0; i < 3; i++)
        {
            skipped *= std::min(state.dims[i], node.getUpper()[i]) - node.getLower()[i];
        }
        state.skippedCells += skipped;
        return false;
    }

    if (node.getLevel() == 0
        || (node.getSize(0) <= state.params.maxCells
            && node.getSize(1) <= state.params.maxCells
            && node.getSize(2) <= state.params.maxCells
            && count <= state.params.maxSplats))
    {
        std::size_t id = state.subregions.size();
        Node::size_type lo[3], hi[3];
        for (int i = 0; i < 3; i++)
        {
            lo[i] = node.getLower()[i] / state.microSize;
            hi[i] = std::min(divUp(node.getUpper()[i], state.microSize),
                             Node::size_type(state.microRegions.shape()[i]));
        }
        for (Node::size_type x = lo[0]; x < hi[0]; x++)
            for (Node::size_type y = lo[1]; y < hi[1]; y++)
                for (Node::size_type z = lo[2]; z < hi[2]; z++)
                    state.microRegions[x][y][z] = id;
        state.subregions.push_back(BucketState::Subregion(node));
        return false; // no more recursion required
    }
    else
        return true;
}

void BucketSplats::operator()(Range::scan_type scan, Range::index_type id,
                              const Splat &splat) const
{
    Node::size_type lo[3], hi[3];
    state.getSplatMicro(splat, lo, hi);
    for (Node::size_type x = lo[0]; x <= hi[0]; x++)
        for (Node::size_type y = lo[1]; y <= hi[1]; y++)
            for (Node::size_type z = lo[2]; z <= hi[2]; z++)
            {
                std::size_t block = state.microRegions[x][y][z];
                assert(block < state.subregions.size());
                BucketState::Subregion &region = state.subregions[block];
                region.collector.append(scan, id);
            }
}

/**
 * Pick the smallest possible power of 2 size for a microblock.
 * The limitation is thta there must be at most @a maxSplit microblocks.
 */
Node::size_type chooseMicroSize(
    const Node::size_type dims[3], std::size_t maxSplit)
{
    Node::size_type microSize = 1;
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
