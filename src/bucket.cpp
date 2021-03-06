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
 * Bucketing of splats into sufficiently small buckets.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <vector>
#include "tr1_cstdint.h"
#include <limits>
#include <stdexcept>
#include <algorithm>
#include <utility>
#include <cassert>
#include <functional>
#include <boost/tr1/cmath.hpp>
#include <boost/array.hpp>
#include <boost/multi_array.hpp>
#include <boost/foreach.hpp>
#include <boost/ref.hpp>
#include <boost/numeric/conversion/converter.hpp>
#include <boost/smart_ptr/make_shared.hpp>
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

namespace detail
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

const std::size_t BucketState::BAD_REGION = (std::size_t) -1;

boost::array<Grid::size_type, 3> BucketState::computeDims(const Grid &grid, Grid::size_type microSize)
{
    boost::array<Grid::size_type, 3> dims;
    for (int i = 0; i < 3; i++)
        dims[i] = divUp(grid.numCells(i), microSize);
    return dims;
}

BucketState::BucketState(
    const BucketParameters &params, const Grid &grid,
    Grid::size_type microSize, int macroLevels)
    : params(params), grid(grid), microSize(microSize), macroLevels(macroLevels),
    dims(computeDims(grid, microSize)),
    subregions("mem.BucketState::subregions")
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
        nodeCounts.push_back(new node_count_type("mem.BucketState::nodeCounts"));
    }
}

void BucketState::upsweepCounts()
{
    for (int level = 0; level + 1 < macroLevels; level++)
    {
        BOOST_FOREACH(node_count_type::const_reference v, nodeCounts[level])
        {
            HashCoord::arg_type parent;
            for (int i = 0; i < 3; i++)
                parent[i] = v.first[i] >> 1;
            nodeCounts[level + 1][parent].numSplats += v.second.numSplats;
        }
    }
}

bool BucketState::clamp(const boost::array<Grid::difference_type, 3> &lower,
                        const boost::array<Grid::difference_type, 3> &upper,
                        boost::array<Node::size_type, 3> &lo,
                        boost::array<Node::size_type, 3> &hi)
{
    for (unsigned int i = 0; i < 3; i++)
    {
        Grid::difference_type l = lower[i];
        Grid::difference_type h = upper[i];
        if (l < 0)
            l = 0;
        if (h >= Grid::difference_type(dims[i]))
            h = Grid::difference_type(dims[i]) - 1;
        if (l > h)
            return false; // does not intersect the grid
        lo[i] = l;
        hi[i] = h;
    }
    return true;
}

std::tr1::int64_t BucketState::getNodeCount(const Node &node) const
{
    assert(node.getLevel() < nodeCounts.size());
    const node_count_type &nc = nodeCounts[node.getLevel()];
    node_count_type::const_iterator pos = nc.find(node.getCoords());
    if (pos == nc.end())
        return 0;
    else
        return pos->second.numSplats;
}

void BucketState::countSplats(const SplatSet::BlobInfo &blob, std::tr1::uint64_t &numUpdates)
{
    int level = 0;
    boost::array<Node::size_type, 3> lo, hi;
    if (!clamp(blob.lower, blob.upper, lo, hi))
        return;
    SplatSet::splat_id numSplats = blob.lastSplat - blob.firstSplat;
    for (Node::size_type x = lo[0]; x <= hi[0]; x++)
        for (Node::size_type y = lo[1]; y <= hi[1]; y++)
            for (Node::size_type z = lo[2]; z <= hi[2]; z++)
            {
                const HashCoord::arg_type coords = {{ x, y, z }};
                nodeCounts[level][coords].numSplats += numSplats;
                numUpdates++;
            }
    while (level + 1 < macroLevels && (lo[0] < hi[0] || lo[1] < hi[1] || lo[2] < hi[2]))
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
                    const HashCoord::arg_type coords = {{ x, y, z }};
                    nodeCounts[level][coords].numSplats -= (hits - 1) * numSplats;
                    numUpdates++;
                }
        for (unsigned int i = 0; i < 3; i++)
        {
            lo[i] >>= 1;
            hi[i] >>= 1;
        }
    }
}

void BucketState::pickNodes()
{
    /* Select cells to bucket splats into */
    forEachNode(getDims(), macroLevels, PickNodes(*this));

    /* Compute subregion for descendants of the cut */
    for (int lvl = macroLevels - 2; lvl >= 0; lvl--)
    {
        BOOST_FOREACH(node_count_type::reference n, nodeCounts[lvl])
        {
            if (n.second.subregion == BAD_REGION)
            {
                HashCoord::arg_type pcoord;
                for (int i = 0; i < 3; i++)
                    pcoord[i] = n.first[i] >> 1;
                node_count_type::iterator parent = nodeCounts[lvl + 1].find(pcoord);
                if (parent != nodeCounts[lvl + 1].end())
                    n.second.subregion = parent->second.subregion;
            }
        }
    }
}

void BucketState::bucketSplats(const SplatSet::BlobInfo &blob)
{
    boost::array<Node::size_type, 3> lo, hi;
    if (!clamp(blob.lower, blob.upper, lo, hi))
        return;

    for (Node::size_type x = lo[0]; x <= hi[0]; x++)
        for (Node::size_type y = lo[1]; y <= hi[1]; y++)
            for (Node::size_type z = lo[2]; z <= hi[2]; z++)
            {
                const HashCoord::arg_type coord = {{ x, y, z }};
                node_count_type::iterator pos = nodeCounts[0].find(coord);
                assert(pos != nodeCounts[0].end());
                std::size_t regionId = pos->second.subregion;
                assert(regionId < subregions.size());
                BucketState::Subregion &region = subregions[regionId];

                /* Only add once per node */
                const Node::size_type nodeSize = region.node.size();
                const Node::size_type mask = nodeSize - 1;
                if ((x == lo[0] || ((x & mask) == 0))
                    && (y == lo[1] || (y & mask) == 0)
                    && (z == lo[2] || (z & mask) == 0))
                {
                    region.subset.addBlob(blob);
                }
            }
}

BucketStateSet::BucketStateSet(
    const boost::array<Grid::difference_type, 3> &chunks,
    Grid::difference_type chunkCells,
    const BucketParameters &params,
    const Grid &grid,
    Grid::size_type microSize,
    int macroLevels)
    : Statistics::Container::multi_array<boost::shared_ptr<BucketState>, 3>("mem.BucketStateSet", chunks),
    chunkRatio(chunkCells / microSize),
    chunkDivider(chunkRatio)
{
    MLSGPU_ASSERT(chunkCells % microSize == 0, std::invalid_argument);
    boost::array<Grid::difference_type, 3> chunkCoord;
    for (chunkCoord[2] = 0; chunkCoord[2] < chunks[2]; chunkCoord[2]++)
        for (chunkCoord[1] = 0; chunkCoord[1] < chunks[1]; chunkCoord[1]++)
            for (chunkCoord[0] = 0; chunkCoord[0] < chunks[0]; chunkCoord[0]++)
            {
                Grid sub = grid;
                for (unsigned int i = 0; i < 3; i++)
                {
                    Grid::difference_type low = grid.getExtent(i).first;
                    Grid::difference_type high = grid.getExtent(i).second;
                    Grid::difference_type offset = chunkCoord[i] * chunkCells;
                    sub.setExtent(i, low + offset,
                                  std::min(low + offset + chunkCells, high));
                }
                (*this)(chunkCoord) = boost::make_shared<BucketState>(params, sub, microSize, macroLevels);
            }
}

bool PickNodes::operator()(const Node &node) const
{
    std::tr1::uint64_t count = state.getNodeCount(node);

    if (count == 0)
        return false;  // skip empty space

    if (node.getLevel() == 0
        || ((state.microSize * node.size() <= state.params.maxCells)
            && count <= state.params.maxSplats))
    {
        std::size_t id = state.subregions.size();
        state.nodeCounts[node.getLevel()][node.getCoords()].subregion = id;
        state.subregions.push_back(BucketState::Subregion(node));
        return false; // no more recursion required
    }
    else
        return true;
}

Grid::size_type chooseMicroSize(
    const Grid::size_type dims[3],
    std::size_t maxSplit,
    std::tr1::uint64_t numSplats,
    std::tr1::uint64_t maxSplats,
    Grid::size_type maxCells)
{
    Grid::size_type microSize = 1;
    std::size_t microBlocks = 1;
    for (int i = 0; i < 3; i++)
        microBlocks = mulSat(microBlocks, std::size_t(divUp(dims[i], microSize)));

    // Assume all the splats are in an axis-aligned plane
    double target = 0.5 * *std::min_element(dims, dims + 3) * sqrt((double) maxSplats / numSplats);
    while (microBlocks > maxSplit
           || (microBlocks > 8 && microSize < target * 0.5 && microSize * 2 <= maxCells))
    {
        microSize *= 2;
        microBlocks = 1;
        for (int i = 0; i < 3; i++)
            microBlocks = mulSat(microBlocks, std::size_t(divUp(dims[i], microSize)));
    }
    return microSize;
}

} // namespace detail

} // namespace Bucket
