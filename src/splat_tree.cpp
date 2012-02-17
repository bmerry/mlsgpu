/**
 * @file
 *
 * Implementation of @ref SplatTree.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <vector>
#include <tr1/cstdint>
#include <algorithm>
#include <stdexcept>
#include <limits>
#include "errors.h"
#include "grid.h"
#include "splat.h"
#include "splat_tree.h"
#include "misc.h"

namespace
{

/**
 * Transient structure only used during construction.
 * It cannot be declared locally in the constructor because otherwise
 * <code>vector&lt;Entry&gt;</code> gives issues.
 */
struct Entry
{
    unsigned int level;               ///< Octree level
    SplatTree::code_type code;        ///< Position code
    SplatTree::command_type splatId;  ///< Original splat ID

    /**
     * Sort by level then by code.
     */
    bool operator<(const Entry &e) const
    {
        if (level != e.level) return level < e.level;
        else return code < e.code;
    }
};

} // anonymous namespace

SplatTree::code_type SplatTree::makeCode(code_type x, code_type y, code_type z)
{
    int shift = 0;
    code_type ans = 0;
    while (x || y || z)
    {
        unsigned int digit = (x & 1) + ((y & 1) << 1) + ((z & 1) << 2);
        ans += digit << shift;
        shift += 3;
        x >>= 1;
        y >>= 1;
        z >>= 1;
    }
    MLSGPU_ASSERT(shift < std::numeric_limits<code_type>::digits, std::range_error);
    return ans;
}

SplatTree::SplatTree(const std::vector<Splat> &splats, const Grid &grid)
    : splats(splats), grid(grid)
{
    MLSGPU_ASSERT(splats.size() < (size_t) std::numeric_limits<command_type>::max() / (2 * maxAmplify), std::length_error);
}

static bool splatCellIntersect(const Splat &splat, const float c0[3], const float c1[3])
{
    float dist[3];
    for (unsigned int i = 0; i < 3; i++)
    {
        float lo = std::min(c0[i], c1[i]);
        float hi = std::max(c0[i], c1[i]);
        // Find the point in the cell closest to the splat center
        float nearest = std::min(hi, std::max(lo, splat.position[i]));
        dist[i] = nearest - splat.position[i];
    }
    return dist[0] * dist[0] + dist[1] * dist[1] + dist[2] * dist[2] <= splat.radius * splat.radius * 1.00001;
}

void SplatTree::initialize()
{
    // Compute the number of levels and related data
    code_type dims[3];
    for (unsigned int i = 0; i < 3; i++)
        dims[i] = grid.numVertices(i);
    code_type size = *std::max_element(dims, dims + 3);
    unsigned int maxLevel = 0;
    while ((1U << maxLevel) < size)
        maxLevel++;
    numLevels = maxLevel + 1;

    /* Make a list of all octree entries, initially ordered by splat ID. TODO:
     * this is memory-heavy, and takes O(N log N) time for sorting. Passes for
     * counting, scanning, emitting would avoid this, but only if we allowed
     * this function to read back data for the scan, or implemented the whole
     * lot in CL.
     */
    std::vector<Entry> entries;
    entries.reserve(maxAmplify * splats.size());
    for (std::size_t splatId = 0; splatId < splats.size(); splatId++)
    {
        const Splat &splat = splats[splatId];
        const float radius = splat.radius;
        float lo[3], hi[3];
        for (unsigned int i = 0; i < 3; i++)
        {
            lo[i] = splat.position[i] - radius;
            hi[i] = splat.position[i] + radius;
        }

        Grid::difference_type ilo[3], ihi[3];
        grid.worldToCell(lo, ilo);
        grid.worldToCell(hi, ihi);

        // Start with the deepest level, then coarsen until we don't
        // take more than maxAmplify cells.
        unsigned int shift = 0;
        for (unsigned int i = 0; i < 3; i++)
        {
            ilo[i] = std::max(std::min(ilo[i] + 1, (Grid::difference_type) dims[i] - 1), Grid::difference_type(0));
            ihi[i] = std::max(std::min(ihi[i], (Grid::difference_type) dims[i] - 1), Grid::difference_type(0));
        }
        while (true)
        {
            size_t sz[3];
            for (unsigned int i = 0; i < 3; i++)
            {
                sz[i] = (ihi[i] >> shift) - (ilo[i] >> shift) + 1;
            }
            if (sz[0] <= maxAmplify && sz[1] <= maxAmplify && sz[2] <= maxAmplify
                && sz[0] * sz[1] * sz[2] <= maxAmplify)
                break;
            shift++;
        }

        // Check we haven't gone right past the coarsest level
        assert(shift < numLevels);

        for (unsigned int i = 0; i < 3; i++)
        {
            ilo[i] >>= shift;
            ihi[i] >>= shift;
        }

        // Create entries for sorting
        Entry e;
        e.splatId = splatId;
        e.level = shift;
        for (unsigned int z = ilo[2]; z <= (unsigned int) ihi[2]; z++)
            for (unsigned int y = ilo[1]; y <= (unsigned int) ihi[1]; y++)
                for (unsigned int x = ilo[0]; x <= (unsigned int) ihi[0]; x++)
                {
                    float c0[3], c1[3];
                    int ic0[3], ic1[3];
                    ic0[0] = x << shift;
                    ic0[1] = y << shift;
                    ic0[2] = z << shift;
                    for (unsigned int i = 0; i < 3; i++)
                    {
                        ic1[i] = std::min((int) dims[i], ic0[i] + (1 << shift)) - 1;
                    }
                    grid.getVertex(ic0[0], ic0[1], ic0[2], c0);
                    grid.getVertex(ic1[0], ic1[1], ic1[2], c1);
                    if (splatCellIntersect(splat, c0, c1))
                    {
                        // Check that the sphere hits the cell, not just the bbox
                        // This is also where splats outside the grid get rejected
                        e.code = makeCode(x, y, z);
                        entries.push_back(e);
                    }
                }
    }
    stable_sort(entries.begin(), entries.end());

    /* Determine memory requirements. Each distinct sort key requires
     * a jump command (or a terminate command).
     */
    size_t numCommands = entries.size() + 1;
    for (size_t i = 1; i < entries.size(); i++)
    {
        if (entries[i].level != entries[i - 1].level || entries[i].code != entries[i - 1].code)
            numCommands++;
    }

    // Build command list, excluding jumps
    // Also build direct entries in start and record jump slots
    command_type *commands = allocateCommands(numCommands);
    size_t nextCommand = 0;
    std::vector<std::vector<command_type> > start(numLevels + 1), jumpPos(numLevels);
    for (unsigned int i = 0; i < numLevels; i++)
    {
        std::size_t levelSize = size_t(1) << (3 * (numLevels - 1 - i));
        start[i].resize(levelSize, -1);
        jumpPos[i].resize(levelSize, -1);
    }
    for (size_t i = 0; i < entries.size(); i++)
    {
        if (i == 0
            || entries[i].level != entries[i - 1].level
            || entries[i].code != entries[i - 1].code)
        {
            start[entries[i].level][entries[i].code] = nextCommand;
        }
        commands[nextCommand++] = entries[i].splatId;

        if (i + 1 == entries.size()
            || entries[i].level != entries[i + 1].level
            || entries[i].code != entries[i + 1].code)
        {
            jumpPos[entries[i].level][entries[i].code] = nextCommand;
            nextCommand++;
        }
    }
    assert(nextCommand = numCommands);

    // build jumps and start array
    start[numLevels].resize(1, -1); // sentinel
    for (int level = numLevels - 1; level >= 0; level--)
    {
        std::size_t levelSize = size_t(1) << (3 * (numLevels - 1 - level));
        for (code_type code = 0; code < levelSize; code++)
        {
            command_type up = start[level + 1][code >> 3];
            if (jumpPos[level][code] != -1)
            {
                commands[jumpPos[level][code]] = up == -1 ? -1 : -2 - up;
            }
            else
            {
                start[level][code] = up;
            }
        }
    }

    // Transfer start array to backing store
    command_type *realStart = allocateStart(start[0].size());
    std::copy(start[0].begin(), start[0].end(), realStart);
}
