/**
 * @file
 *
 * Implementation of @ref SplatTree.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <vector>
#include "tr1_cstdint.h"
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
    MLSGPU_ASSERT(shift < std::numeric_limits<code_type>::digits, std::out_of_range);
    return ans;
}

SplatTree::SplatTree(const std::vector<Splat> &splats,
                     const Grid::size_type size[3],
                     const Grid::difference_type offset[3])
    : splats(splats)
{
    MLSGPU_ASSERT(splats.size() < (size_t) std::numeric_limits<command_type>::max() / (2 * maxAmplify), std::length_error);
    for (unsigned int i = 0; i < 3; i++)
    {
        MLSGPU_ASSERT(size[i] < code_type(1) << (std::numeric_limits<code_type>::digits / 3), std::length_error);
        this->size[i] = size[i];
        this->offset[i] = offset[i];
    }
}

/**
 * Determines whether the cell with bounds @a c0 to @a c1 is intersected by @a splat.
 * The bounds have been pre-biased to the coordinate system of the splats.
 */
static bool splatCellIntersect(const Splat &splat, const Grid::difference_type c0[3], const Grid::difference_type c1[3])
{
    float dist[3];
    for (unsigned int i = 0; i < 3; i++)
    {
        // Find the point in the cell closest to the splat center
        float lo = c0[i];
        float hi = c1[i];
        float nearest = std::min(hi, std::max(lo, splat.position[i]));
        dist[i] = nearest - splat.position[i];
    }
    return dist[0] * dist[0] + dist[1] * dist[1] + dist[2] * dist[2] <= splat.radius * splat.radius * 1.00001;
}

void SplatTree::initialize()
{
    // Compute the number of levels and related data
    code_type maxSize = *std::max_element(size, size + 3);
    unsigned int maxLevel = 0;
    while ((1U << maxLevel) < maxSize)
        maxLevel++;
    numLevels = maxLevel + 1;

    /* Make a list of all octree entries, initially ordered by splat ID.
     */
    std::vector<Entry> entries;
    entries.reserve(maxAmplify * splats.size());
    for (std::size_t splatId = 0; splatId < splats.size(); splatId++)
    {
        const Splat &splat = splats[splatId];
        const float radius = splat.radius;
        Grid::difference_type ilo[3], ihi[3];
        for (unsigned int i = 0; i < 3; i++)
        {
            ilo[i] = Grid::RoundDown::convert(splat.position[i] - radius) - offset[i];
            ihi[i] = Grid::RoundDown::convert(splat.position[i] + radius) - offset[i];
        }

        // Start with the deepest level, then coarsen until we don't
        // take more than maxAmplify cells.
        unsigned int shift = 0;
        for (unsigned int i = 0; i < 3; i++)
        {
            ilo[i] = std::max(std::min(ilo[i], (Grid::difference_type) size[i] - 1), Grid::difference_type(0));
            ihi[i] = std::max(std::min(ihi[i], (Grid::difference_type) size[i] - 1), Grid::difference_type(0));
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
        for (Grid::difference_type z = ilo[2]; z <= ihi[2]; z++)
            for (Grid::difference_type y = ilo[1]; y <= ihi[1]; y++)
                for (Grid::difference_type x = ilo[0]; x <= ihi[0]; x++)
                {
                    Grid::difference_type c0[3], c1[3];
                    c0[0] = x << shift;
                    c0[1] = y << shift;
                    c0[2] = z << shift;
                    for (unsigned int i = 0; i < 3; i++)
                    {
                        c1[i] = std::min((Grid::difference_type) size[i], c0[i] + (1 << shift));
                        // bias both c0 and c1 back into splat coordinate system
                        c0[i] += offset[i];
                        c1[i] += offset[i];
                    }
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
     * a length and a jump command (or a terminate command).
     */
    size_t numCommands = entries.size() + 2;
    for (size_t i = 1; i < entries.size(); i++)
    {
        if (entries[i].level != entries[i - 1].level || entries[i].code != entries[i - 1].code)
            numCommands += 2;
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
            nextCommand++;
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
                commands[start[level][code]] = jumpPos[level][code];
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
