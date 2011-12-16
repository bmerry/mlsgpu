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
    typedef boost::numeric::converter<
        int,
        float,
        boost::numeric::conversion_traits<int, float>,
        boost::numeric::def_overflow_handler,
        boost::numeric::Ceil<float> > RoundUp;
    typedef boost::numeric::converter<
        int,
        float,
        boost::numeric::conversion_traits<int, float>,
        boost::numeric::def_overflow_handler,
        boost::numeric::Floor<float> > RoundDown;

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

        float vlo[3], vhi[3];
        grid.worldToVertex(lo, vlo);
        grid.worldToVertex(hi, vhi);

        // Start with the deepest level, then coarsen until we don't
        // take more than maxAmplify cells.
        int ilo[3], ihi[3];
        unsigned int shift = 0;
        for (unsigned int i = 0; i < 3; i++)
        {
            ilo[i] = RoundUp::convert(vlo[i]);
            ihi[i] = RoundDown::convert(vhi[i]);
            MLSGPU_ASSERT(ihi[i] >= 0 && ihi[i] < (int) dims[i], std::out_of_range);
            MLSGPU_ASSERT(ilo[i] >= 0 && ilo[i] < (int) dims[i], std::out_of_range);
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
        unsigned int level = maxLevel - shift;

        // Create entries for sorting
        Entry e;
        e.splatId = splatId;
        e.level = level;
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

    command_type *commands = allocateCommands(numCommands);
    std::vector<command_type> start(1, -1);

    // Build command list
    command_type curCommand = 0;
    std::size_t p = 0;
    for (unsigned int level = 0; level < numLevels; level++)
    {
        code_type levelSize = code_type(1U) << (3 * level);
        std::vector<command_type> prevStart(levelSize);
        prevStart.swap(start);
        for (code_type code = 0; code < levelSize; code++)
        {
            std::size_t q = p;
            while (entries[q].level == level && entries[q].code == code)
                q++;
            command_type up = prevStart[code >> 3];
            command_type first = up;
            if (p < q)
            {
                // non-empty octree cell, with entries [p, q)
                first = curCommand;
                for (std::size_t i = 0; i < q - p; i++)
                    commands[curCommand++] = entries[p + i].splatId;
                commands[curCommand++] = (up == -1) ? -1 : -2 - up; // terminator or jump
                p = q;
            }
            start[code] = first;
        }
    }

    // Transfer start array to backing store
    command_type *realStart = allocateStart(start.size());
    std::copy(start.begin(), start.end(), realStart);
}
