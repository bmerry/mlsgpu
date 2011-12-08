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
    SplatTree::size_type pos; ///< Position of cell in the @c start array
    SplatTree::size_type splatId;

    bool operator<(const Entry &e) const
    {
        return pos < e.pos;
    }
};

} // anonymous namespace

SplatTree::size_type SplatTree::makeCode(size_type x, size_type y, size_type z)
{
    int shift = 0;
    size_type ans = 0;
    while (x || y || z)
    {
        unsigned int digit = (x & 1) + ((y & 1) << 1) + ((z & 1) << 2);
        ans += digit << shift;
        shift += 3;
        x >>= 1;
        y >>= 1;
        z >>= 1;
    }
    MLSGPU_ASSERT(shift < std::numeric_limits<size_type>::digits, std::range_error);
    return ans;
}

SplatTree::SplatTree(const std::vector<Splat> &splats, const Grid &grid)
    : splats(splats), grid(grid)
{
    MLSGPU_ASSERT(splats.size() <= std::numeric_limits<size_type>::max(), std::length_error);
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
    unsigned int size = std::max(std::max(grid.numVertices(0), grid.numVertices(1)), grid.numVertices(2));
    unsigned int maxLevel = 0;
    while ((1 << maxLevel) < size)
        maxLevel++;
    numLevels = maxLevel + 1;

    // We allocate this locally, because we need to read from it, and we
    // promise not to read from the return of allocateLevelStart.
    std::vector<size_type> levelStart(numLevels + 1);
    size_type numPositions = 0;
    levelStart[0] = 0;
    for (unsigned int i = 0; i < numLevels; i++)
    {
        numPositions += (1U << (3 * i));
        levelStart[i + 1] = numPositions;
    }
    size_type *start = allocateStart(numPositions + 1);

    /* Make a list of all octree entries, initially ordered by splat ID
     * TODO: this is memory-heavy, and scales O(N log N). Passes for
     * counting, scanning, emitting would avoid this, but only if
     * we allowed this function to read back data for the scan, or implemented
     * the whole lot in CL.
     */
    std::vector<Entry> entries;
    entries.reserve(8 * splats.size());
    for (size_type splatId = 0; splatId < splats.size(); splatId++)
    {
        const Splat &splat = splats[splatId];
        float radius = sqrt(splat.radiusSquared);
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
        // take more than 2 cells in any direction.
        int ilo[3], ihi[3];
        unsigned int shift = 0;
        for (unsigned int i = 0; i < 3; i++)
        {
            ilo[i] = RoundUp::convert(vlo[i]);
            ihi[i] = RoundDown::convert(vhi[i]);
            assert(ihi[i] >= 0 && ihi[i] < grid.numVertices(i));
            assert(ilo[i] >= 0 && ilo[i] < grid.numVertices(i));
            while ((ihi[i] >> shift) - (ilo[i] >> shift) > 1)
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
        for (unsigned int z = ilo[2]; z <= (unsigned int) ihi[2]; z++)
            for (unsigned int y = ilo[1]; y <= (unsigned int) ihi[1]; y++)
                for (unsigned int x = ilo[0]; x <= (unsigned int) ihi[0]; x++)
                {
                    e.pos = levelStart[level] + makeCode(x, y, z);
                    entries.push_back(e);
                }
    }

    /* Extract the entries into the persistent structures.
     */
    stable_sort(entries.begin(), entries.end());
    size_type *ids = allocateIds(entries.size());
    for (size_t i = 0; i < entries.size(); i++)
        ids[i] = entries[i].splatId;

    size_type e = 0;  // walks through the entries
    for (size_t pos = 0; pos <= levelStart[numLevels]; pos++)
    {
        start[pos] = e;
        while (e < entries.size() && entries[e].pos <= pos)
            e++;
    }

    // Copy local levelStart to persistent memory
    size_type *realLevelStart = allocateLevelStart(numLevels + 1);
    std::copy(levelStart.begin(), levelStart.end(), realLevelStart);
}
