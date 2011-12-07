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
#include "splattree.h"

namespace
{

/**
 * Transient structure only used during construction.
 * It cannot be declared locally in the constructor because otherwise
 * <code>vector&lt;Entry&gt;</code> gives issues.
 */
struct Entry
{
    unsigned int level;
    SplatTree::size_type code;
    SplatTree::size_type splatId;

    bool operator<(const Entry &e) const
    {
        if (level != e.level)
            return level < e.level;
        else
            return code < e.code;
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

    // Compute the number of levels
    unsigned int size = std::max(std::max(grid.numVertices(0), grid.numVertices(1)), grid.numVertices(2));
    unsigned int maxLevel = 0;
    while ((1 << maxLevel) < size)
        maxLevel++;

    std::vector<Entry> entries;
    entries.reserve(8 * splats.size());
    levels.resize(maxLevel + 1);
    for (unsigned int i = 0; i <= maxLevel; i++)
        levels[i].start.resize(1 << (3 * i));

    // Enter splats into the octree
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
        assert(shift <= maxLevel);
        for (unsigned int i = 0; i < 3; i++)
        {
            ilo[i] >>= shift;
            ihi[i] >>= shift;
        }

        Entry e;
        e.level = maxLevel - shift;
        e.splatId = splatId;
        for (unsigned int z = ilo[2]; z <= (unsigned int) ihi[2]; z++)
            for (unsigned int y = ilo[1]; y <= (unsigned int) ihi[1]; y++)
                for (unsigned int x = ilo[0]; x <= (unsigned int) ihi[0]; x++)
                {
                    e.code = makeCode(x, y, z);
                    entries.push_back(e);
                }
    }

    stable_sort(entries.begin(), entries.end());
    int lastLevel = -1;
    size_type lastCode = 0;
    ids.reserve(entries.size());
    for (size_t i = 0; i < entries.size(); i++)
    {
        const Entry &e = entries[i];
        ids.push_back(e.splatId);
        if ((int) e.level != lastLevel || e.code != lastCode)
        {
            levels[e.level].start[e.code] = i;
            lastLevel = e.level;
            lastCode = e.code;
        }
    }
}
