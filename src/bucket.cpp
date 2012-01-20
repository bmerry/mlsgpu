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
#include "splat.h"
#include "bucket.h"

SplatRange::SplatRange() :
    scan(std::numeric_limits<scan_type>::max()),
    size(0),
    start(std::numeric_limits<index_type>::max())
{
}

SplatRange::SplatRange(scan_type scan, index_type splat) :
    scan(scan),
    size(1),
    start(splat)
{
}

bool SplatRange::append(scan_type scan, index_type splat)
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

SplatRangeCounter::SplatRangeCounter() : ranges(0), splats(0), current()
{
}

void SplatRangeCounter::append(SplatRange::scan_type scan, SplatRange::index_type splat)
{
    splats++;
    /* On the first call, the append will succeed (empty range), but we still
     * need to set ranges to 1 since this is the first real range.
     */
    if (ranges == 0 || !current.append(scan, splat))
    {
        current = SplatRange(scan, splat);
        ranges++;
    }
}

std::tr1::uint64_t SplatRangeCounter::countRanges() const
{
    return ranges;
}

std::tr1::uint64_t SplatRangeCounter::countSplats() const
{
    return splats;
}
