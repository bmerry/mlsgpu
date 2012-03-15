/**
 * @file
 *
 * Implementations of non-template members from @ref SplatSet.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <limits>
#include <boost/smart_ptr/scoped_ptr.hpp>
#include <iosfwd>
#include "splat_set.h"
#include "errors.h"
#include "misc.h"

namespace SplatSet
{

namespace internal
{

const unsigned int SimpleFileSet::scanIdShift;
const std::size_t SimpleFileSet::MySplatStream::bufferSize;

void splatToBuckets(const Splat &splat,
                    const Grid &grid, Grid::size_type bucketSize,
                    boost::array<Grid::difference_type, 3> &lower,
                    boost::array<Grid::difference_type, 3> &upper)
{
    MLSGPU_ASSERT(splat.isFinite(), std::invalid_argument);
    MLSGPU_ASSERT(bucketSize > 0, std::invalid_argument);
    float loWorld[3], hiWorld[3];
    Grid::difference_type lo[3], hi[3];
    for (unsigned int i = 0; i < 3; i++)
    {
        loWorld[i] = splat.position[i] - splat.radius;
        hiWorld[i] = splat.position[i] + splat.radius;
    }
    grid.worldToCell(loWorld, lo);
    grid.worldToCell(hiWorld, hi);
    for (unsigned int i = 0; i < 3; i++)
    {
        lower[i] = divDown(lo[i], bucketSize);
        upper[i] = divDown(hi[i], bucketSize);
    }
}

BlobInfo SimpleBlobStream::operator*() const
{
    BlobInfo ans;
    ans.numSplats = 1;
    ans.id = splatStream->currentId();
    splatToBuckets(**splatStream, grid, bucketSize, ans.lower, ans.upper);
    return ans;
}

BlobInfo SimpleBlobStreamReset::operator*() const
{
    BlobInfo ans;
    ans.numSplats = 1;
    ans.id = splatStream->currentId();
    splatToBuckets(**splatStream, grid, bucketSize, ans.lower, ans.upper);
    return ans;
}

} // namespace internal

} // namespace SplatSet
