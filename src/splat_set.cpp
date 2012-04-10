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

namespace detail
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
    ans.firstSplat = splatStream->currentId();
    ans.lastSplat = ans.firstSplat + 1;
    splatToBuckets(**splatStream, grid, bucketSize, ans.lower, ans.upper);
    return ans;
}

void SimpleVectorSet::MySplatStream::skipNonFinite()
{
    while (cur < last && !owner[cur].isFinite())
        cur++;
}


void SimpleFileSet::addFile(FastPly::Reader *file)
{
    files.push_back(file);
    nSplats += file->size();
}

SplatStream &SimpleFileSet::MySplatStream::operator++()
{
    MLSGPU_ASSERT(!empty(), std::out_of_range);
    bufferCur++;
    cur++;
    refill();
    return *this;
}

void SimpleFileSet::MySplatStream::reset(splat_id first, splat_id last)
{
    MLSGPU_ASSERT(first <= last, std::invalid_argument);
    last = std::min(last, splat_id(owner.files.size()) << scanIdShift);
    this->last = last;
    bufferCur = 0;
    bufferEnd = 0;
    cur = next = first;
    refill();
}

void SimpleFileSet::MySplatStream::skipNonFiniteInBuffer()
{
    while (bufferCur < bufferEnd && !buffer[bufferCur].isFinite())
    {
        bufferCur++;
        cur++;
    }
}

void SimpleFileSet::MySplatStream::refill()
{
    skipNonFiniteInBuffer();
    while (bufferCur == bufferEnd)
    {
        std::size_t file = next >> scanIdShift;
        splat_id offset = next & splatIdMask;
        while (file < owner.files.size() && offset >= owner.files[file].size())
        {
            file++;
            offset = 0;
        }
        next = (splat_id(file) << scanIdShift) + offset;
        if (next >= last)
            break;

        bufferCur = 0;
        bufferEnd = bufferSize;
        if (last - next < bufferEnd)
            bufferEnd = last - next;
        if (owner.files[file].size() - offset < bufferEnd)
            bufferEnd = owner.files[file].size() - offset;
        assert(bufferEnd > 0);
        owner.files[file].read(offset, offset + bufferEnd, &buffer[0]);
        cur = next;
        next += bufferEnd;

        skipNonFiniteInBuffer();
    }
}

} // namespace detail


void SubsetBase::addBlob(const BlobInfo &blob)
{
    MLSGPU_ASSERT(splatRanges.empty() || splatRanges.back().second <= blob.firstSplat,
                  std::invalid_argument);
    if (splatRanges.empty() || splatRanges.back().second != blob.firstSplat)
        splatRanges.push_back(std::make_pair(blob.firstSplat, blob.lastSplat));
    else
        splatRanges.back().second = blob.lastSplat;
    nSplats += blob.lastSplat - blob.firstSplat;
}

void SubsetBase::swap(SubsetBase &other)
{
    splatRanges.swap(other.splatRanges);
    std::swap(nSplats, other.nSplats);
}

} // namespace SplatSet
