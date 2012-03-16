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
    MLSGPU_ASSERT(!empty(), std::runtime_error);
    bufferCur++;
    cur++;
    refill();
    return *this;
}

void SimpleFileSet::MySplatStream::reset(splat_id first, splat_id last)
{
    MLSGPU_ASSERT(first <= last, std::invalid_argument);
    this->first = first;
    this->last = last;
    bufferCur = 0;
    bufferEnd = 0;
    next = first;
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

} // namespace internal


void SubsetBase::addBlob(const BlobInfo &blob)
{
    if (blobRanges.empty() || blobRanges.back().second != blob.id)
        blobRanges.push_back(std::make_pair(blob.id, blob.id + 1));
    else
        blobRanges.back().second++;
    nSplats += blob.numSplats;
}

void SubsetBase::swap(SubsetBase &other)
{
    blobRanges.swap(other.blobRanges);
    std::swap(nSplats, other.nSplats);
}

void SubsetBase::MyBlobStream::refill()
{
    while (child->empty() && blobRange < owner.blobRanges.size())
    {
        const std::pair<blob_id, blob_id> &range = owner.blobRanges[blobRange];
        child->reset(range.first, range.second);
        blobRange++;
    }
}

} // namespace SplatSet
