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
#include <boost/smart_ptr/shared_ptr.hpp>
#include <boost/smart_ptr/make_shared.hpp>
#include <iosfwd>
#include "splat_set.h"
#include "errors.h"
#include "misc.h"

namespace SplatSet
{

namespace detail
{

/// Range of [0, max splat id), so back a reader over the entire set
const std::pair<splat_id, splat_id> rangeAll(0, std::numeric_limits<splat_id>::max());

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

void splatToBuckets(const Splat &splat,
                    float spacing, Grid::size_type bucketSize,
                    boost::array<Grid::difference_type, 3> &lower,
                    boost::array<Grid::difference_type, 3> &upper)
{
    MLSGPU_ASSERT(splat.isFinite(), std::invalid_argument);
    MLSGPU_ASSERT(bucketSize > 0, std::invalid_argument);
    for (unsigned int i = 0; i < 3; i++)
    {
        float loWorld = splat.position[i] - splat.radius;
        float hiWorld = splat.position[i] + splat.radius;
        Grid::difference_type loCell = Grid::RoundDown::convert(loWorld / spacing);
        Grid::difference_type hiCell = Grid::RoundDown::convert(hiWorld / spacing);
        lower[i] = divDown(loCell, bucketSize);
        upper[i] = divDown(hiCell, bucketSize);
    }
}

std::size_t loadBuffer(SplatStream *splats, Statistics::Container::vector<std::pair<Splat, splat_id> > &buffer)
{
    std::size_t nBuffer = 0;
    while (nBuffer < buffer.size() && !splats->empty())
    {
        buffer[nBuffer].first = **splats;
        buffer[nBuffer].second = splats->currentId();
        nBuffer++;
        ++*splats;
    }
    return nBuffer;
}

} // namespace detail

BlobInfo SimpleBlobStream::operator*() const
{
    BlobInfo ans;
    ans.firstSplat = splatStream->currentId();
    ans.lastSplat = ans.firstSplat + 1;
    detail::splatToBuckets(**splatStream, grid, bucketSize, ans.lower, ans.upper);
    return ans;
}

const unsigned int FileSet::scanIdShift = 40;
const splat_id FileSet::splatIdMask = (splat_id(1) << scanIdShift) - 1;

void FileSet::addFile(FastPly::ReaderBase *file)
{
    files.push_back(file);
    nSplats += file->size();
}

FileSet::ReaderThreadBase::ReaderThreadBase(const FileSet &owner) :
    owner(owner), outQueue(256), buffer("mem.FileSet.ReaderThread.buffer", owner.bufferSize)
{
}

void FileSet::ReaderThreadBase::free(const Item &item)
{
    buffer.free(item.ptr, item.bytes);
}

void FileSet::ReaderThreadBase::drain()
{
    Item item;
    while ((item = pop()).ptr != NULL)
    {
        free(item);
    }
}

FileSet::MySplatStream::MySplatStream(
    const FileSet &owner, ReaderThreadBase *reader)
: owner(owner), readerThread(reader), thread(boost::ref(*readerThread))
{
    isEmpty = false;
    bufferCur = 0;
    refill();
}

FileSet::MySplatStream::~MySplatStream()
{
    if (buffer.ptr)
        readerThread->free(buffer);
    if (!isEmpty)
        readerThread->drain();
    thread.join();
}

SplatStream &FileSet::MySplatStream::operator++()
{
    MLSGPU_ASSERT(!isEmpty, std::out_of_range);
    bufferCur++;
    refill();
    return *this;
}

void FileSet::MySplatStream::refill()
{
    while (true)
    {
        while (!buffer.ptr || bufferCur == buffer.numSplats())
        {
            if (buffer.ptr)
                readerThread->free(buffer);
            buffer = readerThread->pop();
            bufferCur = 0;
            if (!buffer.ptr)
            {
                isEmpty = true;
                return;
            }
        }
        const std::size_t fileId = buffer.first >> scanIdShift;
        nextSplat = owner.files[fileId].decode(buffer.ptr, bufferCur);
        if (nextSplat.isFinite())
            return;
        bufferCur++;
    }
}


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
