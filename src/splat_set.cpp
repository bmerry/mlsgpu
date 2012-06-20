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

BlobInfo SimpleBlobStream::operator*() const
{
    BlobInfo ans;
    ans.firstSplat = splatStream->currentId();
    ans.lastSplat = ans.firstSplat + 1;
    splatToBuckets(**splatStream, grid, bucketSize, ans.lower, ans.upper);
    return ans;
}

const unsigned int SimpleFileSet::scanIdShift = 40;
const splat_id SimpleFileSet::splatIdMask = (splat_id(1) << scanIdShift) - 1;

void SimpleFileSet::addFile(FastPly::ReaderBase *file)
{
    files.push_back(file);
    nSplats += file->size();
}

SimpleFileSet::ReaderThreadBase::ReaderThreadBase(const SimpleFileSet &owner) :
    owner(owner), outQueue(2), pool(2)
{
    for (int i = 0; i < 2; i++)
        pool.push(boost::make_shared<Item>());
}

void SimpleFileSet::ReaderThreadBase::drain()
{
    boost::shared_ptr<Item> item;
    while (!!(item = pop()))
    {
        push(item);
    }
}

SimpleFileSet::MySplatStream::MySplatStream(
    const SimpleFileSet &owner, ReaderThreadBase *reader)
: owner(owner), readerThread(reader), thread(boost::ref(*readerThread))
{
    isEmpty = false;
    bufferCur = 0;
    refill();
}

SimpleFileSet::MySplatStream::~MySplatStream()
{
    if (!isEmpty)
        readerThread->drain();
    thread.join();
}

SplatStream &SimpleFileSet::MySplatStream::operator++()
{
    MLSGPU_ASSERT(!isEmpty, std::out_of_range);
    bufferCur++;
    refill();
    return *this;
}

void SimpleFileSet::MySplatStream::refill()
{
    while (true)
    {
        while (!buffer || bufferCur == buffer->nSplats)
        {
            if (buffer)
                readerThread->push(buffer);
            buffer = readerThread->pop();
            bufferCur = 0;
            if (!buffer)
            {
                isEmpty = true;
                return;
            }
        }
        const std::size_t fileId = buffer->first >> scanIdShift;
        nextSplat = owner.files[fileId].decode(&buffer->buffer[0], bufferCur);
        if (nextSplat.isFinite())
            return;
        bufferCur++;
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
