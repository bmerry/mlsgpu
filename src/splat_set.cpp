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

void SubsetBase::flush()
{
    if (first != last)
    {
        nRanges++;
        if (first - prev < (1 << 16)
            && last - first < (1 << 15))
        {
            std::tr1::uint32_t encoded = first - prev;
            encoded |= (last - first) << 16;
            encoded |= std::tr1::uint32_t(1) << 31;
            splatRanges.push_back(encoded);
        }
        else
        {
            splatRanges.push_back(first >> 32);
            splatRanges.push_back(first & UINT32_C(0xFFFFFFFF));
            splatRanges.push_back(last >> 32);
            splatRanges.push_back(last & UINT32_C(0xFFFFFFFF));
        }
        prev = last;
        first = last;
    }
}

void SubsetBase::addBlob(const BlobInfo &blob)
{
    MLSGPU_ASSERT(last <= blob.firstSplat, std::invalid_argument);
    if (last == blob.firstSplat)
        last = blob.lastSplat;
    else
    {
        flush();
        first = blob.firstSplat;
        last = blob.lastSplat;
    }
    nSplats += blob.lastSplat - blob.firstSplat;
}

void SubsetBase::swap(SubsetBase &other)
{
    splatRanges.swap(other.splatRanges);
    std::swap(first, other.first);
    std::swap(last, other.last);
    std::swap(prev, other.prev);
    std::swap(nSplats, other.nSplats);
    std::swap(nRanges, other.nRanges);
}

SubsetBase::const_iterator SubsetBase::begin() const
{
    MLSGPU_ASSERT(first == last, state_error);
    return const_iterator(0, splatRanges.begin());
}

SubsetBase::const_iterator SubsetBase::end() const
{
    MLSGPU_ASSERT(first == last, state_error);
    return const_iterator(prev, splatRanges.end());
}

void SubsetBase::const_iterator::increment()
{
    if (*pos & (std::tr1::uint32_t(1) << 31))
    {
        // Differential
        prev += (*pos) & 0xFFFF;
        prev += (*pos >> 16) & 0x7FFF;
        ++pos;
    }
    else
    {
        // Full encoding
        pos += 2;
        prev = *pos;
        prev <<= 32;
        ++pos;
        prev |= *pos;
        ++pos;
    }
}

bool SubsetBase::const_iterator::equal(const const_iterator &other) const
{
    return pos == other.pos;
}

std::pair<splat_id, splat_id> SubsetBase::const_iterator::dereference() const
{
    if (*pos & (std::tr1::uint32_t(1) << 31))
    {
        // Differential
        splat_id offset = *pos & 0xFFFF;
        splat_id length = (*pos >> 16) & 0x7FFF;
        splat_id first = prev + offset;
        return std::make_pair(first, first + length);
    }
    else
    {
        Statistics::Container::vector<std::tr1::uint32_t>::const_iterator p = pos;
        splat_id first = *p++; first <<= 32;
        first |= *p++;
        splat_id last = *p++; last <<= 32;
        last |= *p++;
        return std::make_pair(first, last);
    }
}

} // namespace SplatSet
