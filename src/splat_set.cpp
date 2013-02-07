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
#include <algorithm>
#include <iosfwd>
#include <utility>
#include <stdexcept>
#include "splat_set.h"
#include "errors.h"
#include "misc.h"
#include "timeplot.h"

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
                    float spacing, const DownDivider &bucketDivider,
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
        lower[i] = bucketDivider(loCell);
        upper[i] = bucketDivider(hiCell);
    }
}

} // namespace detail

BlobInfo SimpleBlobStream::operator*() const
{
    MLSGPU_ASSERT(!empty(), state_error);
    BlobInfo ans;
    ans.firstSplat = currentId;
    ans.lastSplat = currentId + 1;
    detail::splatToBuckets(current, grid, bucketSize, ans.lower, ans.upper);
    return ans;
}

BlobStream &SimpleBlobStream::operator++()
{
    std::size_t n = splatStream->read(&current, &currentId, 1);
    if (n == 0)
        current.radius = -1.0f; //sentinel to mark the stream as empty
    return *this;
}

const unsigned int FileSet::scanIdShift = 40;
const splat_id FileSet::splatIdMask = (splat_id(1) << scanIdShift) - 1;
// An extra bit is subtracted because other bits of code use the top bit for a flag
const std::size_t FileSet::maxFiles = std::size_t(1) << (std::numeric_limits<splat_id>::digits - 1 - scanIdShift);
/* Would probably be safe to add 1, but I haven't tested the effects of having splats from
 * different files have adjacent IDs. It's a big enough number anyway.
 */
const std::size_t FileSet::maxFileSplats = FileSet::splatIdMask;

void FileSet::addFile(FastPly::ReaderBase *file)
{
    files.push_back(file);
    nSplats += file->size();
}

FileSet::ReaderThreadBase::ReaderThreadBase(const FileSet &owner) :
    owner(owner), outQueue(), buffer("mem.FileSet.ReaderThread.buffer", owner.bufferSize),
    tworker("reader")
{
}

void FileSet::ReaderThreadBase::free(const Item &item)
{
    if (item.alloc)
        buffer.free(*item.alloc);
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
    const FileSet &owner, ReaderThreadBase *reader, bool useOMP)
:
    owner(owner), curItem(), pos(0),
    readerThread(reader), thread(boost::ref(*readerThread)),
    useOMP(useOMP)
{
}

std::size_t FileSet::MySplatStream::read(Splat *splats, splat_id *splatIds, std::size_t count)
{
    std::size_t oldCount = count;
    while (count > 0)
    {
        while (curItem.ptr == NULL || pos == curItem.last)
        {
            readerThread->free(curItem);
            curItem = readerThread->pop();
            if (curItem.ptr == NULL)
                return oldCount - count; // end of stream
            pos = curItem.first;
        }

        const std::size_t fileId = curItem.first >> scanIdShift;
        const FastPly::ReaderBase &file = owner.files[fileId];

        // Try a parallel load + decode, and fall back if there are non-finites
        const std::size_t n = std::min(curItem.last - pos, (splat_id) count);
        const std::size_t offset = pos - curItem.first;
        bool nonFinite = false;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (useOMP && n > 16384) reduction(||:nonFinite) shared(file, splats, splatIds) default(none)
#endif
        for (std::size_t i = 0; i < n; i++)
        {
            splats[i] = file.decode(curItem.ptr, offset + i);
            if (splatIds != NULL)
                splatIds[i] = pos + i;
            nonFinite = nonFinite || !splats[i].isFinite();
        }

        std::size_t p;
        if (nonFinite)
        {
            /* Need to compact out the non-finite ones. This could also be parallelised
             * by keeping a per-thread count and prefix summing it, but it's a rare
             * event so is probably not worth the cost.
             */
            p = 0;
            for (std::size_t i = 0; i < n; i++)
            {
                if (splats[i].isFinite())
                {
                    splats[p] = splats[i];
                    splatIds[p] = splatIds[i];
                    p++;
                }
            }
        }
        else
            p = n;

        pos += n;
        splats += p;
        if (splatIds != NULL)
            splatIds += p;
        count -= p;
    }
    return oldCount - count;
}

FileSet::MySplatStream::~MySplatStream()
{
    readerThread->free(curItem);
    readerThread->drain();
    thread.join();
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

void SubsetBase::addRange(splat_id first, splat_id last)
{
    MLSGPU_ASSERT(this->last <= first, std::invalid_argument);
    if (this->last == first)
        this->last = last;
    else
    {
        flush();
        this->first = first;
        this->last = last;
    }
    nSplats += last - first;
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
