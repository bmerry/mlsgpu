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


const unsigned int SimpleFileSet::scanIdShift = 40;
const splat_id SimpleFileSet::splatIdMask = (splat_id(1) << scanIdShift) - 1;

void SimpleFileSet::addFile(FastPly::ReaderBase *file)
{
    files.push_back(file);
    nSplats += file->size();
}

SimpleFileSet::ReaderThread::ReaderThread(const SimpleFileSet &owner)
    : owner(owner), inQueue(1), outQueue(2), pool(2)
{
    for (int i = 0; i < 2; i++)
        pool.push(boost::make_shared<Item>());
}

void SimpleFileSet::ReaderThread::operator()()
{
    // TODO: instrumentation for the memory
    boost::shared_array<char> rawBuffer(new char[BUFFER_SIZE * sizeof(Splat)]);

    Request req;
    while (true)
    {
        req = inQueue.pop();
        if (req.first > req.last) // sentinel value
            break;
        splat_id first = req.first;
        splat_id last = req.last;

        boost::scoped_ptr<FastPly::ReaderBase::Handle> handle;
        std::size_t handleId;
        while (first < last)
        {
            std::size_t fileId = first >> scanIdShift;

            FastPly::ReaderBase::size_type fileSize = owner.files[fileId].size();
            FastPly::ReaderBase::size_type start = first & splatIdMask;
            FastPly::ReaderBase::size_type end = std::min(start + BUFFER_SIZE, fileSize);
            if ((last >> scanIdShift) == fileId)
                end = std::min(end, FastPly::ReaderBase::size_type(last & splatIdMask));

            if (start < end)
            {
                if (!handle || handleId != fileId)
                {
                    handle.reset(owner.files[fileId].createHandle(rawBuffer, BUFFER_SIZE * sizeof(Splat)));
                    handleId = fileId;
                }

                boost::shared_ptr<Item> item = pool.pop();
                item->splats.resize(end - start);
                item->first = first;
                item->last = first + (end - start);
                handle->read(start, end, &item->splats[0]);
                outQueue.push(item);

                first += end - start;
            }
            if (end == fileSize)
            {
                first = (fileId + 1) << scanIdShift;
            }
        }
        // Signal completion
        outQueue.push(boost::shared_ptr<Item>());
    }
}

void SimpleFileSet::ReaderThread::request(splat_id first, splat_id last)
{
    MLSGPU_ASSERT(first <= last, std::invalid_argument);
    Request req;
    req.first = first;
    req.last = last;
    inQueue.push(req);
}

void SimpleFileSet::ReaderThread::stop()
{
    Request req;
    req.first = 1;
    req.last = 0;
    inQueue.push(req);
}

void SimpleFileSet::ReaderThread::drain()
{
    boost::shared_ptr<Item> item;
    while (!!(item = pop()))
    {
        push(item);
    }
}

SimpleFileSet::MySplatStream::MySplatStream(
    const SimpleFileSet &owner, splat_id first, splat_id last)
: owner(owner), readerThread(owner), thread(boost::ref(readerThread))
{
    isEmpty = true;
    reset(first, last);
}

SimpleFileSet::MySplatStream::~MySplatStream()
{
    if (!isEmpty)
        readerThread.drain();
    readerThread.stop();
    thread.join();
}

SplatStream &SimpleFileSet::MySplatStream::operator++()
{
    MLSGPU_ASSERT(!isEmpty, std::out_of_range);
    bufferCur++;
    refill();
    return *this;
}

void SimpleFileSet::MySplatStream::reset(splat_id first, splat_id last)
{
    MLSGPU_ASSERT(first <= last, std::invalid_argument);
    last = std::min(last, splat_id(owner.files.size()) << scanIdShift);
    if (first > last)
        first = last;

    if (!isEmpty)
        readerThread.drain();

    if (buffer)
    {
        readerThread.push(buffer);
        buffer.reset();
    }
    readerThread.request(first, last);
    isEmpty = false;
    refill();
}

void SimpleFileSet::MySplatStream::skipNonFiniteInBuffer()
{
    while (buffer && bufferCur < buffer->splats.size() && !buffer->splats[bufferCur].isFinite())
    {
        bufferCur++;
    }
}

void SimpleFileSet::MySplatStream::refill()
{
    skipNonFiniteInBuffer();
    while (!isEmpty && (!buffer || bufferCur == buffer->splats.size()))
    {
        if (buffer)
            readerThread.push(buffer);
        buffer = readerThread.pop();
        bufferCur = 0;
        if (!buffer)
            isEmpty = true;
        else
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
