/**
 * @file
 *
 * Implementations of template members from @ref splat_set.h.
 */

#ifndef SPLAT_SET_IMPL_H
#define SPLAT_SET_IMPL_H

#if HAVE_XMMINTRIN_H && HAVE_EMMINTRIN_H
# define BLOBS_USE_SSE2 1
#else
# define BLOBS_USE_SSE2 0
#endif

#if HAVE_CONFIG_H
# include <config.h>
#endif
#ifdef _OPENMP
# include <omp.h>
#else
# ifndef omp_get_num_threads
#  define omp_get_num_threads() (1)
# endif
# ifndef omp_get_thread_num
#  define omp_get_thread_num() (0)
# endif
#endif
#include <algorithm>
#include <iterator>
#include <utility>
#include <boost/smart_ptr/shared_ptr.hpp>
#include <boost/smart_ptr/make_shared.hpp>
#include <boost/next_prior.hpp>
#include "errors.h"
#include "splat_set.h"
#include "thread_name.h"
#include "timeplot.h"
#include "misc.h"
#if BLOBS_USE_SSE2
# include <xmmintrin.h>
# include <emmintrin.h>
#endif

namespace SplatSet
{

template<typename Iterator>
template<typename RangeIterator>
std::size_t SequenceSet<Iterator>::MySplatStream<RangeIterator>::read(
    Splat *splats, splat_id *splatIds, std::size_t count)
{
    std::size_t oldCount = count;
    while (count > 0 && curRange != lastRange)
    {
        splat_id end = curRange->second;
        splat_id ownerSize = ownerLast - ownerFirst;
        if (ownerSize < end)
            end = ownerSize;

        while (cur < end && count > 0)
        {
            Iterator x = ownerFirst + cur;
            if (x->isFinite())
            {
                *splats++ = *x;
                if (splatIds != NULL)
                    *splatIds++ = cur;
                count--;
            }
            cur++;
        }
        if (cur >= end)
        {
            ++curRange;
            if (curRange != lastRange)
                cur = curRange->first;
        }
    }
    return oldCount - count;
}


template<typename RangeIterator>
void FileSet::FileRangeIterator<RangeIterator>::increment()
{
    MLSGPU_ASSERT(curRange != lastRange, state_error);
    MLSGPU_ASSERT(owner != NULL, state_error);
    const std::size_t fileId = first >> scanIdShift;
    const std::size_t vertexSize = owner->files[fileId].getVertexSize();
    first = std::min(first + maxSize / vertexSize, curRange->second);
    refill();
}

template<typename RangeIterator>
void FileSet::FileRangeIterator<RangeIterator>::refill()
{
    if (curRange != lastRange)
    {
        while (true)
        {
            std::size_t fileId = first >> scanIdShift;
            if (first >= curRange->second || fileId >= owner->files.size())
            {
                ++curRange;
                if (curRange == lastRange)
                {
                    first = 0;
                    return;
                }
                else
                {
                    first = curRange->first;
                }
            }
            else if ((first & splatIdMask) >= owner->files[fileId].size())
            {
                first = (splat_id(fileId) + 1) << scanIdShift; // advance to next file
            }
            else
                break;
        }
    }
}

template<typename RangeIterator>
bool FileSet::FileRangeIterator<RangeIterator>::equal(const FileRangeIterator<RangeIterator> &other) const
{
    if (curRange == lastRange)
        return other.curRange == other.lastRange;
    else
        return curRange == other.curRange && first == other.first;
}

template<typename RangeIterator>
FileSet::FileRange FileSet::FileRangeIterator<RangeIterator>::dereference() const
{
    MLSGPU_ASSERT(curRange != lastRange, state_error);
    MLSGPU_ASSERT(owner != NULL, state_error);
    FileRange ans;

    ans.fileId = first >> scanIdShift;
    ans.start = first & splatIdMask;
    assert(ans.fileId < owner->files.size());
    ans.end = owner->files[ans.fileId].size();
    if ((curRange->second >> scanIdShift) == ans.fileId)
        ans.end = std::min(ans.end, FastPly::ReaderBase::size_type(curRange->second & splatIdMask));
    const std::size_t vertexSize = owner->files[ans.fileId].getVertexSize();
    if ((ans.end - ans.start) * vertexSize > maxSize)
        ans.end = ans.start + maxSize / vertexSize;
    return ans;
}

template<typename RangeIterator>
FileSet::FileRangeIterator<RangeIterator>::FileRangeIterator(
    const FileSet &owner,
    RangeIterator firstRange,
    RangeIterator lastRange,
    FastPly::ReaderBase::size_type maxSize)
: owner(&owner), curRange(firstRange), lastRange(lastRange), first(0), maxSize(maxSize)
{
    MLSGPU_ASSERT(maxSize > 0, std::invalid_argument);
    if (curRange != lastRange)
        first = curRange->first;
    refill();
}

template<typename RangeIterator>
FileSet::FileRangeIterator<RangeIterator>::FileRangeIterator(
    const FileSet &owner,
    RangeIterator lastRange)
: owner(&owner), curRange(lastRange), lastRange(lastRange), first(0)
{
}

template<typename RangeIterator>
FileSet::ReaderThread<RangeIterator>::ReaderThread(const FileSet &owner, RangeIterator firstRange, RangeIterator lastRange)
    : FileSet::ReaderThreadBase(owner), firstRange(firstRange), lastRange(lastRange)
{
}

template<typename RangeIterator>
void FileSet::ReaderThread<RangeIterator>::operator()()
{
    thread_set_name("reader");

    // Maximum number of bytes to load at one time. This must be less than the buffer
    // size, and should be much less for efficiency.
    const std::size_t maxChunk = buffer.size() / 8;
    Statistics::Variable &readTimeStat = Statistics::getStatistic<Statistics::Variable>("files.read.time");
    Statistics::Variable &readRangeStat = Statistics::getStatistic<Statistics::Variable>("files.read.splats");
    Statistics::Variable &readMergedStat = Statistics::getStatistic<Statistics::Variable>("files.read.merged");

    boost::scoped_ptr<FastPly::ReaderBase::Handle> handle;
    std::size_t handleId = 0;
    FileRangeIterator<RangeIterator> first(owner, firstRange, lastRange, maxChunk);
    FileRangeIterator<RangeIterator> last(owner, lastRange);

    Timeplot::Action totalTimer("compute", tworker);
    FileRangeIterator<RangeIterator> cur = first;
    while (cur != last)
    {
        FileRange range = *cur;
        const std::size_t vertexSize = owner.files[range.fileId].getVertexSize();

        if (!handle || range.fileId != handleId)
        {
            if (vertexSize > maxChunk)
            {
                // TODO: associate the filename with it? Might be too late.
                throw std::runtime_error("Far too many bytes per vertex");
            }
            handle.reset(); // close the old handle
            handle.reset(owner.files[range.fileId].createHandle());
            handleId = range.fileId;
        }

        const FastPly::ReaderBase::size_type start = range.start;
        FastPly::ReaderBase::size_type end = range.end;
        /* Request merging */
        FileRangeIterator<RangeIterator> next = cur;
        ++next;
        while (next != last)
        {
            const FileRange nextRange = *next;
            if (nextRange.start < end
                || (nextRange.fileId != range.fileId)
                || (nextRange.start - end) * vertexSize > maxChunk / 2
                || (nextRange.end - start) * vertexSize > maxChunk)
                break;
            end = nextRange.end;
            ++next;
        }

        CircularBuffer::Allocation alloc = buffer.allocate(tworker, vertexSize, end - start);
        char *chunk = (char *) alloc.get();
        {
            Timeplot::Action readTimer("load", tworker, readTimeStat);
            handle->readRaw(start, end, chunk);
        }
        readMergedStat.add(end - start);

        {
            Timeplot::Action pushTimer("push", tworker);
            while (cur != next)
            {
                readRangeStat.add(range.end - range.start);

                Item item;
                item.first = range.start + (splat_id(range.fileId) << scanIdShift);
                item.last = item.first + (range.end - range.start);
                item.ptr = chunk + (range.start - start) * vertexSize;
                ++cur;
                if (cur != next)
                    range = *cur;
                else
                    item.alloc = alloc;

                outQueue.push(item);
            }
        }
    }

    // Signal completion
    outQueue.stop();
}

static inline std::tr1::int32_t extractUnsigned(std::tr1::uint32_t value, int lbit, int hbit)
{
    assert(0 <= lbit && lbit < hbit && hbit <= 32);
    assert(hbit - lbit < 32);
    value >>= lbit;
    value &= (std::tr1::uint32_t(1) << (hbit - lbit)) - 1;
    return value;
}

static inline std::tr1::uint32_t extractSigned(std::tr1::uint32_t value, int lbit, int hbit)
{
    int bits = hbit - lbit;
    std::tr1::int32_t ans = extractUnsigned(value, lbit, hbit);
    if (ans & (std::tr1::uint32_t(1) << (bits - 1)))
        ans -= std::tr1::int32_t(1) << bits;
    return ans;
}

static inline std::tr1::uint32_t insertUnsigned(std::tr1::uint32_t payload, std::tr1::uint32_t value, int lbit, int hbit)
{
    assert(0 <= lbit && lbit < hbit && hbit <= 32);
    assert(hbit - lbit < 32);
    assert(value < std::tr1::uint32_t(1) << (hbit - lbit));
    (void) hbit;
    return payload | (value << lbit);
}

static inline std::tr1::uint32_t insertSigned(std::tr1::uint32_t payload, std::tr1::int32_t value, int lbit, int hbit)
{
    assert(0 <= lbit && lbit < hbit && hbit <= 32);
    assert(hbit - lbit < 32);
    assert(value >= -(std::tr1::int32_t(1) << (hbit - lbit))
           && value < (std::tr1::int32_t(1) << (hbit - lbit)));
    if (value < 0)
        value += std::tr1::uint32_t(1) << (hbit - lbit);
    return payload | (value << lbit);
}

template<typename Base, typename BlobVector>
BlobStream &FastBlobSet<Base, BlobVector>::MyBlobStream::operator++()
{
    MLSGPU_ASSERT(!empty(), std::length_error);
    refill();
    return *this;
}

template<typename Base, typename BlobVector>
void FastBlobSet<Base, BlobVector>::MyBlobStream::refill()
{
    if (nextPtr == owner.blobData.size())
    {
        curBlob.firstSplat = 1;
        curBlob.lastSplat = 0;
    }
    else
    {
        std::tr1::uint32_t data = owner.blobData[nextPtr];
        if (data & UINT32_C(0x80000000))
        {
            // Differential record
            for (unsigned int i = 0; i < 3; i++)
            {
                curBlob.lower[i] = curBlob.upper[i] + extractSigned(data, i * 4, i * 4 + 3);
                curBlob.upper[i] = curBlob.lower[i] + extractUnsigned(data, i * 4 + 3, i * 4 + 4);
            }
            curBlob.firstSplat = curBlob.lastSplat;
            curBlob.lastSplat = curBlob.firstSplat + extractUnsigned(data, 12, 31);
            nextPtr += 1;
        }
        else
        {
            // Full record
            MLSGPU_ASSERT(nextPtr + 10 <= owner.blobData.size(), std::length_error);
            std::tr1::uint64_t firstHi = data;
            std::tr1::uint64_t firstLo = owner.blobData[nextPtr + 1];
            std::tr1::uint64_t lastHi = owner.blobData[nextPtr + 2];
            std::tr1::uint64_t lastLo = owner.blobData[nextPtr + 3];
            curBlob.firstSplat = (firstHi << 32) | firstLo;
            curBlob.lastSplat = (lastHi << 32) | lastLo;
            for (unsigned int i = 0; i < 3; i++)
            {
                curBlob.lower[i] = static_cast<std::tr1::int32_t>(owner.blobData[nextPtr + 4 + 2 * i]);
                curBlob.upper[i] = static_cast<std::tr1::int32_t>(owner.blobData[nextPtr + 5 + 2 * i]);
            }
            nextPtr += 10;
        }
    }
}

template<typename Base, typename BlobVector>
BlobInfo FastBlobSet<Base, BlobVector>::MyBlobStream::operator*() const
{
    BlobInfo ans;
    MLSGPU_ASSERT(!empty(), std::out_of_range);

    ans.firstSplat = curBlob.firstSplat;
    ans.lastSplat = curBlob.lastSplat;
    for (unsigned int i = 0; i < 3; i++)
        ans.lower[i] = bucketDivider(curBlob.lower[i] - offset[i]);
    for (unsigned int i = 0; i < 3; i++)
        ans.upper[i] = bucketDivider(curBlob.upper[i] - offset[i]);
    return ans;
}

template<typename Base, typename BlobVector>
FastBlobSet<Base, BlobVector>::MyBlobStream::MyBlobStream(
    const FastBlobSet<Base, BlobVector> &owner, const Grid &grid,
    Grid::size_type bucketSize)
:
    owner(owner),
    bucketDivider(bucketSize / owner.internalBucketSize)
{
    MLSGPU_ASSERT(bucketSize > 0 && owner.internalBucketSize > 0
                  && bucketSize % owner.internalBucketSize == 0, std::invalid_argument);
    for (unsigned int i = 0; i < 3; i++)
        offset[i] = grid.getExtent(i).first / Grid::difference_type(owner.internalBucketSize);
    nextPtr = 0;
    refill();
}

template<typename Base, typename BlobVector>
FastBlobSet<Base, BlobVector>::FastBlobSet()
: Base(), internalBucketSize(0), nSplats(0)
{
}

template<typename Base, typename BlobVector>
FastBlobSet<Base, BlobVector>::~FastBlobSet()
{
    // STXXL 1.3.1 will write the vector cache back to disk unless this is
    // added.
    blobData.clear();
}

template<typename Base, typename BlobVector>
template<typename T>
FastBlobSet<Base, BlobVector>::FastBlobSet(const T &blobVectorArg)
: Base(), internalBucketSize(0), blobData(blobVectorArg), nSplats(0)
{
}

template<typename Base, typename BlobVector>
BlobStream *FastBlobSet<Base, BlobVector>::makeBlobStream(
    const Grid &grid, Grid::size_type bucketSize) const
{
    if (fastPath(grid, bucketSize))
        return new MyBlobStream(*this, grid, bucketSize);
    else
        return Base::makeBlobStream(grid, bucketSize);
}

namespace detail
{

struct Bbox
{
    boost::array<float, 3> bboxMin, bboxMax;

    Bbox()
    {
        std::fill(bboxMin.begin(), bboxMin.end(), std::numeric_limits<float>::infinity());
        std::fill(bboxMax.begin(), bboxMax.end(), -std::numeric_limits<float>::infinity());
    }

    Bbox &operator+=(const Bbox &b)
    {
        for (int j = 0; j < 3; j++)
        {
            bboxMin[j] = std::min(bboxMin[j], b.bboxMin[j]);
            bboxMax[j] = std::max(bboxMax[j], b.bboxMax[j]);
        }
        return *this;
    }

    Bbox &operator+=(const Splat &splat)
    {
        for (int j = 0; j < 3; j++)
        {
            bboxMin[j] = std::min(bboxMin[j], splat.position[j] - splat.radius);
            bboxMax[j] = std::max(bboxMax[j], splat.position[j] + splat.radius);
        }
        return *this;
    }
};

/**
 * Computes the range of buckets that will be occupied by a splat's bounding
 * box. See @ref BlobInfo for the definition of buckets.
 *
 * The coordinates are given in units of buckets, with (0,0,0) being the bucket
 * overlapping cell (0,0,0).
 *
 * @param      splat         Input splat
 * @param      grid          Grid for spacing and alignment
 * @param      bucketSize    Size of buckets in cells
 * @param[out] lower         Lower bound coordinates (inclusive)
 * @param[out] upper         Upper bound coordinates (inclusive)
 *
 * @pre
 * - <code>splat.isFinite()</code>
 * - @a bucketSize &gt; 0
 */
void splatToBuckets(const Splat &splat,
                    const Grid &grid, Grid::size_type bucketSize,
                    boost::array<Grid::difference_type, 3> &lower,
                    boost::array<Grid::difference_type, 3> &upper);

/**
 * Computes the range of buckets that will be occupied by a splat's bounding
 * box. See @ref BlobInfo for the definition of buckets. This is a version that
 * is optimized and specialized for a grid based at the origin.
 *
 * The coordinates are given in units of buckets, with (0,0,0) being the bucket
 * overlapping cell (0,0,0).
 */
class SplatToBuckets
{
private:
#if BLOBS_USE_SSE2
    __m128i negAdd;
    __m128i posAdd;
    __m128 invSpacing;
    std::tr1::int64_t inverse;
    int shift;

    inline void divide(__m128i in, boost::array<Grid::difference_type, 3> &out) const;

#else
    float invSpacing;
    DownDivider divider;
#endif

public:
    typedef void result_type;

    /**
     * Perform the conversion.
     * @param      splat         Input splat
     * @param[out] lower         Lower bound coordinates (inclusive)
     * @param[out] upper         Upper bound coordinates (inclusive)
     *
     * @pre splat.isFinite().
     */
    void operator()(
        const Splat &splat,
        boost::array<Grid::difference_type, 3> &lower,
        boost::array<Grid::difference_type, 3> &upper) const;

    /**
     * Constructor.
     * @param      spacing       Grid spacing
     * @param      bucketSize    Bucket size in cells
     * @pre @a bucketSize &gt; 0
     */
    SplatToBuckets(float spacing, Grid::size_type bucketSize);
};

} // namespace detail

template<typename Base, typename BlobVector>
void FastBlobSet<Base, BlobVector>::addBlob(Statistics::Container::vector<BlobData> &blobData, const BlobInfo &prevBlob, const BlobInfo &curBlob)
{
    bool differential;

    if (!blobData.empty()
        && prevBlob.lastSplat == curBlob.firstSplat
        && curBlob.lastSplat - curBlob.firstSplat < (1U << 19))
    {
        differential = true;
        for (unsigned int i = 0; i < 3 && differential; i++)
            if (curBlob.upper[i] - curBlob.lower[i] > 1
                || curBlob.lower[i] < prevBlob.upper[i] - 4
                || curBlob.lower[i] > prevBlob.upper[i] + 3)
                differential = false;
    }
    else
        differential = false;

    if (differential)
    {
        std::tr1::uint32_t payload = 0;
        payload |= UINT32_C(0x80000000); // signals a differential record
        for (unsigned int i = 0; i < 3; i++)
        {
            std::tr1::int32_t d = curBlob.lower[i] - prevBlob.upper[i];
            payload = insertSigned(payload, d, i * 4, i * 4 + 3);
            std::tr1::uint32_t s = curBlob.upper[i] - curBlob.lower[i];
            payload = insertUnsigned(payload, s, i * 4 + 3, i * 4 + 4);
        }
        payload = insertUnsigned(payload, curBlob.lastSplat - curBlob.firstSplat, 12, 31);
        blobData.push_back(payload);
    }
    else
    {
        blobData.push_back(curBlob.firstSplat >> 32);
        blobData.push_back(curBlob.firstSplat & UINT32_C(0xFFFFFFFF));
        blobData.push_back(curBlob.lastSplat >> 32);
        blobData.push_back(curBlob.lastSplat & UINT32_C(0xFFFFFFFF));
        for (unsigned int i = 0; i < 3; i++)
        {
            blobData.push_back(static_cast<std::tr1::uint32_t>(curBlob.lower[i]));
            blobData.push_back(static_cast<std::tr1::uint32_t>(curBlob.upper[i]));
        }
    }
}

template<typename Base, typename BlobVector>
void FastBlobSet<Base, BlobVector>::computeBlobs(
    const float spacing, const Grid::size_type bucketSize, std::ostream *progressStream, bool warnNonFinite)
{
    const float ref[3] = {0.0f, 0.0f, 0.0f};

    MLSGPU_ASSERT(bucketSize > 0, std::invalid_argument);
    Statistics::Registry &registry = Statistics::Registry::getInstance();

    blobData.clear();
    internalBucketSize = bucketSize;

    // Reference point will be 0,0,0. Extents are set after reading all the splats.
    boundingGrid.setSpacing(spacing);
    boundingGrid.setReference(ref);
    for (unsigned int i = 0; i < 3; i++)
        boundingGrid.setExtent(i, 0, 1);

    boost::scoped_ptr<ProgressDisplay> progress;
    if (progressStream != NULL)
    {
        *progressStream << "Computing bounding box\n";
        progress.reset(new ProgressDisplay(Base::maxSplats(), *progressStream));
    }

    detail::Bbox bbox;

    boost::scoped_ptr<SplatStream> splats(Base::makeSplatStream());
    nSplats = 0;

    static const std::size_t BUFFER_SIZE = 64 * 1024;
    Statistics::Container::vector<Splat> buffer("mem.computeBlobs.buffer", BUFFER_SIZE);
    Statistics::Container::vector<splat_id> bufferIds("mem.computeBlobs.buffer", BUFFER_SIZE);

    std::tr1::uint64_t nBlobs = 0;
    const detail::SplatToBuckets toBuckets(spacing, bucketSize);

    while (true)
    {
        const std::size_t nBuffer = splats->read(&buffer[0], &bufferIds[0], BUFFER_SIZE);
        if (nBuffer == 0)
            break;

#ifdef _OPENMP
#pragma omp parallel shared(buffer, bufferIds, bbox, nBlobs) default(none)
#endif
        {
            const int nThreads = omp_get_num_threads();
            /* Divide the splats into subblocks, based on an estimate of how many threads
             * will be involved. We have to manually strip-mine the loop to guarantee that
             * the distribution is in contiguous chunks.
             */
#ifdef _OPENMP
#pragma omp for schedule(static,1) ordered
#endif
            for (int tid = 0; tid < nThreads; tid++)
            {
                std::size_t first = tid * nBuffer / nThreads;
                std::size_t last = (tid + 1) * nBuffer / nThreads;
                detail::Bbox threadBbox;
                Statistics::Container::vector<BlobData> threadBlobData("mem.computeBlobs.threadBlobData");
                BlobInfo curBlob, prevBlob;
                bool haveCurBlob = false;
                std::tr1::uint64_t threadBlobs = 0;

                // Compute the blobs for a single subrange. The first blob will always
                // be a non-differential encoding, so the encoding depends on the number
                // of subchunks chosen.
                for (std::size_t i = first; i < last; i++)
                {
                    const Splat &splat = buffer[i];
                    BlobInfo blob;
                    toBuckets(splat, blob.lower, blob.upper);
                    blob.firstSplat = bufferIds[i];
                    blob.lastSplat = blob.firstSplat + 1;
                    threadBbox += splat;

                    if (!haveCurBlob)
                    {
                        curBlob = blob;
                        haveCurBlob = true;
                    }
                    else if (curBlob.lower == blob.lower
                             && curBlob.upper == blob.upper
                             && curBlob.lastSplat == blob.firstSplat)
                        curBlob.lastSplat++;
                    else
                    {
                        addBlob(threadBlobData, prevBlob, curBlob);
                        threadBlobs++;
                        prevBlob = curBlob;
                        curBlob = blob;
                    }
                }
                if (haveCurBlob)
                {
                    addBlob(threadBlobData, prevBlob, curBlob);
                    threadBlobs++;
                }

#ifdef _OPENMP
#pragma omp ordered
#endif
                {
                    // Write the blobs for this subrange out to the global blob list.
                    bbox += threadBbox;
                    nBlobs += threadBlobs;
                    std::copy(threadBlobData.begin(), threadBlobData.end(), std::back_inserter(blobData));
                }
            }
        }

        nSplats += nBuffer;
        if (progress != NULL)
            *progress += nBuffer;
    }

    assert(nSplats <= Base::maxSplats());
    splat_id nonFinite = Base::maxSplats() - nSplats;
    if (nonFinite > 0)
    {
        if (progress != NULL)
            *progress += nonFinite;
        if (warnNonFinite)
            Log::log[Log::warn] << "Input contains " << nonFinite << " splat(s) with non-finite values\n";
    }
    registry.getStatistic<Statistics::Variable>("blobset.nonfinite").add(nonFinite);

    if (bbox.bboxMin[0] > bbox.bboxMax[0])
        throw std::runtime_error("Must be at least one splat");

    for (unsigned int i = 0; i < 3; i++)
    {
        float l = bbox.bboxMin[i] / spacing;
        float h = bbox.bboxMax[i] / spacing;
        Grid::difference_type lo = Grid::RoundDown::convert(l);
        Grid::difference_type hi = Grid::RoundUp::convert(h);
        /* The lower extent must be a multiple of the bucket size, to
         * make the blob data align properly.
         */
        lo = divDown(lo, bucketSize) * bucketSize;
        assert(lo % Grid::difference_type(bucketSize) == 0);

        boundingGrid.setExtent(i, lo, hi);
    }
    registry.getStatistic<Statistics::Variable>("blobset.blobs").add(nBlobs);
    registry.getStatistic<Statistics::Variable>("blobset.blobs.size").add(blobData.size() * sizeof(BlobData));

    const char * const names[3] =
    {
        "blobset.bboxX",
        "blobset.bboxY",
        "blobset.bboxZ"
    };

    for (int i = 0; i < 3; i++)
    {
        registry.getStatistic<Statistics::Variable>(names[i]).add(bbox.bboxMax[i] - bbox.bboxMin[i]);
    }
}

template<typename Base, typename BlobVector>
bool FastBlobSet<Base, BlobVector>::fastPath(const Grid &grid, Grid::size_type bucketSize) const
{
    MLSGPU_ASSERT(internalBucketSize > 0, state_error);
    MLSGPU_ASSERT(bucketSize > 0, std::invalid_argument);
    if (bucketSize % internalBucketSize != 0)
        return false;
    if (boundingGrid.getSpacing() != grid.getSpacing())
        return false;
    for (unsigned int i = 0; i < 3; i++)
    {
        if (grid.getReference()[i] != 0.0f
            || grid.getExtent(i).first % Grid::difference_type(internalBucketSize) != 0)
            return false;
    }
    return true;
}


template<typename InputIterator1, typename InputIterator2, typename OutputIterator>
OutputIterator merge(
    InputIterator1 first1, InputIterator1 last1,
    InputIterator2 first2, InputIterator2 last2,
    OutputIterator out)
{
    InputIterator1 p1 = first1;
    InputIterator2 p2 = first2;
    while (p1 != last1 && p2 != last2)
    {
        splat_id first = std::min(p1->first, p2->first);
        splat_id last = first;
        // Extend last for as far as we have contiguous ranges
        while (true)
        {
            if (p1 != last1 && p1->first <= last)
            {
                last = std::max(last, p1->second);
                ++p1;
            }
            else if (p2 != last2 && p2->first <= last)
            {
                last = std::max(last, p2->second);
                ++p2;
            }
            else
                break;
        }
        *out++ = std::make_pair(first, last);
    }
    // Copy tail pieces
    out = std::copy(p1, last1, out);
    out = std::copy(p2, last2, out);
    return out;
}


template<typename Super>
BlobStream *Subset<Super>::makeBlobStream(const Grid &grid, Grid::size_type bucketSize) const
{
    MLSGPU_ASSERT(bucketSize > 0, std::invalid_argument);
    return new SimpleBlobStream(makeSplatStream(), grid, bucketSize);
}

} // namespace SplatSet

#endif /* !SPLAT_SET_IMPL_H */
