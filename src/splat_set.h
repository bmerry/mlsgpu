/**
 * @file
 *
 * Containers for splats supporting various forms of iteration.
 */

#ifndef SPLAT_SET_H
#define SPLAT_SET_H

#if HAVE_CONFIG_H
# include <config.h>
#endif

#include "tr1_cstdint.h"
#include <vector>
#include <utility>
#include <algorithm>
#include <stdexcept>
#include <cassert>
#include <cstddef>
#include <iosfwd>
#include <memory>
#include <boost/array.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/smart_ptr/scoped_ptr.hpp>
#include <boost/smart_ptr/shared_ptr.hpp>
#include <boost/type_traits/integral_constant.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/locks.hpp>
#include "grid.h"
#include "misc.h"
#include "splat.h"
#include "errors.h"
#include "fast_ply.h"
#include "statistics.h"
#include "logging.h"
#include "work_queue.h"
#include "progress.h"
#include "allocator.h"
#include "circular_buffer.h"
#include "tr1_cstdint.h"

template<typename BaseType>
class TestFastBlobSet;

/**
 * Data structures for iteration over sets of splats.
 */
namespace SplatSet
{

typedef std::tr1::uint64_t splat_id;

/**
 * Metadata about a sequence of splats. The range of splat IDs must all be
 * valid splat IDs, and hence the number of splats in the blob is exactly
 * @a lastSplat - @a firstSplat.
 */
struct BlobInfo
{
    splat_id firstSplat; ///< First splat ID in the blob
    splat_id lastSplat;  ///< One past the last splat ID in the blob

    /**
     * Lower bound for bucket range hit by the bounding boxes of the splats.
     * The coordinate system is determined by a grid and a @a bucketSize. A
     * bucket with coordinate @a x covers the real interval [@a x * @a
     * bucketSize, (@a x + 1) * @a bucketSize) in the grid space, and
     * similarly for the other dimensions.
     */
    boost::array<Grid::difference_type, 3> lower;

    /**
     * Upper bound (inclusive) of bucket range.
     * @see @ref lower.
     */
    boost::array<Grid::difference_type, 3> upper;

    bool operator==(const BlobInfo &b) const
    {
        return firstSplat == b.firstSplat
            && lastSplat == b.lastSplat
            && lower == b.lower
            && upper == b.upper;
    }
};

/**
 * Polymorphic interface for iteration over a sequence of splats. This is based
 * on the STXXL stream interface. The splats returns must all be finite.
 *
 * The general pattern of implementations is to maintain some form of pointer
 * to the next splat to return. Both during construction and after advancing
 * the pointer, it will pre-emptively skip over non-finite splats until a
 * finite one is found or the end is reached.
 *
 * In some cases the collection is really a collection-of-collections under
 * the hood. Again, as soon as the pointer is advanced to the end of a
 * sub-collection it is preemptively advanced until the pointer becomes valid
 * again or the end is reached. This is done in a method calls @c refill.
 *
 * @see @ref SetConcept
 */
class SplatStream : public boost::noncopyable
{
public:
    typedef Splat value_type;
    typedef const Splat &reference;

    virtual ~SplatStream() {}

    /**
     * Advance to the next splat in the stream.
     * @pre <code>!empty()</code>
     */
    virtual SplatStream &operator++() = 0;

    /**
     * Return the currently pointed-to splat.
     * @pre <code>!empty()</code>
     */
    virtual Splat operator*() const = 0;

    /**
     * Determines whether there are any more splats to advance over.
     */
    virtual bool empty() const = 0;

    /**
     * Returns an ID for the current splat (the one that would be returned by
     * <code>operator *</code>).
     * @pre <code>!empty()</code>
     */
    virtual splat_id currentId() const = 0;
};

/**
 * Polymorphic interface for iteration over a sequence of splats, with groups
 * of splats potentially bucketed together. This is based on the STXXL stream
 * interface. The implementation is required to filter out non-finite splats,
 * and the counts returned in the @ref BlobInfo must accurately reflect this.
 * Any blobs that end up empty must also be filtered out.
 */
class BlobStream : public boost::noncopyable
{
public:
    typedef BlobInfo value_type;
    typedef BlobInfo reference;

    virtual ~BlobStream() {}

    /**
     * Advance to the next blob in the stream.
     * @pre <code>!empty()</code>
     */
    virtual BlobStream &operator++() = 0;

    /**
     * Retrieve information about the current blob.
     * @pre <code>!empty()</code>
     */
    virtual BlobInfo operator*() const = 0;

    /**
     * Determine whether there are any more blobs in the stream.
     */
    virtual bool empty() const = 0;
};

#ifdef DOXYGEN_FAKE_CODE
/**
 * Concept documentation for a splat set.  A <em>splat set</em> is an ordered
 * collection of splats, each of which has a unique ID. The IDs do not need to
 * be contiguous, but for some purposes it is best if there are long runs of
 * contiguous IDs.
 *
 * A splat set can be iterated using @ref makeSplatStream, but for bucketing it
 * may be more efficient to use @ref makeBlobStream to iterate over
 * <em>blobs</em>. A blob is a contiguous sequence of splats which occupy the
 * same set of buckets. Note that the size and alignment of buckets is a
 * run-time choice: thus, a single splat set will iterate a different set of
 * blobs depending on which grid and bucket size are provided.
 *
 * Splats for which @ref Splat::isFinite is false are never enumerated as part
 * of @ref makeSplatStream, nor are they counted in blobs. It is thus up to
 * the models of this concept to filter them out.
 */
class SetConcept
{
public:
    /**
     * Return an upper bound on the number of splats that would be enumerated
     * by @ref makeSplatStream. This must be usable for memory allocation so
     * it should not be a gross overestimate. In typical use, this will be the
     * number of splats in a backing store which might include non-finite
     * splats.
     *
     * Some models of this concept provide a <code>numSplats</code> member
     * which give an exact count. Where this member exists, it is guaranteed
     * that @a maxSplats will provide the same count.
     */
    splat_id maxSplats() const;

    /**
     * Returns a stream that iterates over all (finite) splats in the set.
     */
    SplatStream *makeSplatStream() const;

    /**
     * Returns a stream that iterates over all blobs in the set.
     */
    BlobStream *makeBlobStream(const Grid &grid, Grid::size_type bucketSize) const;
};

/**
 * A set that can be wrapped in @ref Subset. It contains extra methods for
 * random access that are used by @ref Subset to iterate over its subset.
 */
template<typename RangeIterator>
class SubsettableConcept : public SetConcept
{
public:
    /**
     * Create a splat stream that can be used to pull a collection of ranges.
     *
     * @param firstRange,lastRange   A sequence of instances of <code>std::pair<splat_id, splat_id></code>, each a range of splat IDs
     *
     * @warning The iterators are accessed while the stream is walked, so the backing storage
     * for them must remain intact and unaltered until the stream is destroyed.
     */
    SplatStream *makeSplatStream(RangeIterator firstRange, RangeIterator lastRange) const;
};

#endif // DOXYGEN_FAKE_CODE

/**
 * Internal implementation details of @ref SplatSet.
 */
namespace detail
{

/// Range of [0, max splat id), so back a reader over the entire set
extern const std::pair<splat_id, splat_id> rangeAll;

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

} // namespace detail

/**
 * Implementation of @ref BlobStream that just has one blob for each splat. It
 * is created by passing the underlying splat stream to the constructor.
 */
class SimpleBlobStream : public BlobStream
{
public:
    virtual BlobInfo operator*() const;

    virtual BlobStream &operator++()
    {
        ++*splatStream;
        return *this;
    }

    virtual bool empty() const
    {
        return splatStream->empty();
    }

    SimpleBlobStream(SplatStream *splatStream, const Grid &grid, Grid::size_type bucketSize)
        : splatStream(splatStream), grid(grid), bucketSize(bucketSize)
    {
        MLSGPU_ASSERT(bucketSize > 0, std::invalid_argument);
    }

private:
    boost::scoped_ptr<SplatStream> splatStream;
    const Grid grid;
    Grid::size_type bucketSize;
};

/**
 * Splat set interface for a simple vector of splats. The splat IDs are
 * simply the positions in the vector. It is legal to store non-finite splats;
 * they will be skipped over by the stream. Blobs just contain one splat each.
 *
 * All the public methods of @ref Statistics::Container::vector&lt;Splat&gt;
 * are available, but modifying the data while a stream exists yields undefined
 * behavior.
 */
class VectorSet : public Statistics::Container::vector<Splat>
{
public:
    splat_id maxSplats() const { return size(); }

    SplatStream *makeSplatStream() const
    {
        return makeSplatStream(&detail::rangeAll, &detail::rangeAll + 1);
    }

    template<typename RangeIterator>
    SplatStream *makeSplatStream(RangeIterator firstRange, RangeIterator lastRange) const
    {
        return new MySplatStream<RangeIterator>(*this, firstRange, lastRange);
    }

    BlobStream *makeBlobStream(const Grid &grid, Grid::size_type bucketSize) const
    {
        return new SimpleBlobStream(makeSplatStream(), grid, bucketSize);
    }

    VectorSet() : Statistics::Container::vector<Splat>("mem.VectorSet") {}

private:
    /// Splat stream implementation
    template<typename RangeIterator>
    class MySplatStream : public SplatStream
    {
    public:
        virtual Splat operator*() const
        {
            MLSGPU_ASSERT(!empty(), std::out_of_range);
            return owner[cur];
        }

        virtual SplatStream &operator++()
        {
            MLSGPU_ASSERT(!empty(), std::out_of_range);
            cur++;
            refill();
            return *this;
        }

        virtual bool empty() const
        {
            return curRange == lastRange;
        }

        virtual splat_id currentId() const
        {
            return cur;
        }

        MySplatStream(const VectorSet &owner, RangeIterator firstRange, RangeIterator lastRange)
            : owner(owner), curRange(firstRange), lastRange(lastRange)
        {
            if (curRange != lastRange)
                cur = curRange->first;
            refill();
        }

    private:
        const VectorSet &owner;
        RangeIterator curRange, lastRange;
        splat_id cur; ///< Index of current splat in owner (undefined if stream is empty)

        ///< Advances until reading a finite splat or the end of the stream
        void refill();
    };
};

/**
 * Splat-set interface for a collection of on-disk PLY files.
 *
 * The splat IDs use the upper bits to store the file ID and the remaining
 * bits to store the splat index within the file.
 */
class FileSet
{
private:
    struct FileRange
    {
        std::size_t fileId;  ///< Index into list of files
        FastPly::ReaderBase::size_type start, end;  ///< Indices within @ref fileId
    };

    /**
     * Iterator class that is used to wrap an iterator over ranges and presents
     * range that are always within a single file, clamped to the range of the
     * file size, and non-empty.
     */
    template<typename RangeIterator>
    class FileRangeIterator : public boost::iterator_facade<
        FileRangeIterator<RangeIterator>,
        FileRange,
        boost::forward_traversal_tag,
        FileRange>
    {
    private:
        const FileSet *owner;
        RangeIterator curRange;
        RangeIterator lastRange;
        splat_id first; ///< First splat in the returned range (0 if singular)

        friend class boost::iterator_core_access;

        void refill(); ///< Advance until a non-empty output range is found

        void increment();

        bool equal(const FileRangeIterator &other) const;

        FileRange dereference() const;

    public:
        FileRangeIterator() : owner(NULL), curRange(), lastRange(), first(0) {}

        /// Begin iterator
        FileRangeIterator(const FileSet &owner, RangeIterator firstRange, RangeIterator lastRange);

        /// End iterator
        explicit FileRangeIterator(const FileSet &owner, RangeIterator lastRange);
    };

public:
    enum
    {
        /**
         * Default size of internal buffer for reading file data.
         *
         * This is the total buffer size, but only a fixed fraction of it
         * is used in any one read, so that reads can be pipelined.
         *
         * @see @ref setBufferSize
         */
        DEFAULT_BUFFER_SIZE = 256 * 1024 * 1024
    };

    /// Number of bits used to store the within-file splat ID
    static const unsigned int scanIdShift;
    /// Mask of the bits used to store the within-file splat ID
    static const splat_id splatIdMask;

    /**
     * Append a new file to the set. The set takes over ownership of the file.
     * This must not be called while a stream is in progress.
     */
    void addFile(FastPly::ReaderBase *file);

    SplatStream *makeSplatStream() const
    {
        return makeSplatStream(&detail::rangeAll, &detail::rangeAll + 1);
    }

    BlobStream *makeBlobStream(const Grid &grid, Grid::size_type bucketSize) const
    {
        return new SimpleBlobStream(makeSplatStream(), grid, bucketSize);
    }

    template<typename RangeIterator>
    SplatStream *makeSplatStream(RangeIterator firstRange, RangeIterator lastRange) const
    {
        std::auto_ptr<ReaderThread<RangeIterator> > reader(
            new ReaderThread<RangeIterator>(*this, firstRange, lastRange));
        SplatStream *ans = new MySplatStream(*this, reader.get());
        reader.release();
        return ans;
    }

    splat_id maxSplats() const { return nSplats; }

    /**
     * Set the buffer size that is used by the reader thread. It is not safe
     * to call this function at the same time as another thread creates a
     * stream, but it can be called while streams exist and they will each
     * have their own buffer size.
     *
     * @warning This must be at least twice as big as any of the splats in any
     * of the files, and ideally a lot bigger or performance will suffer.
     */
    void setBufferSize(std::size_t bufferSize) { this->bufferSize = bufferSize; }

    FileSet() : nSplats(0), bufferSize(DEFAULT_BUFFER_SIZE) {}

private:
    /**
     * Base class for @ref ReaderThread that is agnostic to the range iterator
     * type. It provides the management of the queues but not the actual thread
     * function.
     */
    class ReaderThreadBase : public boost::noncopyable
    {
    public:
        /**
         * Describes a contiguous range of splats. It can also be a sentinel
         * value (marked with @ref ptr of @c NULL), which marks the end out
         * the splat stream.
         */
        struct Item
        {
            splat_id first;      ///< ID of first splat in the range
            splat_id last;       ///< One more than the ID of the last splat in the range

            /**
             * A pointer to the raw splat data. This can be decoded with @ref decode,
             * extracting the file ID from @ref first. It is guaranteed that all splats
             * in the range have the same layout.
             */
            char *ptr;

            /**
             * Total bytes allocated in the buffer (including any padding). This is
             * not intended to be used externally, just by @ref free.
             */
            std::size_t bytes;

            Item() : first(0), last(0), ptr(NULL), bytes(0)
            {
            }

            std::size_t numSplats() const { return last - first; }
        };
    protected:
        const FileSet &owner;   ///< Owning splat stream
        /**
         * Queue of splat ranges as they're read. This will produce a stream of
         * real ranges (non-NULL pointer), followed by exactly one default-constructed
         * item.
         */
        WorkQueue<Item> outQueue;

        CircularBuffer buffer;

    public:
        explicit ReaderThreadBase(const FileSet &owner);

        /// Virtual destructor to allow dynamic storage management
        virtual ~ReaderThreadBase() {}

        /// Thread function
        virtual void operator()() = 0;

        /**
         * Remove all remaining items from the out queue, immediately
         * restoring them to the pool. This is called by the stream thread.
         *
         * @pre The end-of-stream marker has not yet been seen (otherwise this will deadlock)
         */
        void drain();

        /**
         * Retrieve the next range of splats from the reader.
         * This is called by the stream thread, and is thread-safe.
         */
        Item pop() { return outQueue.pop(); }

        /**
         * Return memory retrieved by @ref pop. The stream thread must
         * eventually call this for every non-sentinel value returned by
         * @ref pop, and it must do so in the same order.
         */
        void free(const Item &item);
    };

    /**
     * Thread class that does reads and provides the data for the stream.
     *
     * @todo Optimize so that multiple ranges can share a buffer slot.
     * @todo Optimize to make fewer reads overall
     */
    template<typename RangeIterator>
    class ReaderThread : public ReaderThreadBase
    {
    private:
        RangeIterator firstRange, lastRange;

    public:
        ReaderThread(const FileSet &owner, RangeIterator firstRange, RangeIterator lastRange);

        virtual void operator()();
    };

    /**
     * Splat stream implementation.
     */
    class MySplatStream : public SplatStream
    {
    public:
        virtual Splat operator*() const
        {
            MLSGPU_ASSERT(!empty(), std::out_of_range);
            return nextSplat;
        }

        virtual SplatStream &operator++();

        virtual bool empty() const
        {
            return isEmpty;
        }

        virtual splat_id currentId() const
        {
            return buffer.first + bufferCur;
        }

        MySplatStream(const FileSet &owner, ReaderThreadBase *reader);
        virtual ~MySplatStream();

    private:
        const FileSet &owner;           ///< Owning set
        std::size_t bufferCur;          ///< First position in @ref buffer with data (points at @ref nextSplat)
        Splat nextSplat;                ///< The splat to return from #operator*
        ReaderThreadBase::Item buffer;  ///< Current buffer (possibly NULL)
        bool isEmpty;                   ///< Set to true when hitting the end
        boost::scoped_ptr<ReaderThreadBase> readerThread;
        boost::thread thread;

        /**
         * Advance the stream until empty or a finite splat is reached, and
         * set @ref nextSplat. This will refill the buffer if necessary.
         */
        void refill();
    };

    /// Backing store of files
    boost::ptr_vector<FastPly::ReaderBase> files;

    /// Number of splats stored in the files (including non-finites)
    splat_id nSplats;

    /// Buffer sized used by streams
    std::size_t bufferSize;
};

/**
 * Internal data stored in @ref FastBlobSet. This class is not accessed
 * directly by the user, but is in the namespace so that the user can generate
 * the type for the second template parameter to @ref FastBlobSet.
 */
typedef std::tr1::uint32_t BlobData;

/**
 * Subsettable splat set with accelerated blob interface. This class takes a
 * model of the blobbed interface and extends it by precomputing information
 * about splats for a specific bucket size so that iteration using that bucket
 * size (or any multiple thereof) is potentially faster.
 *
 * To use this class, it is required to call @ref computeBlobs to generate the
 * blob information before calling any of the other functions. To hit the fast
 * path, it is necessary to use a grid whose origin is at the world origin and
 * whose spacing is a multiple of the spacing given to @ref computeBlobs.
 *
 * @ref computeBlobs will also generate a bounding box for the set, which can
 * be retrieved with @ref getBoundingGrid. Since both operations are done in a
 * single pass, this is usually more efficient than computing the bounding
 * grid separately.
 *
 * The blobs are stored in a variable-length encoding, as either 1 or 10 32-bit
 * words. The "full" encoding (10 words) is non-differential and as follows:
 *  -# firstSplat (high)
 *  -# firstSplat (low)
 *  -# lastSplat (high)
 *  -# lastSplat (low)
 *  -# lower[0]
 *  -# upper[0]
 *  -# lower[1]
 *  -# upper[1]
 *  -# lower[2]
 *  -# upper[2]
 *
 * The differential encoding is bit-packed into a 32-bit word as follows (from
 * least to most significant bit):
 *  - [0:3] a[0]
 *  - [3:4] b[0]
 *  - [4:7] a[1]
 *  - [7:8] b[1]
 *  - [8:11] a[2]
 *  - [11:12] b[2]
 *  - [12:31] c
 *  - [31:32] 1
 *
 * The high bit being set is what marks it as differential - note that this means
 * that only half the splat ID range can be used in a FastBlobSet. The @a a values
 * are signed while the other values are unsigned.
 *
 * To complete the decoding, let @a p be the previous decoded blob. Then
 *  - firstSplat = p.lastSplat
 *  - lastSplat = firstSplat + c
 *  - lower[i] = p.upper[i] + a[i]
 *  - upper[i] = lower[i] + b[i]
 *
 * Thus, the differential encoding can only be used when the blob
 *  - is not the first in the file;
 *  - contains at most 2<sup>19</sup> splats;
 *  - follows directly after the previous one in splat ID order;
 *  - covers at most two buckets in each axis;
 *  - is sufficiently close to the previous one.
 *
 * @param Base A model of @ref SubsettableConcept.
 */
template<typename Base, typename BlobVector>
class FastBlobSet : public Base
#ifdef DOXYGEN_FAKE_CODE
, public SubsettableConcept
#endif
{
    template<typename BaseType> friend class ::TestFastBlobSet;
public:
    /**
     * Class returned by makeBlobStream only in the fast path.
     */
    class MyBlobStream : public SplatSet::BlobStream
    {
    public:
        virtual BlobInfo operator*() const;

        virtual BlobStream &operator++();

        virtual bool empty() const
        {
            return curBlob.firstSplat > curBlob.lastSplat;
        }

        MyBlobStream(const FastBlobSet<Base, BlobVector> &owner, const Grid &grid,
                     Grid::size_type bucketSize);

    private:
        const FastBlobSet<Base, BlobVector> &owner;
        /// The stream blob size over the blob size used to construct the blob data
        Grid::size_type bucketRatio;
        /**
         * Offset between the stream grid and the grid used to construct the
         * blob data, in units of @a owner.internalBucketSize.
         */
        Grid::difference_type offset[3];
        /// Index into owner's blobData for the blob after curInfo is extracted
        typename BlobVector::size_type nextPtr;
        /**
         * A blob to return from operator*, but prior to adjustment for @ref
         * offset and @ref bucketRatio. It is also the base for differential
         * encoding of the next blob.
         *
         * In the special case firstSplat > lastSplat, the stream is empty.
         */
        BlobInfo curBlob;

        void refill(); ///< Load curBlob from the stream
    };

    BlobStream *makeBlobStream(const Grid &grid, Grid::size_type bucketSize) const;

    FastBlobSet() : Base(), internalBucketSize(0), nSplats(0) {}

    /**
     * Generate the internal acceleration structures and compute bounding box.
     * This must only be called once the base class has been populated with
     * its splats, and after calling it the underlying data must not be
     * modified again. This function must be called before any of the other
     * functions defined in this class.
     *
     * @param spacing        Grid spacing for grids to be accelerated.
     * @param bucketSize     Common factor for bucket sizes to be accelerated.
     * @param progressStream If non-NULL, will be used to report progress.
     * @param warnNonFinite  If true (the default), a warning will be displayed if
     *                       non-finite splats are encountered.
     */
    void computeBlobs(float spacing, Grid::size_type bucketSize,
                      std::ostream *progressStream = NULL,
                      bool warnNonFinite = true);

    /**
     * Return the bounding grid generated by @ref computeBlobs. The grid will
     * have an origin at the world origin and the @a spacing passed to @ref
     * computeBlobs.
     */
    const Grid &getBoundingGrid() const { return boundingGrid; }

    /**
     * Return the exact number of splats in the splat stream.
     * @pre @ref computeBlobs has been called.
     */
    splat_id numSplats() const
    {
        MLSGPU_ASSERT(internalBucketSize > 0, state_error);
        return nSplats;
    }

    splat_id maxSplats() const { return numSplats(); }

private:
    /**
     * The bucket size used to generate the blob data. It is initially zero
     * when the class is constructed, and is populated by @ref computeBlobs.
     */
    Grid::size_type internalBucketSize;
    /**
     * Bounding grid computed by @ref computeBlobs. It is initially undefined.
     */
    Grid boundingGrid;
    /**
     * Blob metadata computed by @ref computeBlobs. It is initially empty.
     */
    BlobVector blobData;
    std::size_t nSplats;  ///< Exact splat count computed during blob generation

    /**
     * Determines whether the given @a grid and @a bucketSize can use the
     * pre-generated blob data.
     */
    bool fastPath(const Grid &grid, Grid::size_type bucketSize) const;

    /**
     * Append a blob to @ref blobData.
     * @param blobData The list of encoded blobs to append to.
     * @param prevBlob The value @a curBlob had on previous call.
     * @param curBlob  The blob to append.
     * On the first call (i.e., when @a blobData is empty), the value of @a
     * prevBlob is irrelevant.
     */
    static void addBlob(BlobVector &blobData, const BlobInfo &prevBlob, const BlobInfo &curBlob);
};

/**
 * Abstracts the parts of @ref Subset that do not require knowledge of the
 * superclass type.
 *
 * @see @ref Subset.
 */
class SubsetBase
{
public:
    /**
     * Add a blob to the subset.
     * @pre
     * - @a blob.firstSplat is greater than any previously added splat.
     */
    void addBlob(const BlobInfo &blob);

    /**
     * Swap blob IDs with another subset. Note that this does not check that
     * the IDs make sense in the other set; use with caution. The intended use
     * case is to allow an instance of @ref SubsetBase to be used for
     * accumulating the information in type-agnostic code, then swapping into
     * an instance of @ref Subset at the end.
     */
    void swap(SubsetBase &other);

    /**
     * The number of splat ID ranges.
     */
    std::size_t numRanges() const { return splatRanges.size(); }

    splat_id numSplats() const { return nSplats; }
    splat_id maxSplats() const { return nSplats; }

    SubsetBase() : splatRanges("mem.SubsetBase::splatRange"), nSplats(0) {}

protected:
    /**
     * Store of splat ID ranges. Each range is a half-open interval of valid
     * IDs.
     */
    Statistics::Container::vector<std::pair<splat_id, splat_id> > splatRanges;

    /// Number of splats in the supplied blobs.
    splat_id nSplats;
};

/**
 * A subset of the splats from another set. Note that this class does not
 * implement the @ref SubsettableConcept, but since it matches its superset in
 * blob and splat IDs, it is possible to create a subset of a subset by
 * subsetting the superset. @ref Traits can be used to handle this distinction
 * transparently.
 *
 * Blob iteration is currently not very efficient: it iterates one splat at a time,
 * even if the superset is a @ref FastBlobSet.
 *
 * @param Super a model of @ref SubsettableConcept
 */
template<typename Super>
class Subset : public SubsetBase
#ifdef DOXYGEN_FAKE_CODE
, public SetConcept
#endif
{
public:
    SplatStream *makeSplatStream() const
    {
        return super.makeSplatStream(splatRanges.begin(), splatRanges.end());
    }

    BlobStream *makeBlobStream(const Grid &grid, Grid::size_type bucketSize) const;

    /**
     * Constructor to wrap superset.
     * @param super         Superset. This object holds a reference, so it must not be destroyed.
     *                      It must also not be modified.
     */
    Subset(const Super &super) : super(super)
    {
    }

    /**
     * Constructor to wrap superset of another subset. This allows a subset of
     * type <code>Subset<Traits<T>::subset_type></code> to be constructed by passing a
     * @c T, where @c T is either a @c SubsettableConcept or is @c Subset.
     */
    Subset(const Subset<Super> &peer) : super(peer.super)
    {
    }

private:
    const Super &super;             ///< Containing superset.
};

/**
 * Metaprogramming information about splat set types.
 *
 * @param T A model of @ref SetConcept
 */
template<typename T>
class Traits
{
public:
    /**
     * A type suitable for representing a subset of @a T.
     */
    typedef Subset<T> subset_type;
    /**
     * @c boost::true_type if @a T is @c Subset, otherwise @c
     * boost::false_type.
     */
    typedef boost::false_type is_subset;
};

template<typename T>
class Traits<Subset<T> >
{
public:
    typedef Subset<T> subset_type;
    typedef boost::true_type is_subset;
};

} // namespace SplatSet

#include "splat_set_impl.h"

#endif /* !SPLAT_SET_H */
