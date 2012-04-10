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

#include <tr1/cstdint>
#include <vector>
#include <utility>
#include <algorithm>
#include <stdexcept>
#include <cassert>
#include <cstddef>
#include <iosfwd>
#include <boost/array.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/smart_ptr/scoped_ptr.hpp>
#include <boost/type_traits/integral_constant.hpp>
#include "grid.h"
#include "misc.h"
#include "splat.h"
#include "errors.h"
#include "fast_ply.h"
#include "statistics.h"
#include "logging.h"
#include "progress.h"
#include "allocator.h"

/**
 * Data structures for iteration over sets of splats.
 */
namespace SplatSet
{

typedef std::tr1::uint64_t splat_id;

/**
 * Metadata about a sequence of splats. The range of splat IDs must all be
 * valid splat ID, and hence the number of splats in the blob is exactly
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
    virtual const Splat &operator*() const = 0;

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
 * A subclass of splat stream that is able to relatively efficiently do random
 * access. It is still likely to be most effective when @ref reset is used
 * sparingly due to overheads.
 */
class SplatStreamReset : public SplatStream
{
public:
    /**
     * Restart iteration over a new range of IDs.
     * @pre @a first &lt;= @a last.
     */
    virtual void reset(splat_id first, splat_id last) = 0;
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
class SubsettableConcept : public SetConcept
{
public:
    /**
     * Create a splat stream that can be used for random access to the splats.
     * The returned stream is empty, and @ref SplatStreamReset::reset must be
     * used to select ranges to iterate over.
     */
    SplatStreamReset *makeSplatStreamReset() const;
};

#endif // DOXYGEN_FAKE_CODE

/**
 * Internal implementation details of @ref SplatSet.
 */
namespace internal
{

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
 * Partial splat set interface for a simple vector of splats. This is turned
 * into a model of @ref SetConcept using @ref BlobbedSet. The splat IDs are
 * simply the positions in the vector. It is legal to store non-finite splats;
 * they will be skipped over by the stream.
 *
 * All the public methods of @ref Statistics::Container::vector&lt;Splat&gt;
 * are available, but modifying the data while a stream exists yields undefined
 * behavior.
 */
class SimpleVectorSet : public Statistics::Container::vector<Splat>
{
public:
    splat_id maxSplats() const { return size(); }

    SplatStream *makeSplatStream() const
    {
        return new MySplatStream(*this, 0, size());
    }

    SplatStreamReset *makeSplatStreamReset() const
    {
        return new MySplatStream(*this, 0, 0);
    }

    SimpleVectorSet() : Statistics::Container::vector<Splat>("mem.SimpleVectorSet") {}

private:
    /// Splat stream implementation
    class MySplatStream : public SplatStreamReset
    {
    public:
        virtual const Splat &operator*() const
        {
            MLSGPU_ASSERT(!empty(), std::out_of_range);
            return owner.at(cur);
        }

        virtual SplatStream &operator++()
        {
            MLSGPU_ASSERT(!empty(), std::out_of_range);
            cur++;
            skipNonFinite();
            return *this;
        }

        virtual bool empty() const
        {
            return cur == last;
        }

        virtual void reset(splat_id first, splat_id last)
        {
            MLSGPU_ASSERT(first <= last, std::invalid_argument);
            if (owner.size() < last)
                last = owner.size();
            if (first > last)
                first = last;
            cur = first;
            this->last = last;
            skipNonFinite();
        }

        virtual splat_id currentId() const
        {
            return cur;
        }

        MySplatStream(const SimpleVectorSet &owner, splat_id first, splat_id last)
            : owner(owner)
        {
            reset(first, last);
        }

    private:
        const SimpleVectorSet &owner;
        splat_id cur;
        splat_id last;

        ///< Advances until reading a finite splat or the end of the range
        void skipNonFinite();
    };
};

/**
 * Splat-set core interface for a collection of on-disk PLY files.
 *
 * The splat IDs use the upper bits to store the file ID and the remaining
 * bits to store the splat index within the file.
 */
class SimpleFileSet
{
public:
    /// Number of bits used to store the within-file splat ID
    static const unsigned int scanIdShift = 40;
    /// Mask of the bits used to store the within-file splat ID
    static const splat_id splatIdMask = (splat_id(1) << scanIdShift) - 1;

    /**
     * Append a new file to the set. The set takes over ownership of the file.
     * This must not be called while a stream is in progress.
     */
    void addFile(FastPly::Reader *file);

    SplatStream *makeSplatStream() const
    {
        return new MySplatStream(*this, 0, splat_id(files.size()) << scanIdShift);
    }

    SplatStreamReset *makeSplatStreamReset() const
    {
        return new MySplatStream(*this, 0, 0);
    }

    splat_id maxSplats() const { return nSplats; }

    SimpleFileSet() : nSplats(0) {}

private:
    /// Splat stream implementation
    class MySplatStream : public SplatStreamReset
    {
    public:
        virtual const Splat &operator*() const
        {
            MLSGPU_ASSERT(!empty(), std::out_of_range);
            return buffer[bufferCur];
        }

        virtual SplatStream &operator++();

        virtual bool empty() const
        {
            return bufferCur == bufferEnd;
        }

        virtual splat_id currentId() const
        {
            return cur;
        }

        virtual void reset(splat_id first, splat_id last);

        MySplatStream(const SimpleFileSet &owner, splat_id first, splat_id last)
            : owner(owner)
        {
            reset(first, last);
        }

    private:
        /// Size of internal buffer for holding splats
        static const std::size_t bufferSize = 16384;

        const SimpleFileSet &owner;     ///< Owning set
        splat_id last;                  ///< End of range to iterate over
        splat_id next;                  ///< Next ID to load into the buffer once exhausted
        splat_id cur;                   ///< Splat ID at the front of the buffer
        std::size_t bufferCur;          ///< First position in @ref buffer with data
        std::size_t bufferEnd;          ///< Past-the-end position for @ref buffer
        Splat buffer[bufferSize];       ///< Buffer for splats read from file

        /**
         * Advances over non-finite elements in the buffer. It stops when a
         * finite splat is reached or the buffer is empty. This should only
         * be called by @ref refill.
         */
        void skipNonFiniteInBuffer();

        /**
         * Advance the stream until empty or a finite splat is reached. This
         * will refill the buffer if necessary.
         */
        void refill();
    };

    /// Backing store of files
    boost::ptr_vector<FastPly::Reader> files;

    /// Number of splats stored in the files (including non-finites)
    splat_id nSplats;
};

/**
 * Implementation of @ref BlobStream that just has one blob for each splat.
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
 * Adds the blob interface to a base class that does not have it, with blob
 * IDs simply equaling splat IDs.
 *
 * @param CoreSet A model of @ref SubsettableConcept that is missing the blob functions.
 */
template<typename CoreSet>
class BlobbedSet : public CoreSet
#ifdef DOXYGEN_FAKE_CODE
, public SubsettableConcept
#endif
{
public:
    BlobStream *makeBlobStream(const Grid &grid, Grid::size_type bucketSize) const
    {
        return new internal::SimpleBlobStream(CoreSet::makeSplatStream(), grid, bucketSize);
    }
};

} // namespace internal

/**
 * An implementation of @ref SubsettableConcept with data stored in multiple
 * PLY files.
 */
typedef internal::BlobbedSet<internal::SimpleFileSet> FileSet;

/**
 * An implementation of @ref SubsettableConcept with data stored in a single
 * vector.
 */
typedef internal::BlobbedSet<internal::SimpleVectorSet> VectorSet;

/**
 * Internal data stored in @ref FastBlobSet. This class is not accessed
 * directly by the user, but is in the namespace so that the user can generate
 * the type for the second template parameter to @ref FastBlobSet.
 */
typedef BlobInfo BlobData;

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
 * @param Base A model of @ref SubsettableConcept.
 */
template<typename Base, typename BlobVector>
class FastBlobSet : public Base
#ifdef DOXYGEN_FAKE_CODE
, public SubsettableConcept
#endif
{
public:
    /**
     * Class returned by makeBlobStream only in the fast path.
     */
    class MyBlobStream : public SplatSet::BlobStream
    {
    public:
        virtual BlobInfo operator*() const;

        virtual BlobStream &operator++()
        {
            MLSGPU_ASSERT(curBlob < lastBlob, std::length_error);
            ++curBlob;
            return *this;
        }

        virtual bool empty() const
        {
            return curBlob == lastBlob;
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
        typename BlobVector::size_type curBlob;     ///< Blob ID for the current blob
        typename BlobVector::size_type lastBlob;    ///< Past-the-end ID
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
    BlobVector blobs;
    std::size_t nSplats;  ///< Exact splat count computed during blob generation

    /**
     * Determines whether the given @a grid and @a bucketSize can use the
     * pre-generated blob data.
     */
    bool fastPath(const Grid &grid, Grid::size_type bucketSize) const;
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
     * The number of blob ID ranges.
     */
    std::size_t numRanges() const { return splatRanges.size(); }

    splat_id numSplats() const { return nSplats; }
    splat_id maxSplats() const { return nSplats; }

    SubsetBase() : splatRanges("mem.SubsetBase::splatRange"), nSplats(0) {}

protected:
    /**
     * Store of blob ID ranges. Each range is a half-open interval of valid
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
 * The members of the subset are specified using blobs of the superset. Thus,
 * the subset is specialized with a specific grid and bucket size which give
 * the blob view on the superset. Blob iteration over the subset using the
 * same grid and bucket size will be efficient since it will just iterate over
 * blobs from the superset (which may be particularly efficient if the
 * superset is a @ref FastBlobSet). Blob iteration using a different grid or
 * bucket size will simply iterate over all the splats.
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
        return new MySplatStream(*this);
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
    /// Splat stream implementation
    class MySplatStream : public SplatStream
    {
    public:
        virtual const Splat &operator *() const
        {
            return **child;
        }

        virtual SplatStream &operator++()
        {
            ++*child;
            refill();
            return *this;
        }

        virtual bool empty() const
        {
            return child->empty();
        }

        virtual splat_id currentId() const
        {
            return child->currentId();
        }

        MySplatStream(const Subset<Super> &owner)
            : owner(owner), splatRange(0), child(owner.super.makeSplatStreamReset())
        {
            refill();
        }

    private:
        const Subset<Super> &owner;
        std::size_t splatRange;        ///< Next blob range to load into @ref child
        boost::scoped_ptr<SplatStreamReset> child; ///< Stream for iterating over ranges of superset splats

        void refill();                ///< Advance to the next valid splat
    };

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
