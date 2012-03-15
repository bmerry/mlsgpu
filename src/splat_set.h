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

namespace SplatSet
{

typedef std::tr1::uint64_t blob_id;
typedef std::tr1::uint64_t splat_id;

struct BlobInfo
{
    blob_id id;
    splat_id numSplats;
    boost::array<Grid::difference_type, 3> lower, upper;
};

/**
 * Polymorphic interface for iteration over a sequence of splats. This is based
 * on the STXXL stream interface.
 *
 * Each splat has an ID. IDs are monotonic but not necessarily contiguous
 * (although they must be mostly contiguous for efficiency). A splat stream
 * iterates over all the valid IDs in a given range, or over all splats in a
 * splat set.
 *
 * An implementation of this interface is required to filter out splats with
 * non-finite elements. This can itself lead to discontiguous IDs, even if
 * the the container is flat.
 */
class SplatStream : public boost::noncopyable
{
public:
    typedef Splat value_type;
    typedef const Splat &reference;

    virtual ~SplatStream() {}

    virtual SplatStream &operator++() = 0; ///< Advance to the next legal splat

    /**
     * Return the currently pointed-to splat.
     * @pre !empty()
     */
    virtual const Splat &operator*() const = 0;

    /**
     * Determines whether there are any more splats to advance over.
     */
    virtual bool empty() const = 0;

    /**
     * Returns an ID for the current splat (the one that would be returned by
     * <code>operator *</code>).
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

    virtual BlobStream &operator++() = 0;
    virtual BlobInfo operator*() const = 0;
    virtual bool empty() const = 0;
};

class BlobStreamReset : public BlobStream
{
public:
    virtual void reset(blob_id firstBlob, blob_id lastBlob) = 0;
};

namespace internal
{

void splatToBuckets(const Splat &splat,
                    const Grid &grid, Grid::size_type bucketSize,
                    boost::array<Grid::difference_type, 3> &lower,
                    boost::array<Grid::difference_type, 3> &upper);

class SimpleVectorSet : public std::vector<Splat>
{
public:
    size_type maxSplats() const { return size(); }

    SplatStream *makeSplatStream() const
    {
        return new MySplatStream(*this, 0, size());
    }

    SplatStreamReset *makeSplatStreamReset() const
    {
        return new MySplatStream(*this, 0, 0);
    }

private:
    class MySplatStream : public SplatStreamReset
    {
    public:
        virtual const Splat &operator*() const
        {
            return owner.at(cur);
        }

        virtual SplatStream &operator++()
        {
            MLSGPU_ASSERT(!empty(), std::runtime_error);
            cur++;
            return *this;
        }

        virtual bool empty() const
        {
            return cur == last;
        }

        virtual void reset(splat_id first, splat_id last)
        {
            MLSGPU_ASSERT(first <= last, std::invalid_argument);
            MLSGPU_ASSERT(last <= owner.size(), std::length_error);
            cur = first;
            this->last = last;
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
    static const unsigned int scanIdShift = 40;
    static const splat_id splatIdMask = (splat_id(1) << scanIdShift) - 1;

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
    class MySplatStream : public SplatStreamReset
    {
    public:
        virtual const Splat &operator*() const
        {
            MLSGPU_ASSERT(!empty(), std::runtime_error);
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
        static const std::size_t bufferSize = 16384;

        const SimpleFileSet &owner;
        /// Total range to iterate over
        splat_id first, last;

        splat_id next;                  ///< Next ID to load into the buffer once exhausted
        splat_id cur;                   ///< Splat ID at the front of the buffer
        std::size_t bufferCur;          ///< First position in @ref buffer with data
        std::size_t bufferEnd;          ///< Past-the-end position for @ref buffer
        Splat buffer[bufferSize];       ///< Buffer for splats read from file

        void skipNonFiniteInBuffer();

        void refill();
    };

    boost::ptr_vector<FastPly::Reader> files;
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
    }

private:
    boost::scoped_ptr<SplatStream> splatStream;
    const Grid grid;
    Grid::size_type bucketSize;
};

class SimpleBlobStreamReset : public BlobStreamReset
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

    virtual void reset(blob_id firstId, blob_id lastId)
    {
        splatStream->reset(firstId, lastId);
    }

    SimpleBlobStreamReset(SplatStreamReset *splatStream, const Grid &grid, Grid::size_type bucketSize)
        : splatStream(splatStream), grid(grid), bucketSize(bucketSize)
    {
    }

private:
    boost::scoped_ptr<SplatStreamReset> splatStream;
    const Grid grid;
    Grid::size_type bucketSize;
};

/**
 * Takes a model of the basic interface and adds the blob interface, with blob IDs
 * simply equaling splat IDs and no acceleration.
 */
template<typename CoreSet>
class BlobbedSet : public CoreSet
{
public:
    BlobStream *makeBlobStream(const Grid &grid, Grid::size_type bucketSize) const
    {
        return new internal::SimpleBlobStream(CoreSet::makeSplatStream(), grid, bucketSize);
    }

    BlobStreamReset *makeBlobStreamReset(const Grid &grid, Grid::size_type bucketSize) const
    {
        return new internal::SimpleBlobStreamReset(CoreSet::makeSplatStreamReset(), grid, bucketSize);
    }

    std::pair<splat_id, splat_id> blobsToSplats(
        const Grid &grid, Grid::size_type bucketSize,
        blob_id firstBlob, blob_id lastBlob) const;
};

} // namespace internal

typedef internal::BlobbedSet<internal::SimpleFileSet> FileSet;
typedef internal::BlobbedSet<internal::SimpleVectorSet> VectorSet;

struct BlobData
{
    boost::array<Grid::difference_type, 3> lower, upper;
    splat_id firstSplat, lastSplat;
};

/**
 * Takes a model of the blobbed interface and extends it by precomputing
 * information about splats for a specific bucket size so that iteration
 * using that bucket size (or any multiple thereof) is potentially faster.
 */
template<typename Base, typename BlobVector>
class FastBlobSet : public Base
{
public:
    /**
     * Class returned by makeBlobStream only in the fast path.
     */
    class MyBlobStream : public SplatSet::BlobStreamReset
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

        virtual void reset(blob_id firstBlob, blob_id lastBlob);

        MyBlobStream(const FastBlobSet<Base, BlobVector> &owner, const Grid &grid,
                     Grid::size_type bucketSize,
                     blob_id firstBlob, blob_id lastBlob);

    private:
        const FastBlobSet<Base, BlobVector> &owner;
        Grid::size_type bucketRatio;
        Grid::difference_type offset[3];
        blob_id curBlob;
        blob_id lastBlob;
    };

    BlobStream *makeBlobStream(const Grid &grid, Grid::size_type bucketSize) const;

    BlobStreamReset *makeBlobStreamReset(const Grid &grid, Grid::size_type bucketSize) const;

    std::pair<splat_id, splat_id> blobsToSplats(const Grid &grid, Grid::size_type bucketSize,
                                                blob_id firstBlob, blob_id lastBlob) const;

    FastBlobSet() : Base(), internalBucketSize(0), nSplats(0) {}

    void computeBlobs(float spacing, Grid::size_type bucketSize, std::ostream *progressStream);

    const Grid &getBoundingGrid() const { return boundingGrid; }

    splat_id numSplats() const
    {
        MLSGPU_ASSERT(internalBucketSize > 0, std::runtime_error);
        return nSplats;
    }

private:
    Grid::size_type internalBucketSize;
    Grid boundingGrid;
    BlobVector blobs;
    std::size_t nSplats;  ///< Exact splat count computed during blob generation

    bool fastPath(const Grid &grid, Grid::size_type bucketSize) const;
};

class SubsetBase
{
public:
    void addBlob(const BlobInfo &blob);

    void swap(SubsetBase &other);

    std::size_t numRanges() const { return blobRanges.size(); }

    splat_id numSplats() const { return nSplats; }
    splat_id maxSplats() const { return nSplats; }

    SubsetBase() : nSplats(0) {}

protected:
    class MyBlobStream : public BlobStream
    {
    public:
        virtual BlobInfo operator*() const
        {
            return **child;
        }

        virtual BlobStream &operator++()
        {
            ++*child;
            refill();
            return *this;
        }

        virtual bool empty() const
        {
            return child->empty();
        }

        MyBlobStream(const SubsetBase &owner, BlobStreamReset *blobStream)
            : owner(owner), blobRange(0),
            child(blobStream)
        {
        }

    private:
        const SubsetBase &owner;
        std::size_t blobRange;       ///< Next blob range to load into child
        boost::scoped_ptr<BlobStreamReset> child;

        void refill();
    };

    std::vector<std::pair<blob_id, blob_id> > blobRanges;
    splat_id nSplats;
};

template<typename Super>
class Subset : public SubsetBase
{
public:
    SplatStream *makeSplatStream() const
    {
        return new MySplatStream(*this);
    }

    BlobStream *makeBlobStream(const Grid &grid, Grid::size_type bucketSize) const;

    Subset(const Super &super, const Grid &subGrid, Grid::size_type subBucketSize)
    : super(super), subGrid(subGrid), subBucketSize(subBucketSize)
    {
        MLSGPU_ASSERT(subBucketSize > 0, std::invalid_argument);
    }

    Subset(const Subset<Super> &peer, const Grid &subGrid, Grid::size_type subBucketSize)
    : super(peer.super), subGrid(subGrid), subBucketSize(subBucketSize)
    {
        MLSGPU_ASSERT(subBucketSize > 0, std::invalid_argument);
    }

private:
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
            : owner(owner), blobRange(0), child(owner.super.makeSplatStreamReset())
        {
            refill();
        }

    private:
        const Subset<Super> &owner;
        std::size_t blobRange;        ///< Next blob range to load into child
        boost::scoped_ptr<SplatStreamReset> child;

        void refill();
    };

    // TODO move this down to the base class along with subGrid and subBucketSize
    bool fastPath(const Grid &grid, Grid::size_type bucketSize) const;

    const Super &super;
    Grid subGrid;
    Grid::size_type subBucketSize;
};

template<typename T>
class Traits
{
public:
    typedef Subset<T> subset_type;
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
