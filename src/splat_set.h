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

    void addFile(FastPly::Reader *file)
    {
        files.push_back(file);
        nSplats += file->size();
    }

    SplatStream *makeSplatStream() const
    {
        return new MySplatStream(*this, 0, splat_id(files.size()) << scanIdShift);
    }

    SplatStreamReset *makeSplatStreamReset() const
    {
        return new MySplatStream(*this, 0, 0);
    }

    splat_id maxSplats() const
    {
        return nSplats;
    }

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

        virtual SplatStream &operator++()
        {
            MLSGPU_ASSERT(!empty(), std::runtime_error);
            bufferCur++;
            cur++;
            refill();
            return *this;
        }

        virtual bool empty() const
        {
            return bufferCur == bufferEnd;
        }

        virtual splat_id currentId() const
        {
            return cur;
        }

        virtual void reset(splat_id first, splat_id last)
        {
            MLSGPU_ASSERT(first <= last, std::invalid_argument);
            this->first = first;
            this->last = last;
            bufferCur = 0;
            bufferEnd = 0;
            next = first;
            refill();
        }

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

        void skipNonFiniteInBuffer()
        {
            while (bufferCur < bufferEnd && !buffer[bufferCur].isFinite())
            {
                bufferCur++;
                cur++;
            }
        }

        void refill()
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

    std::pair<splat_id, splat_id> blobsToSplats(const Grid &grid, Grid::size_type bucketSize, blob_id firstBlob, blob_id lastBlob) const
    {
        (void) grid;
        (void) bucketSize;
        MLSGPU_ASSERT(bucketSize > 0, std::invalid_argument);
        return std::make_pair(firstBlob, lastBlob);
    }
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
        virtual BlobInfo operator*() const
        {
            BlobInfo ans;
            MLSGPU_ASSERT(curBlob < lastBlob, std::length_error);
            BlobData data = owner.blobs[curBlob];
            ans.numSplats = data.lastSplat - data.firstSplat;
            ans.id = curBlob;
            for (unsigned int i = 0; i < 3; i++)
                ans.lower[i] = divDown(data.lower[i] - offset[i], bucketRatio);
            for (unsigned int i = 0; i < 3; i++)
                ans.upper[i] = divDown(data.upper[i] - offset[i], bucketRatio);
            return ans;
        }

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

        virtual void reset(blob_id firstBlob, blob_id lastBlob)
        {
            MLSGPU_ASSERT(firstBlob <= lastBlob, std::invalid_argument);
            MLSGPU_ASSERT(lastBlob <= owner.blobs.size(), std::length_error);
            curBlob = firstBlob;
            this->lastBlob = lastBlob;
        }

        MyBlobStream(const FastBlobSet<Base, BlobVector> &owner, const Grid &grid,
                     Grid::size_type bucketSize,
                     blob_id firstBlob, blob_id lastBlob)
            : owner(owner)
        {
            MLSGPU_ASSERT(bucketSize > 0 && owner.internalBucketSize > 0
                   && bucketSize % owner.internalBucketSize == 0, std::invalid_argument);
            for (unsigned int i = 0; i < 3; i++)
                offset[i] = grid.getExtent(i).first / Grid::difference_type(owner.internalBucketSize);
            reset(firstBlob, lastBlob);
            bucketRatio = bucketSize / owner.internalBucketSize;
        }

    private:
        const FastBlobSet<Base, BlobVector> &owner;
        Grid::size_type bucketRatio;
        Grid::difference_type offset[3];
        blob_id curBlob;
        blob_id lastBlob;
    };

    BlobStream *makeBlobStream(const Grid &grid, Grid::size_type bucketSize) const
    {
        if (fastPath(grid, bucketSize))
            return new MyBlobStream(*this, grid, bucketSize, 0, blobs.size());
        else
            return Base::makeBlobStream(grid, bucketSize);
    }

    BlobStreamReset *makeBlobStreamReset(const Grid &grid, Grid::size_type bucketSize) const
    {
        if (fastPath(grid, bucketSize))
            return new MyBlobStream(*this, grid, bucketSize, 0, 0);
        else
            return Base::makeBlobStreamReset(grid, bucketSize);
    }

    std::pair<splat_id, splat_id> blobsToSplats(const Grid &grid, Grid::size_type bucketSize,
                                                blob_id firstBlob, blob_id lastBlob) const
    {
        if (fastPath(grid, bucketSize))
        {
            MLSGPU_ASSERT(firstBlob <= lastBlob, std::invalid_argument);
            MLSGPU_ASSERT(lastBlob <= blobs.size(), std::length_error);
            if (firstBlob == lastBlob)
                return std::make_pair(splat_id(0), splat_id(0));
            else
            {
                splat_id firstSplat = blobs[firstBlob].firstSplat;
                splat_id lastSplat = blobs[lastBlob - 1].lastSplat;
                return std::make_pair(firstSplat, lastSplat);
            }
        }
        else
            return Base::blobsToSplats(grid, bucketSize, firstBlob, lastBlob);
    }

    FastBlobSet() : Base(), internalBucketSize(0), nSplats(0) {}

    void computeBlobs(float spacing, Grid::size_type bucketSize, std::ostream *progressStream)
    {
        // TODO: move into separate impl file
        const float ref[3] = {0.0f, 0.0f, 0.0f};

        MLSGPU_ASSERT(bucketSize > 0, std::invalid_argument);
        Statistics::Registry &registry = Statistics::Registry::getInstance();

        blobs.clear();
        internalBucketSize = bucketSize;

        // Reference point will be 0,0,0. Extents are set after reading all the spla
        boundingGrid.setSpacing(spacing);
        boundingGrid.setReference(ref);

        boost::scoped_ptr<ProgressDisplay> progress;
        if (progressStream != NULL)
        {
            *progressStream << "Computing bounding box\n";
            progress.reset(new ProgressDisplay(Base::maxSplats(), *progressStream));
        }

        boost::array<float, 3> bboxMin, bboxMax;
        // Set sentinel values
        std::fill(bboxMin.begin(), bboxMin.end(), std::numeric_limits<float>::infinity());
        std::fill(bboxMax.begin(), bboxMax.end(), -std::numeric_limits<float>::infinity());

        boost::scoped_ptr<SplatStream> splats(Base::makeSplatStream());
        nSplats = 0;
        while (!splats->empty())
        {
            const Splat &splat = **splats;
            splat_id id = splats->currentId();

            BlobData blob;
            internal::splatToBuckets(splat, boundingGrid, bucketSize, blob.lower, blob.upper);
            if (blobs.empty()
                || blobs.back().lower != blob.lower
                || blobs.back().upper != blob.upper
                || blobs.back().lastSplat != id)
            {
                blob.firstSplat = id;
                blob.lastSplat = id + 1;
                blobs.push_back(blob);
            }
            else
            {
                blobs.back().lastSplat++;
            }
            ++*splats;
            ++nSplats;
            if (progress != NULL)
                ++*progress;

            for (unsigned int i = 0; i < 3; i++)
            {
                bboxMin[i] = std::min(bboxMin[i], splat.position[i] - splat.radius);
                bboxMax[i] = std::max(bboxMax[i], splat.position[i] + splat.radius);
            }
        }

        assert(nSplats <= Base::maxSplats());
        splat_id nonFinite = Base::maxSplats() - nSplats;
        if (nonFinite > 0)
        {
            *progress += nonFinite;
            Log::log[Log::warn] << "Input contains " << nonFinite << " splat(s) with non-finite values\n";
        }
        registry.getStatistic<Statistics::Variable>("blobset.nonfinite").add(nonFinite);

        if (bboxMin[0] > bboxMax[0])
            throw std::length_error("Must be at least one splat");

        for (unsigned int i = 0; i < 3; i++)
        {
            float l = bboxMin[i] / spacing;
            float h = bboxMax[i] / spacing;
            Grid::difference_type lo = Grid::RoundDown::convert(l);
            Grid::difference_type hi = Grid::RoundUp::convert(h);
            /* The lower extent must be a multiple of the bucket size, to
             * make the blob data align properly.
             */
            lo = divDown(lo, bucketSize) * bucketSize;
            assert(lo % Grid::difference_type(bucketSize) == 0);

            boundingGrid.setExtent(i, lo, hi);
        }
        registry.getStatistic<Statistics::Variable>("blobset.blobs").add(blobs.size());
    }

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

    bool fastPath(const Grid &grid, Grid::size_type bucketSize) const
    {
        MLSGPU_ASSERT(internalBucketSize > 0, std::runtime_error);
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
};

class SubsetBase
{
public:
    void addBlob(const BlobInfo &blob)
    {
        if (blobRanges.empty() || blobRanges.back().second != blob.id)
            blobRanges.push_back(std::make_pair(blob.id, blob.id + 1));
        else
            blobRanges.back().second++;
        nSplats += blob.numSplats;
    }

    void swap(SubsetBase &other)
    {
        blobRanges.swap(other.blobRanges);
        std::swap(nSplats, other.nSplats);
    }

    std::size_t numRanges() const { return blobRanges.size(); }

    splat_id numSplats() const { return nSplats; }
    splat_id maxSplats() const { return nSplats; }

    SubsetBase() : nSplats(0) {}

protected:
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

    BlobStream *makeBlobStream(const Grid &grid, Grid::size_type bucketSize) const
    {
        MLSGPU_ASSERT(bucketSize > 0, std::invalid_argument);
        if (fastPath(grid, bucketSize))
            return new MyBlobStream(*this);
        else
            return new internal::SimpleBlobStream(makeSplatStream(), grid, bucketSize);
    }

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

        void refill()
        {
            while (child->empty() && blobRange < owner.blobRanges.size())
            {
                const std::pair<blob_id, blob_id> &range = owner.blobRanges[blobRange];
                const std::pair<splat_id, splat_id> splatRange = owner.super.blobsToSplats(
                    owner.subGrid, owner.subBucketSize, range.first, range.second);
                child->reset(splatRange.first, splatRange.second);
                blobRange++;
            }
        }
    };

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

        MyBlobStream(const Subset<Super> &owner)
            : owner(owner), blobRange(0),
            child(owner.super.makeBlobStreamReset(owner.subGrid, owner.subBucketSize))
        {
        }

    private:
        const Subset<Super> &owner;
        std::size_t blobRange;       ///< Next blob range to load into child
        boost::scoped_ptr<BlobStreamReset> child;

        void refill()
        {
            while (child->empty() && blobRange < owner.blobRanges.size())
            {
                const std::pair<blob_id, blob_id> &range = owner.blobRanges[blobRange];
                child->reset(range.first, range.second);
                blobRange++;
            }
        }
    };

    bool fastPath(const Grid &grid, Grid::size_type bucketSize) const
    {
        MLSGPU_ASSERT(bucketSize > 0, std::invalid_argument);
        if (bucketSize != subBucketSize)
            return false;
        if (subGrid.getSpacing() != grid.getSpacing())
            return false;
        for (unsigned int i = 0; i < 3; i++)
        {
            if (subGrid.getReference()[i] != grid.getReference()[i]
                || subGrid.getExtent(i).first != grid.getExtent(i).first)
            return false;
        }
        return true;
    }

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

#endif /* !SPLAT_SET_H */
