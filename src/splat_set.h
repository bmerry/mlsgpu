/**
 * @file
 *
 * Data structures for convenient and efficient iterations of collections of
 * collections of splats.
 */

#ifndef SPLAT_SET_H
#define SPLAT_SET_H

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <boost/noncopyable.hpp>
#include <boost/numeric/conversion/converter.hpp>
#include <boost/type_traits/remove_pointer.hpp>
#include <boost/smart_ptr/scoped_ptr.hpp>
#include <boost/array.hpp>
#include <boost/ref.hpp>
#include <boost/foreach.hpp>
#include <tr1/cstdint>
#include <ostream>
#include <cstddef>
#include "grid.h"
#include "splat.h"
#include "progress.h"
#include "errors.h"
#include "logging.h"

/**
 * Data structures for convenient and efficient iterations of collections of
 * collections of splats.
 */
namespace SplatSet
{

/**
 * A collection of collections of splats. Each collection is called a @em scan.
 * This is both a concrete class and a concept that is reimplemented by subclasses.
 * It provides mechanisms for iterating over all splats using an interface designed
 * for efficient bucketing.
 */
template<typename SplatCollectionSet>
class SimpleSet : public boost::noncopyable
{
public:
    typedef SplatCollectionSet value_type;
    typedef typename SplatCollectionSet::size_type scan_type;
    typedef typename boost::remove_pointer<typename SplatCollectionSet::value_type>::type Collection;
    typedef typename Collection::size_type index_type;

    SimpleSet(const SplatCollectionSet &splats, const Grid &grid);

    const SplatCollectionSet &getSplats() const { return splats; }
    const Grid &getGrid() const { return grid; }

    /**
     * Call a function for each contiguous range of splats occupying the
     * same bucket. Note that it is not guaranteed that the contiguous
     * ranges are maximal; it is simply an interface that allows more efficient
     * computation if the callee can process batches faster than individual
     * elements.
     *
     * The function signature should be
     * <code>void(scan_type scan, size_type first, size_type last,
     *            const boost::array<Grid::difference_type, 3> &lower,
     *            const boost::array<Grid::difference_type, 3> &upper);</code>
     * The lower and upper bounds are inclusive and are in units of the
     * buckets given on input. The function will be called sequentially (i.e.,
     * in order) on the splats.
     *
     * Splats that do not intersect the grid might or might not be passed to
     * the callback. The same applies to non-finite splats.
     *
     * @param func        The function to call.
     * @param bucketSize  The number of grid cells per bucket (in each dimension).
     */
    template<typename Func>
    void forEach(const Func &func, Grid::size_type bucketSize) const;

protected:
    /// Constructor that does not require a grid
    explicit SimpleSet(const SplatCollectionSet &splats);

    /**
     * The raw splats themselves.
     */
    const SplatCollectionSet &splats;

    /**
     * Grid bounding region to process.
     */
    Grid grid;

private:
    template<typename Func>
    void forEachOne(
        scan_type scan, index_type index, float bucketSpacing,
        const Splat &splat, const Func &func) const;
};

/**
 * Entry in a list representing splat positions in a compacted form.
 * Contiguous ranges of splats from the scans are placed into buckets
 * (which size is stored in @ref SplatSetBlobbed). Each contiguous range must
 * intersect the same cuboid of buckets. There are two cases:
 *  -# They intersect a single bucket. The coordinates of the bucket
 *     (in units of buckets) are stored in @ref coords, and the number
 *     of splats is stored in @ref size.
 *  -# They intersect multiple buckets. Two consecutive blobs are used
 *     to encode the range. The first has the lower-bound (inclusive)
 *     coordinates, and a @ref size of zero. The second has the
 *     upper-bound (inclusive) coordinates and the actual size of the
 *     range.
 * The coordinates are relative to the grid reference point rather than
 * the edge of the grid, and so need correcting before they can be useful.
 * The lower grid extent is deliberately chosen to be a multiple of the
 * bucket size so that alignment is not an issue.
 */
struct Blob
{
    typedef std::tr1::uint32_t size_type;

    boost::array<Grid::difference_type, 3> coords;
    std::tr1::uint32_t size;
};

/**
 * Contains a list of splat collections, together with metadata
 * that allows the splats to be bucketed more efficiently. It also computes
 * a bounding box during construction.
 *
 * @param SplatCollectionSet A random access container of @ref Collection of @ref Splat.
 * @param BlobCollection A forward container of @ref Blob.
 */
template<typename SplatCollectionSet, typename BlobCollection>
class BlobSet : public SimpleSet<SplatCollectionSet>
{
public:
    typedef typename SimpleSet<SplatCollectionSet>::scan_type scan_type;
    typedef typename SimpleSet<SplatCollectionSet>::index_type index_type;
    typedef typename SimpleSet<SplatCollectionSet>::Collection Collection;

    /**
     * Constructor. This will make a pass over all the splats to compute
     * the bounding box grid and populate the blob bucket.
     *
     * @param splats          The input splats.
     * @param spacing         The grid spacing for the final reconstruction.
     * @param blobBucket      The number of grid cells per bucket for blobs.
     * @param progressStream  If non-NULL, will be used to log progress through the splats.
     */
    BlobSet(const SplatCollectionSet &splats, float spacing, Grid::size_type blobBucket,
             std::ostream *progressStream = NULL);

    template<typename Func>
    void forEach(const Func &func, Grid::size_type bucketSize) const;

private:
    /// Data only needed during construction
    struct Build
    {
        float blobSpacing;
        typename SplatCollectionSet::size_type nonFinite;

        float bboxMin[3];
        float bboxMax[3];

        boost::array<Grid::difference_type, 3> blobLower, blobUpper;
        Blob::size_type blobSize;
    };

    /**
     * The number of grid cells per bucket for decoding the blobs in @ref blobs.
     */
    Grid::size_type blobBucket;

    /**
     * A list of blobs. The blobs can only be sensibly processed sequentially,
     * because this is the only way to maintain which splats each blob refers
     * to. Each blob refers to a range of splats immediately following the
     * previous blob. The blob collection will cover all scans in @ref splats,
     * but an individual blob does not span scan boundaries.
     */
    BlobCollection blobs;

    void flushBlob(Build &build);

    void processSplat(const Splat &splat, Build &build, ProgressDisplay *progress);
};


typedef boost::numeric::converter<
    Grid::difference_type,
    float,
    boost::numeric::conversion_traits<Grid::difference_type, float>,
    boost::numeric::def_overflow_handler,
    boost::numeric::Ceil<float> > GridRoundUp;
typedef boost::numeric::converter<
    Grid::difference_type,
    float,
    boost::numeric::conversion_traits<Grid::difference_type, float>,
    boost::numeric::def_overflow_handler,
    boost::numeric::Floor<float> > GridRoundDown;

template<typename SplatCollectionSet>
SimpleSet<SplatCollectionSet>::SimpleSet(const SplatCollectionSet &splats, const Grid &grid)
: splats(splats), grid(grid)
{
}

template<typename SplatCollectionSet>
SimpleSet<SplatCollectionSet>::SimpleSet(const SplatCollectionSet &splats)
: splats(splats)
{
}

template<typename SplatCollectionSet>
template<typename Func>
void SimpleSet<SplatCollectionSet>::forEachOne(
    scan_type scan, index_type index, float bucketSpacing,
    const Splat &splat, const Func &func) const
{
    if (splat.isFinite())
    {
        boost::array<Grid::difference_type, 3> lower, upper;
        for (int i = 0; i < 3; i++)
        {
            lower[i] = GridRoundDown::convert((splat.position[i] - splat.radius) / bucketSpacing);
            upper[i] = GridRoundDown::convert((splat.position[i] + splat.radius) / bucketSpacing);
        }
        func(scan, index, index + 1, lower, upper);
    }
}

template<typename SplatCollectionSet>
template<typename Func>
void SimpleSet<SplatCollectionSet>::forEach(const Func &func, Grid::size_type bucketSize) const
{
    scan_type scan = 0;
    const float bucketSpacing = grid.getSpacing() * bucketSize;
    BOOST_FOREACH(const Collection &c, splats)
    {
        c.forEach(0, c.size(), boost::bind(&SimpleSet<SplatCollectionSet>::forEachOne,
                                           this, scan, _1, bucketSpacing, _2, boost::cref(func)));
        scan++;
    }
}

template<typename SplatCollectionSet, typename BlobCollection>
BlobSet<SplatCollectionSet, BlobCollection>::BlobSet(
    const SplatCollectionSet &splats, float spacing, Grid::size_type blobBucket,
    std::ostream *progressStream)
: SimpleSet<SplatCollectionSet>(splats), blobBucket(blobBucket)
{
    MLSGPU_ASSERT(blobBucket > 0, std::invalid_argument);

    /* ptr_vector has T* as the value_type, so we have to strip off the
     * pointerness. No pointer type is a model of Collection so this won't
     * break for other containers.
     */
    typedef typename boost::remove_pointer<typename SplatCollectionSet::value_type>::type SplatCollection;

    boost::scoped_ptr<ProgressDisplay> progress;
    if (progressStream != NULL)
    {
        typename SplatCollection::size_type numSplats = 0;
        for (typename SplatCollectionSet::const_iterator i = splats.begin(); i != splats.end(); ++i)
        {
            numSplats += i->size();
        }
        *progressStream << "Computing bounding box\n";
        progress.reset(new ProgressDisplay(numSplats, *progressStream));
    }

    Build build;
    build.blobSpacing = spacing * blobBucket;
    build.nonFinite = 0;
    fill(build.bboxMin, build.bboxMin + 3, std::numeric_limits<float>::infinity());
    fill(build.bboxMax, build.bboxMax + 3, -std::numeric_limits<float>::infinity());
    build.blobSize = 0;
    for (typename SplatCollectionSet::const_iterator i = splats.begin(); i != splats.end(); ++i)
    {
        i->forEach(0, i->size(), boost::bind(&BlobSet<SplatCollectionSet, BlobCollection>::processSplat,
                                             this, _2, boost::ref(build), progress.get()));
        flushBlob(build); // ensure that blobs don't span scan boundaries
    }

    if (build.nonFinite > 0)
        Log::log[Log::warn] << "Input contains " << build.nonFinite << " splat(s) with non-finite values\n";
    if (build.bboxMin[0] > build.bboxMax[0])
        throw std::length_error("Must be at least one splat");

    int extents[3][2];
    for (unsigned int i = 0; i < 3; i++)
    {
        float l = build.bboxMin[i] / spacing;
        float h = build.bboxMax[i] / spacing;
        extents[i][0] = GridRoundDown::convert(l);
        extents[i][1] = GridRoundUp::convert(h);
        /* The lower extent must be a multiple of the blob bucket size, to
         * make the blob data align properly.
         */
        Grid::difference_type rem = extents[i][0] % blobBucket;
        if (rem < 0) rem += blobBucket;
        extents[i][0] -= rem;
        assert(extents[i][0] % blobBucket == 0);
    }

    const float ref[3] = {0.0f, 0.0f, 0.0f};
    this->grid = Grid(ref, spacing,
                      extents[0][0], extents[0][1],
                      extents[1][0], extents[1][1],
                      extents[2][0], extents[2][1]);

    Log::log[Log::info] << blobs.size() << " blobs extracted\n";
}

template<typename SplatCollectionSet, typename BlobCollection>
void BlobSet<SplatCollectionSet, BlobCollection>::flushBlob(Build &build)
{
    if (build.blobSize > 0)
    {
        if (build.blobLower != build.blobUpper)
        {
            Blob blob;
            blob.coords = build.blobLower;
            blob.size = 0;
            blobs.push_back(blob);
        }
        Blob blob;
        blob.coords = build.blobUpper;
        blob.size = build.blobSize;
        blobs.push_back(blob);

        build.blobSize = 0;
    }
}

template<typename SplatCollectionSet, typename BlobCollection>
void BlobSet<SplatCollectionSet, BlobCollection>::processSplat(const Splat &splat, Build &build, ProgressDisplay *progress)
{
    if (!splat.isFinite())
    {
        build.nonFinite++;
        flushBlob(build);
        // Create a blob with zero coverage
        Blob blob;
        for (unsigned int i = 0; i < 3; i++)
        {
            build.blobLower[i] = 1;
            build.blobUpper[i] = 0;
        }
        blob.size = 1;
        blobs.push_back(blob);
    }
    else
    {
        boost::array<Grid::difference_type, 3> lower, upper;
        for (unsigned int i = 0; i < 3; i++)
        {
            float lo = splat.position[i] - splat.radius;
            float hi = splat.position[i] + splat.radius;
            build.bboxMin[i] = std::min(build.bboxMin[i], lo);
            build.bboxMax[i] = std::max(build.bboxMax[i], hi);
            lower[i] = GridRoundDown::convert(lo / build.blobSpacing);
            upper[i] = GridRoundDown::convert(hi / build.blobSpacing);
        }

        if (build.blobSize == 0 || build.blobSize == std::numeric_limits<Blob::size_type>::max()
            || lower != build.blobLower || upper != build.blobUpper)
        {
            flushBlob(build);
            build.blobLower = lower;
            build.blobUpper = upper;
            build.blobSize = 0;
        }
        build.blobSize++;
    }

    if (progress != NULL)
        ++*progress;
}

template<typename SplatCollectionSet, typename BlobCollection>
template<typename Func>
void BlobSet<SplatCollectionSet, BlobCollection>::forEach(
    const Func &func, Grid::size_type bucketSize) const
{
    if (bucketSize % blobBucket == 0)
    {
        typename SplatCollectionSet::iterator curScan = this->splats.begin();
        scan_type scan = 0;
        index_type index = 0;

        Grid::size_type ratio = bucketSize / blobBucket;
        // Adjustments from reference-relative to extent-relative buckets
        Grid::difference_type adjust[3];
        for (unsigned int i = 0; i < 3; i++)
            adjust[i] = this->grid.getExtent(i).first / blobBucket;

        typename BlobCollection::const_iterator cur = blobs.begin();
        while (cur != blobs.end())
        {
            boost::array<Grid::difference_type, 3> lower, upper;
            for (unsigned int i = 0; i < 3; i++)
                lower[i] = (cur->coords[i] - adjust[i]) / ratio;

            if (cur->size == 0)
            {
                ++cur;
                for (unsigned int i = 0; i < 3; i++)
                    lower[i] = (cur->coords[i] - adjust[i]) / ratio;
            }
            else
                upper = lower;
            if (lower[0] <= upper[0]) // skips over non-finite splats
            {
                func(scan, index, index + cur->size, lower, upper);
            }
            index += cur->size;
            assert(curScan != this->splats.end() && index <= curScan);
            if (index == curScan->size())
            {
                ++scan;
                ++curScan;
                index = 0;
            }

            ++cur;
        }
    }
    else
    {
        SimpleSet<SplatCollectionSet>::forEach(func, bucketSize);
    }
}

} // namespace SplatSet

#endif /* !SPLAT_SET_H */
