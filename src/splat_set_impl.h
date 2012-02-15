/**
 * @file
 *
 * Implementations of the templates in splat_set.h.
 */

#ifndef SPLAT_SET_IMPL_H
#define SPLAT_SET_IMPL_H

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <boost/noncopyable.hpp>
#include <boost/numeric/conversion/converter.hpp>
#include <boost/smart_ptr/scoped_ptr.hpp>
#include <boost/array.hpp>
#include <boost/ref.hpp>
#include <boost/foreach.hpp>
#include <tr1/cstdint>
#include <ostream>
#include <cstddef>
#include <iterator>
#include <stxxl.h>
#include "grid.h"
#include "splat.h"
#include "progress.h"
#include "errors.h"
#include "logging.h"
#include "misc.h"
#include "statistics.h"

namespace SplatSet
{

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
SimpleSet<SplatCollectionSet>::SimpleSet(const SplatCollectionSet &splats)
: splats(splats)
{
}

template<typename SplatCollectionSet>
template<typename Func>
void SimpleSet<SplatCollectionSet>::forEachOne(
    scan_type scan, index_type index,
    const Grid &grid, Grid::size_type bucketSize,
    const Splat &splat, const Func &func) const
{
    if (splat.isFinite())
    {
        boost::array<Grid::difference_type, 3> lower, upper;
        detail::splatToBuckets(splat, grid, bucketSize, lower, upper);
        func(scan, index, index + 1, lower, upper);
    }
}

template<typename SplatCollectionSet>
template<typename Func>
void SimpleSet<SplatCollectionSet>::forEach(
    const Grid &grid, Grid::size_type bucketSize,
    const Func &func) const
{
    scan_type scan = 0;
    BOOST_FOREACH(const Collection &c, splats)
    {
        c.forEach(0, c.size(), boost::bind(&SimpleSet<SplatCollectionSet>::forEachOne<Func>,
                                           this, scan, _1, boost::cref(grid), bucketSize, _2, boost::cref(func)));
        scan++;
    }
}

template<typename SplatCollectionSet>
template<typename RangeIterator, typename Func>
void SimpleSet<SplatCollectionSet>::forEachRange(
    RangeIterator first, RangeIterator last,
    const Grid &grid, Grid::size_type bucketSize,
    const Func &func) const
{
    for (RangeIterator i = first; i != last; ++i)
    {
        splats[i->scan].forEach(i->start, i->start + i->size,
                                boost::bind(&SimpleSet<SplatCollectionSet>::forEachOne<Func>,
                                            this, i->scan, _1, boost::cref(grid), bucketSize, _2, boost::cref(func)));
    }
}

template<typename SplatCollectionSet, typename BlobCollection>
BlobSet<SplatCollectionSet, BlobCollection>::BlobSet(
    const SplatCollectionSet &splats, float spacing, Grid::size_type blobBucket,
    std::ostream *progressStream)
: SimpleSet<SplatCollectionSet>(splats), blobBucket(blobBucket)
{
    MLSGPU_ASSERT(blobBucket > 0, std::invalid_argument);
    Statistics::Registry &registry = Statistics::Registry::getInstance();

    // Reference point will be 0,0,0. Extents are set after reading all the splats
    boundingGrid.setSpacing(spacing);

    typedef typename std::iterator_traits<typename SplatCollectionSet::iterator>::value_type SplatCollection;

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
    build.nonFinite = 0;
    // Set sentinel values
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
    registry.getStatistic<Statistics::Variable>("blobset.nonfinite").add(build.nonFinite);

    if (build.bboxMin[0] > build.bboxMax[0])
        throw std::length_error("Must be at least one splat");

    for (unsigned int i = 0; i < 3; i++)
    {
        float l = build.bboxMin[i] / spacing;
        float h = build.bboxMax[i] / spacing;
        Grid::difference_type lo = GridRoundDown::convert(l);
        Grid::difference_type hi = GridRoundUp::convert(h);
        /* The lower extent must be a multiple of the blob bucket size, to
         * make the blob data align properly.
         */
        lo = divDown(lo, blobBucket) * blobBucket;
        assert(lo % Grid::difference_type(blobBucket) == 0);

        boundingGrid.setExtent(i, lo, hi);
    }
    registry.getStatistic<Statistics::Variable>("blobset.blobs").add(blobs.size());
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
void BlobSet<SplatCollectionSet, BlobCollection>::processSplat(
    const Splat &splat, Build &build, ProgressDisplay *progress)
{
    if (!splat.isFinite())
    {
        build.nonFinite++;
        flushBlob(build);
        // Create a blob with zero coverage
        for (unsigned int i = 0; i < 3; i++)
        {
            build.blobLower[i] = 1;
            build.blobUpper[i] = 0;
        }
        build.blobSize = 1;
        flushBlob(build);
    }
    else
    {
        boost::array<Grid::difference_type, 3> lower, upper;
        detail::splatToBuckets(splat, boundingGrid, blobBucket, lower, upper);

        if (build.blobSize == 0 || build.blobSize == std::numeric_limits<Blob::size_type>::max()
            || lower != build.blobLower || upper != build.blobUpper)
        {
            flushBlob(build);
            build.blobLower = lower;
            build.blobUpper = upper;
            build.blobSize = 0;
        }
        build.blobSize++;

        for (unsigned int i = 0; i < 3; i++)
        {
            build.bboxMin[i] = std::min(build.bboxMin[i], splat.position[i] - splat.radius);
            build.bboxMax[i] = std::max(build.bboxMax[i], splat.position[i] + splat.radius);
        }
    }

    if (progress != NULL)
        ++*progress;
}

template<typename SplatCollectionSet, typename BlobCollection>
bool BlobSet<SplatCollectionSet, BlobCollection>::fastGrid(
    const Grid &grid, Grid::size_type bucketSize) const
{
    // Check whether we can use the fast path
    if (bucketSize % blobBucket != 0)
        return false;
    if (boundingGrid.getSpacing() != grid.getSpacing())
        return false;
    for (unsigned int i = 0; i < 3; i++)
    {
        if (grid.getReference()[i] != 0.0f
            || grid.getExtent(i).first % Grid::difference_type(blobBucket) != 0)
            return false;
    }
    return true;
}

template<typename SplatCollectionSet, typename BlobCollection>
template<typename Func>
void BlobSet<SplatCollectionSet, BlobCollection>::forEachFast(
    const Grid &grid, Grid::size_type bucketSize,
    const Func &func) const
{
    typename SplatCollectionSet::const_iterator curScan = this->splats.begin();
    scan_type scan = 0;
    index_type index = 0;
    while (curScan != this->splats.end() && curScan->size() == 0)
    {
        ++scan;
        ++curScan;
    }

    Grid::size_type ratio = bucketSize / blobBucket;
    // Adjustments from reference-relative to extent-relative buckets
    Grid::difference_type adjust[3];
    for (unsigned int i = 0; i < 3; i++)
        adjust[i] = grid.getExtent(i).first / Grid::difference_type(blobBucket);

    typename stxxl::stream::streamify_traits<typename BlobCollection::const_iterator>::stream_type
        blobStream = stxxl::stream::streamify(blobs.begin(), blobs.end());
    while (!blobStream.empty())
    {
        Blob cur = *blobStream; ++blobStream;
        assert(curScan != this->splats.end());
        assert(index < curScan->size());
        bool good = true;

        boost::array<Grid::difference_type, 3> lower, upper;
        for (unsigned int i = 0; i < 3; i++)
            lower[i] = divDown(cur.coords[i] - adjust[i], ratio);

        if (cur.size == 0)
        {
            Grid::difference_type lx = cur.coords[0];
            cur = *blobStream; ++blobStream;
            if (cur.coords[0] < lx)
                good = false;   // this is how non-finite splats are encoded
            for (unsigned int i = 0; i < 3; i++)
                upper[i] = divDown(cur.coords[i] - adjust[i], ratio);
        }
        else
            upper = lower;
        if (good) // skips over non-finite splats
        {
            func(scan, index, index + cur.size, lower, upper);
        }
        index += cur.size;
        while (curScan != this->splats.end() && index >= curScan->size())
        {
            assert(index == curScan->size());
            ++scan;
            ++curScan;
            index = 0;
        }
    }
}

template<typename SplatCollectionSet, typename BlobCollection>
template<typename Func>
void BlobSet<SplatCollectionSet, BlobCollection>::forEach(
    const Grid &grid, Grid::size_type bucketSize,
    const Func &func) const
{
    bool fast = fastGrid(grid, bucketSize);
    if (fast)
    {
        forEachFast(grid, bucketSize, func);
    }
    else
    {
        SimpleSet<SplatCollectionSet>::forEach(grid, bucketSize, func);
    }
    Statistics::getStatistic<Statistics::Variable>("blobset.foreach.fast").add(fast);
}

template<typename SplatCollectionSet, typename BlobCollection>
template<typename RangeIterator, typename Func>
void BlobSet<SplatCollectionSet, BlobCollection>::forEachRange(
    RangeIterator first, RangeIterator last,
    const Grid &grid, Grid::size_type bucketSize,
    const Func &func) const
{
    bool fast = fastGrid(grid, bucketSize);
    if (fast)
    {
        /* Special case: check whether the list of ranges covers the entire
         * set.
         */
        scan_type scan = 0;
        index_type index = 0;
        while (scan < this->splats.size() && this->splats[scan].size() == 0)
            scan++;
        for (RangeIterator i = first; i != last; ++i)
        {
            assert(i->scan < this->splats.size());
            if (i->size == 0)
                continue; // probably should not happen, but it might for empty scans
            if (i->scan == scan && i->start == index)
            {
                index += i->size;
                assert(index <= this->splats[scan].size());
                while (scan < this->splats.size() && index == this->splats[scan].size())
                {
                    scan++;
                    index = 0;
                }
            }
            else
                break;
        }
        fast = (scan == this->splats.size());
    }

    if (fast)
        forEachFast(grid, bucketSize, func);
    else
        SimpleSet<SplatCollectionSet>::forEachRange(first, last, grid, bucketSize, func);

    Statistics::getStatistic<Statistics::Variable>("blobset.foreachrange.fast").add(fast);
}

} // namespace SplatSet

#endif /* !SPLAT_SET_IMPL_H */
