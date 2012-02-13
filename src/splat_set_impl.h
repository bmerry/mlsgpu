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
        c.forEach(0, c.size(), boost::bind(&SimpleSet<SplatCollectionSet>::forEachOne<Func>,
                                           this, scan, _1, bucketSpacing, _2, boost::cref(func)));
        scan++;
    }
}

template<typename SplatCollectionSet>
template<typename RangeIterator, typename Func>
void SimpleSet<SplatCollectionSet>::forEachRange(
    RangeIterator first, RangeIterator last,
    const Func &func, Grid::size_type bucketSize) const
{
    const float bucketSpacing = grid.getSpacing() * bucketSize;
    for (RangeIterator i = first; i != last; ++i)
    {
        splats[i->scan].forEach(i->start, i->start + i->size,
                                boost::bind(&SimpleSet<SplatCollectionSet>::forEachOne<Func>,
                                            this, i->scan, _1, bucketSpacing, _2, boost::cref(func)));
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
        typename SplatCollectionSet::const_iterator curScan = this->splats.begin();
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
            assert(curScan != this->splats.end() && index <= curScan->size());
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

template<typename SplatCollectionSet, typename BlobCollection>
template<typename RangeIterator, typename Func>
void BlobSet<SplatCollectionSet, BlobCollection>::forEachRange(
    RangeIterator first, RangeIterator last,
    const Func &func, Grid::size_type bucketSize) const
{
    if (bucketSize % blobBucket == 0)
    {
        /* Special case: check whether the list of ranges covers the entire
         * set.
         */
        scan_type scan = 0;
        index_type index = 0;
        for (RangeIterator i = first; i != last; ++i)
        {
            assert(i->scan < this->splats.size());
            if (i->scan == scan && i->start == index)
            {
                index += i->size;
                assert(index <= this->splats[scan].size());
                if (index == this->splats[scan].size())
                {
                    scan++;
                    index = 0;
                }
            }
            else
                break;
        }
        if (scan == this->splats.size())
        {
            forEach(func, bucketSize);
            return;
        }
    }
    // General case
    SimpleSet<SplatCollectionSet>::forEachRange(first, last, func, bucketSize);
}

} // namespace SplatSet

#endif /* !SPLAT_SET_IMPL_H */
