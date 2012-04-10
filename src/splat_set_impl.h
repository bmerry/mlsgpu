/**
 * @file
 *
 * Implementations of template members from @ref splat_set.h.
 */

#ifndef SPLAT_SET_IMPL_H
#define SPLAT_SET_IMPL_H

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include "splat_set.h"

namespace SplatSet
{

template<typename Base, typename BlobVector>
BlobInfo FastBlobSet<Base, BlobVector>::MyBlobStream::operator*() const
{
    BlobInfo ans;
    MLSGPU_ASSERT(curBlob < lastBlob, std::out_of_range);
    BlobData data = owner.blobs[curBlob];
    ans.firstSplat = data.firstSplat;
    ans.lastSplat = data.lastSplat;
    for (unsigned int i = 0; i < 3; i++)
        ans.lower[i] = divDown(data.lower[i] - offset[i], bucketRatio);
    for (unsigned int i = 0; i < 3; i++)
        ans.upper[i] = divDown(data.upper[i] - offset[i], bucketRatio);
    return ans;
}

template<typename Base, typename BlobVector>
FastBlobSet<Base, BlobVector>::MyBlobStream::MyBlobStream(
    const FastBlobSet<Base, BlobVector> &owner, const Grid &grid,
    Grid::size_type bucketSize)
: owner(owner)
{
    MLSGPU_ASSERT(bucketSize > 0 && owner.internalBucketSize > 0
                  && bucketSize % owner.internalBucketSize == 0, std::invalid_argument);
    for (unsigned int i = 0; i < 3; i++)
        offset[i] = grid.getExtent(i).first / Grid::difference_type(owner.internalBucketSize);
    bucketRatio = bucketSize / owner.internalBucketSize;
    curBlob = 0;
    lastBlob = owner.blobs.size();
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

template<typename Base, typename BlobVector>
void FastBlobSet<Base, BlobVector>::computeBlobs(
    float spacing, Grid::size_type bucketSize, std::ostream *progressStream, bool warnNonFinite)
{
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
        detail::splatToBuckets(splat, boundingGrid, bucketSize, blob.lower, blob.upper);
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
        if (progress != NULL)
            *progress += nonFinite;
        if (warnNonFinite)
            Log::log[Log::warn] << "Input contains " << nonFinite << " splat(s) with non-finite values\n";
    }
    registry.getStatistic<Statistics::Variable>("blobset.nonfinite").add(nonFinite);

    if (bboxMin[0] > bboxMax[0])
        throw std::runtime_error("Must be at least one splat");

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


template<typename Super>
BlobStream *Subset<Super>::makeBlobStream(const Grid &grid, Grid::size_type bucketSize) const
{
    MLSGPU_ASSERT(bucketSize > 0, std::invalid_argument);
    return new detail::SimpleBlobStream(makeSplatStream(), grid, bucketSize);
}

template<typename Super>
void Subset<Super>::MySplatStream::refill()
{
    while (child->empty() && splatRange < owner.splatRanges.size())
    {
        const std::pair<splat_id, splat_id> &range = owner.splatRanges[splatRange];
        child->reset(range.first, range.second);
        splatRange++;
    }
}

} // namespace SplatSet

#endif /* !SPLAT_SET_IMPL_H */
