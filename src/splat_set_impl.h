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

namespace internal
{

template<typename CoreSet>
std::pair<splat_id, splat_id> BlobbedSet<CoreSet>::blobsToSplats(
    const Grid &grid, Grid::size_type bucketSize,
    blob_id firstBlob, blob_id lastBlob) const
{
    (void) grid;
    (void) bucketSize;
    MLSGPU_ASSERT(bucketSize > 0, std::invalid_argument);
    // Splat IDs are the same as blob IDs
    return std::make_pair(firstBlob, lastBlob);
}

} // namespace internal

template<typename Base, typename BlobVector>
BlobInfo FastBlobSet<Base, BlobVector>::MyBlobStream::operator*() const
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

template<typename Base, typename BlobVector>
void FastBlobSet<Base, BlobVector>::MyBlobStream::reset(blob_id firstBlob, blob_id lastBlob)
{
    MLSGPU_ASSERT(firstBlob <= lastBlob, std::invalid_argument);
    if (owner.blobs.size() < lastBlob)
        lastBlob = owner.blobs.size();
    curBlob = std::min(firstBlob, lastBlob);
    this->lastBlob = lastBlob;
}

template<typename Base, typename BlobVector>
FastBlobSet<Base, BlobVector>::MyBlobStream::MyBlobStream(
    const FastBlobSet<Base, BlobVector> &owner, const Grid &grid,
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

template<typename Base, typename BlobVector>
BlobStream *FastBlobSet<Base, BlobVector>::makeBlobStream(
    const Grid &grid, Grid::size_type bucketSize) const
{
    if (fastPath(grid, bucketSize))
        return new MyBlobStream(*this, grid, bucketSize, 0, blobs.size());
    else
        return Base::makeBlobStream(grid, bucketSize);
}

template<typename Base, typename BlobVector>
BlobStreamReset *FastBlobSet<Base, BlobVector>::makeBlobStreamReset(
    const Grid &grid, Grid::size_type bucketSize) const
{
    if (fastPath(grid, bucketSize))
        return new MyBlobStream(*this, grid, bucketSize, 0, 0);
    else
        return Base::makeBlobStreamReset(grid, bucketSize);
}

template<typename Base, typename BlobVector>
std::pair<splat_id, splat_id> FastBlobSet<Base, BlobVector>::blobsToSplats(
    const Grid &grid, Grid::size_type bucketSize,
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

template<typename Base, typename BlobVector>
void FastBlobSet<Base, BlobVector>::computeBlobs(
    float spacing, Grid::size_type bucketSize, std::ostream *progressStream)
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
        if (progress != NULL)
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

template<typename Base, typename BlobVector>
bool FastBlobSet<Base, BlobVector>::fastPath(const Grid &grid, Grid::size_type bucketSize) const
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


template<typename Super>
BlobStream *Subset<Super>::makeBlobStream(const Grid &grid, Grid::size_type bucketSize) const
{
    MLSGPU_ASSERT(bucketSize > 0, std::invalid_argument);
    if (fastPath(grid, bucketSize))
        return new MyBlobStream(*this, super.makeBlobStreamReset(grid, bucketSize));
    else
        return new internal::SimpleBlobStream(makeSplatStream(), grid, bucketSize);
}

template<typename Super>
void Subset<Super>::MySplatStream::refill()
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

template<typename Super>
bool Subset<Super>::fastPath(const Grid &grid, Grid::size_type bucketSize) const
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

} // namespace SplatSet

#endif /* !SPLAT_SET_IMPL_H */
