/**
 * @file
 *
 * Bucketing of splats into sufficiently small buckets.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <vector>
#include <tr1/cstdint>
#include <limits>
#include <stdexcept>
#include <algorithm>
#include <boost/array.hpp>
#include <boost/multi_array.hpp>
#include <boost/foreach.hpp>
#include <boost/bind.hpp>
#include "splat.h"
#include "bucket.h"
#include "errors.h"

SplatRange::SplatRange() :
    scan(std::numeric_limits<scan_type>::max()),
    size(0),
    start(std::numeric_limits<index_type>::max())
{
}

SplatRange::SplatRange(scan_type scan, index_type splat) :
    scan(scan),
    size(1),
    start(splat)
{
}

SplatRange::SplatRange(scan_type scan, index_type start, size_type size)
    : scan(scan), size(size), start(start)
{
    MLSGPU_ASSERT(size == 0 || start <= std::numeric_limits<index_type>::max() - size + 1, std::out_of_range);
}

bool SplatRange::append(scan_type scan, index_type splat)
{
    if (size == 0)
    {
        /* An empty range can always be extended. */
        this->scan = scan;
        size = 1;
        start = splat;
    }
    else if (this->scan == scan && splat >= start && splat - start <= size)
    {
        if (splat - start == size)
        {
            if (size == std::numeric_limits<size_type>::max())
                return false; // would overflow
            size++;
        }
    }
    else
        return false;
    return true;
}

SplatRangeCounter::SplatRangeCounter() : ranges(0), splats(0), current()
{
}

void SplatRangeCounter::append(SplatRange::scan_type scan, SplatRange::index_type splat)
{
    splats++;
    /* On the first call, the append will succeed (empty range), but we still
     * need to set ranges to 1 since this is the first real range.
     */
    if (ranges == 0 || !current.append(scan, splat))
    {
        current = SplatRange(scan, splat);
        ranges++;
    }
}

std::tr1::uint64_t SplatRangeCounter::countRanges() const
{
    return ranges;
}

std::tr1::uint64_t SplatRangeCounter::countSplats() const
{
    return splats;
}

namespace
{

/**
 * Multiply @a a and @a b, clamping the result to the maximum value of the type
 * instead of overflowing.
 *
 * @pre @a a and @a b are non-negative.
 */
template<typename T>
static inline T mulSat(T a, T b)
{
    if (a == 0 || std::numeric_limits<T>::max() / a >= b)
        return a * b;
    else
        return std::numeric_limits<T>::max();
}

/**
 * Divide and round up.
 */
template<typename S, typename T>
static inline S divUp(S a, T b)
{
    return (a + b - 1) / b;
}

struct BucketParameters
{
    /// Input files holding the raw splats
    const std::vector<FastPly::Reader *> &files;
    const Grid &grid;                   ///< Bounding box for the entire region
    const BucketProcessor &process;     ///< Processing function
    SplatRange::index_type maxSplats;   ///< Maximum splats permitted for processing
    unsigned int maxCells;              ///< Maximum cells along any dimension
    std::size_t maxSplit;               ///< Maximum fan-out for recursion

    BucketParameters(const std::vector<FastPly::Reader *> &files, const Grid &grid,
                     const BucketProcessor &process)
        : files(files), grid(grid), process(process) {}
};

struct BucketState
{
    const BucketParameters &params;
    std::size_t dims[3];
    int microShift;
    int macroLevels;
    std::vector<boost::multi_array<SplatRangeCounter, 3> > counters;
    std::vector<boost::multi_array<std::size_t, 3> > blockIds;

    BucketState(const BucketParameters &params, const std::size_t dims[3],
                int microShift, int macroLevels);
};

BucketState::BucketState(
    const BucketParameters &params, const std::size_t dims[3],
    int microShift, int macroLevels)
    : params(params), microShift(microShift), macroLevels(macroLevels),
    counters(macroLevels), blockIds(macroLevels)
{
    this->dims[0] = dims[0];
    this->dims[1] = dims[1];
    this->dims[2] = dims[2];

    for (int level = 0; level < macroLevels; level++)
    {
        boost::array<std::size_t, 3> s;
        for (int i = 0; i < 3; i++)
            s[i] = divUp(dims[i], std::size_t(1) << (microShift + level));
        counters[level] = boost::multi_array<SplatRangeCounter, 3>(s);
        blockIds[level] = boost::multi_array<std::size_t, 3>(s);
        std::fill(blockIds[level].origin(),
                  blockIds[level].origin() + blockIds[level].num_elements(),
                  std::numeric_limits<std::size_t>::max());
    }
}

template<typename Func>
static void forEachCell_r(const std::size_t dims[3], const std::size_t base[3], int level, const Func &func)
{
    if (func(base, level))
    {
        if (level > 0)
        {
            const std::size_t half = std::size_t(1) << (level - 1);
            for (int i = 0; i < 8; i++)
            {
                const std::size_t base2[3] =
                {
                    base[0] + (i & 1 ? half : 0),
                    base[1] + (i & 2 ? half : 0),
                    base[2] + (i & 4 ? half : 0)
                };
                if (base2[0] < dims[0] && base2[1] < dims[1] && base2[2] < dims[2])
                    forEachCell_r(dims, base2, level - 1, func);
            }
        }
    }
}

template<typename Func>
static void forEachCell(const std::size_t dims[3], int levels, const Func &func)
{
    assert(levels >= 1);
    int level = levels - 1;
    assert((std::size_t(1) << level) >= dims[0]);
    assert((std::size_t(1) << level) >= dims[1]);
    assert((std::size_t(1) << level) >= dims[2]);

    const std::size_t base[3] = {0, 0, 0};
    forEachCell_r(dims, base, level, func);
}

template<typename Func>
static void forEachSplat(
    const std::vector<FastPly::Reader *> &files,
    const std::vector<SplatRange> &ranges,
    const Func &func)
{
    static const std::size_t splatBufferSize = 8192;

    /* First pass over the splats: count things up */
    BOOST_FOREACH(const SplatRange &range, ranges)
    {
        Splat buffer[splatBufferSize];
        SplatRange::size_type size = range.size;
        SplatRange::index_type start = range.start;
        while (size != 0)
        {
            SplatRange::size_type chunkSize = size;
            if (splatBufferSize < size)
                size = splatBufferSize;
            files[range.scan]->readVertices(start, chunkSize, buffer);
            for (std::size_t j = 0; j < chunkSize; j++)
            {
                func(range.scan, start + j, buffer[j]);
            }
            size -= chunkSize;
            start += chunkSize;
        }
    }
}

/**
 * Function object for use with forEachSplat that enters the splat
 * into all corresponding counters in the tree.
 */
class CountSplat
{
private:
    BucketState &state;

public:
    CountSplat(BucketState &state) : state(state) {};

    void operator()(SplatRange::scan_type scan, SplatRange::index_type id, const Splat &splat) const;
    bool doCell(
        SplatRange::scan_type scan, SplatRange::index_type id, const Splat &splat,
        const std::size_t base[3], int level) const;
};

void CountSplat::operator()(SplatRange::scan_type scan, SplatRange::index_type id, const Splat &splat) const
{
    forEachCell(state.dims, state.microShift + state.macroLevels,
                boost::bind(&CountSplat::doCell, this, scan, id, splat, _1, _2));
}

bool CountSplat::doCell(
    SplatRange::scan_type scan, SplatRange::index_type id, const Splat &splat,
    const std::size_t base[3], int level) const
{
    assert(level >= state.microShift);

    std::size_t size = std::size_t(1) << level;
    float lo[3], hi[3];

    state.params.grid.getVertex(base[0], base[1], base[2], lo);
    state.params.grid.getVertex(base[0] + size, base[1] + size, base[2] + size, hi);

    // Bounding box test. We don't bother an exact sphere-box test
    for (int i = 0; i < 3; i++)
        if (splat.position[i] + splat.radius < lo[i]
            || splat.position[i] - splat.radius > hi[i])
            return false;

    // Add to the counters
    boost::array<std::size_t, 3> coords;
    for (int i = 0; i < 3; i++)
        coords[i] = base[i] >> level;
    state.counters[level - state.microShift][coords[0]][coords[1]][coords[2]].append(scan, id);

    // Recurse into children, unless we've reached microblock level
    return level > state.microShift;
}

static void bucketRecurse(const std::vector<SplatRange> &node,
                          SplatRange::index_type numSplats,
                          const BucketParameters &params)
{
    std::size_t dims[3];
    for (int i = 0; i < 3; i++)
        dims[i] = params.grid.numCells(i);
    std::size_t maxDim = std::max(std::max(dims[0], dims[1]), dims[2]);

    if (numSplats <= params.maxSplats && maxDim <= params.maxCells)
    {
        params.process(params.files, node, params.grid);
    }
    else
    {
        /* Pick a power-of-two size such that we don't exceed maxSplit
         * microblocks.
         */
        std::size_t microSize = 1;
        int microShift = 0;
        std::size_t microBlocks;
        do
        {
            microSize *= 2;
            microShift++;
            microBlocks = 1;
            for (int i = 0; i < 3; i++)
                microBlocks = mulSat(microBlocks, divUp(dims[i], microSize));
        } while (microBlocks > params.maxSplit);

        /* Levels in octree-like structure */
        int macroLevels = 1;
        while (microSize << (macroLevels - 1) < maxDim)
            macroLevels++;

        BucketState state(params, dims, microShift, macroLevels);
        CountSplat countSplat(state);
        forEachSplat(params.files, node, countSplat);
    }
}

} // namespace

void bucket(const std::vector<FastPly::Reader *> &files,
            const Grid &bbox,
            SplatRange::index_type maxSplats,
            int maxCells,
            std::size_t maxSplit,
            const BucketProcessor &process)
{
    /* Create a root bucket will all splats in it */
    std::vector<SplatRange> root;
    SplatRange::index_type numSplats = 0;
    root.reserve(files.size());
    for (size_t i = 0; i < files.size(); i++)
    {
        const SplatRange::index_type vertices = files[i]->numVertices();
        numSplats += vertices;
        SplatRange::index_type start = 0;
        while (start < vertices)
        {
            SplatRange::size_type size = std::numeric_limits<SplatRange::size_type>::max();
            if (start + size > vertices)
                size = vertices - start;
            root.push_back(SplatRange(i, start, size));
            start += size;
        }
    }

    BucketParameters params(files, bbox, process);
    params.maxSplats = maxSplats;
    params.maxCells = maxCells;
    params.maxSplit = maxSplit;
    bucketRecurse(root, numSplats, params);
}
