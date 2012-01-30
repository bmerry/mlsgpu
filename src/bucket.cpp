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
#include <utility>
#include <cassert>
#include <functional>
#include <boost/array.hpp>
#include <boost/multi_array.hpp>
#include <boost/foreach.hpp>
#include <boost/ref.hpp>
#include <boost/numeric/conversion/converter.hpp>
#include <stxxl.h>
#include "splat.h"
#include "bucket.h"
#include "bucket_internal.h"
#include "errors.h"
#include "statistics.h"
#include "timer.h"
#include "misc.h"

namespace Bucket
{

Range::Range() :
    size(0),
    start(std::numeric_limits<index_type>::max())
{
}

Range::Range(index_type splat) :
    size(1),
    start(splat)
{
}

Range::Range(index_type start, size_type size)
    : size(size), start(start)
{
    MLSGPU_ASSERT(size == 0 || start <= std::numeric_limits<index_type>::max() - size + 1, std::out_of_range);
}

bool Range::append(index_type splat)
{
    if (size == 0)
    {
        /* An empty range can always be extended. */
        size = 1;
        start = splat;
    }
    else if (splat >= start && splat - start <= size)
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

namespace internal
{

Cell::Cell(const size_type lower[3], const size_type upper[3], unsigned int level) : level(level)
{
    for (unsigned int i = 0; i < 3; i++)
    {
        MLSGPU_ASSERT(lower[i] < upper[i], std::invalid_argument);
        this->lower[i] = lower[i];
        this->upper[i] = upper[i];
    }
}

Cell::Cell(
    size_type lowerX, size_type lowerY, size_type lowerZ,
    size_type upperX, size_type upperY, size_type upperZ,
    unsigned int level) : level(level)
{
    lower[0] = lowerX;
    lower[1] = lowerY;
    lower[2] = lowerZ;
    upper[0] = upperX;
    upper[1] = upperY;
    upper[2] = upperZ;
    for (unsigned int i = 0; i < 3; i++)
        MLSGPU_ASSERT(lower[i] < upper[i], std::invalid_argument);
}

bool Cell::operator==(const Cell &c) const
{
    return std::equal(lower, lower + 3, c.lower)
        && std::equal(upper, upper + 3, c.upper)
        && level == c.level;
}

Cell Cell::child(unsigned int idx) const
{
    MLSGPU_ASSERT(level > 0, std::invalid_argument);
    MLSGPU_ASSERT(idx < 8, std::invalid_argument);
    Cell c = *this;
    c.level--;
    for (int i = 0; i < 3; i++)
    {
        size_type mid = (lower[i] + upper[i]) / 2;
        if ((idx >> i) & 1)
            c.lower[i] = mid;
        else
            c.upper[i] = mid;
    }
    return c;
}

RangeCounter::RangeCounter() : ranges(0), splats(0), current()
{
}

void RangeCounter::append(Range::index_type splat)
{
    splats++;
    /* On the first call, the append will succeed (empty range), but we still
     * need to set ranges to 1 since this is the first real range.
     */
    if (ranges == 0 || !current.append(splat))
    {
        current = Range(splat);
        ranges++;
    }
}

std::tr1::uint64_t RangeCounter::countRanges() const
{
    return ranges;
}

std::tr1::uint64_t RangeCounter::countSplats() const
{
    return splats;
}

} // namespace internal

namespace
{

/// Contains static information used to process a cell.
struct BucketParameters
{
    const SplatVector &splats;          ///< Input vector holding the sorted splats
    const Processor &process;           ///< Processing function
    Range::index_type maxSplats;        ///< Maximum splats permitted for processing
    unsigned int maxCells;              ///< Maximum cells along any dimension
    std::size_t maxSplit;               ///< Maximum fan-out for recursion

    BucketParameters(const SplatVector &splats,
                     const Processor &process, Range::index_type maxSplats,
                     unsigned int maxCells, std::size_t maxSplit)
        : splats(splats), process(process),
        maxSplats(maxSplats), maxCells(maxCells), maxSplit(maxSplit) {}
};

static const std::size_t BAD_BLOCK = std::size_t(-1);

/**
 * Dynamic state that is updated as part of processing a cell.
 */
struct BucketState
{
    /**
     * A single node in an octree of counters.
     * Each node counts the number of splats and ranges that would be
     * necessary to turn that node's spatial extent into a bucket.
     */
    struct CellState
    {
        /// Counts potential splats and ranges for this octree node
        internal::RangeCounter counter;
        /**
         * Index of the block. This is initially BAD_BLOCK, but for
         * blocks that are selected for the next level of recursion
         * it becomes an index into the @c picked and @c pickedOffset
         * arrays.
         */
        std::size_t blockId;

        CellState() : blockId(BAD_BLOCK) {}
    };

    const BucketParameters &params;
    /// Grid covering just the region being processed
    const Grid &grid;
    /**
     * Size (in grid cells) of the region being processed.
     * This is just a cache of grid.numCells for ease of passing to @ref internal::forEachCell.
     */
    internal::Cell::size_type dims[3];
    /// Side length of a microblock
    internal::Cell::size_type microSize;
    /// Number of levels in the octree of counters.
    int macroLevels;
    /**
     * Octree of counters. Each element of the vector is one level of the
     * octree.  Element zero contains the finest level, higher elements the
     * coarser levels.
     */
    std::vector<boost::multi_array<CellState, 3> > cellStates;
    /**
     * Cells from the octree that were selected for the next level of
     * recursion, either because they're microblocks.
     */
    std::vector<internal::Cell> picked;
    /**
     * Start of the ranges for each picked cell. The ranges for
     * <code>picked[i]</code> are in slots <code>pickedOffset[i]</code> to
     * <code>pickedOffset[i+1]</code> of @ref childCur.
     */
    std::vector<std::tr1::uint64_t> pickedOffset;
    /// The next value to write into @ref pickedOffset when a new cell is picked
    std::tr1::uint64_t nextOffset;

    /**
     * The ranges for the next level of the hierarchy. This is semantically a list
     * of lists of ranges, with splits designated by @ref pickedOffset.
     */
    std::vector<internal::RangeCollector<std::vector<Range>::iterator> > childCur;

    BucketState(const BucketParameters &params, const Grid &grid,
                internal::Cell::size_type microSize, int macroLevels);

    /// Retrieves a reference to an octree node
    CellState &getCellState(const internal::Cell &cell);
    /// Retrieves a reference to an octree node
    const CellState &getCellState(const internal::Cell &cell) const;
};

BucketState::BucketState(
    const BucketParameters &params, const Grid &grid,
    internal::Cell::size_type microSize, int macroLevels)
    : params(params), grid(grid), microSize(microSize), macroLevels(macroLevels),
    cellStates(macroLevels), nextOffset(0)
{
    for (int i = 0; i < 3; i++)
        this->dims[i] = grid.numCells(i);

    for (int level = 0; level < macroLevels; level++)
    {
        /* We don't need a full power-of-two allocation for a level of the octree,
         * just enough to completely cover the original dimensions.
         */
        boost::array<internal::Cell::size_type, 3> s;
        for (int i = 0; i < 3; i++)
        {
            s[i] = divUp(dims[i], microSize << level);
            assert(level != macroLevels - 1 || s[i] == 1);
        }
        cellStates[level].resize(s);
    }
}

BucketState::CellState &BucketState::getCellState(const internal::Cell &cell)
{
    boost::array<internal::Cell::size_type, 3> coords;
    internal::Cell::size_type factor = microSize << cell.getLevel();
    for (int i = 0; i < 3; i++)
    {
        assert(cell.getLower()[i] % factor == 0);
        coords[i] = cell.getLower()[i] / factor;
    }
    assert(cell.getLevel() < cellStates.size());
    return cellStates[cell.getLevel()](coords);
}

const BucketState::CellState &BucketState::getCellState(const internal::Cell &cell) const
{
    return const_cast<BucketState *>(this)->getCellState(cell);
}

/**
 * Function object for use with @ref Bucket::internal::forEachSplatCell that enters the splat
 * into all corresponding counters in the tree.
 */
class CountSplat
{
private:
    BucketState &state;

public:
    CountSplat(BucketState &state) : state(state) {};

    bool operator()(Range::index_type id, const Splat &splat, const internal::Cell &cell) const;
};

bool CountSplat::operator()(Range::index_type id, const Splat &splat, const internal::Cell &cell) const
{
    (void) splat;

    // Add to the counters
    state.getCellState(cell).counter.append(id);

    // Recurse into children, unless we've reached microblock level
    return cell.getLevel() > 0;
}

/**
 * Functor for @ref Bucket::internal::forEachCell that chooses which cells to make blocks
 * out of. A cell is chosen if it contains few enough splats and is
 * small enough, or if it is a microblock. Otherwise it is split.
 */
class PickCells
{
private:
    BucketState &state;
public:
    PickCells(BucketState &state) : state(state) {}
    bool operator()(const internal::Cell &cell) const;
};

bool PickCells::operator()(const internal::Cell &cell) const
{
    BucketState::CellState &cs = state.getCellState(cell);

    // Skip completely empty regions
    if (cs.counter.countSplats() == 0)
        return false;

    if (cell.getLevel() == 0
        || (cell.getSize(0) <= state.params.maxCells
            && cell.getSize(1) <= state.params.maxCells
            && cell.getSize(2) <= state.params.maxCells
            && cs.counter.countSplats() <= state.params.maxSplats))
    {
        cs.blockId = state.picked.size();
        state.picked.push_back(cell);
        state.pickedOffset.push_back(state.nextOffset);
        state.nextOffset += cs.counter.countRanges();
        return false; // no more recursion required
    }
    else
        return true;
}

/**
 * Functor for @ref Bucket::internal::forEachSplatCell that places splat information into the allocated buckets.
 */
class BucketSplats
{
private:
    BucketState &state;

public:
    BucketSplats(BucketState &state) : state(state) {}

    bool operator()(Range::index_type id, const Splat &splat, const internal::Cell &cell) const;
};

bool BucketSplats::operator()(Range::index_type id, const Splat &splat, const internal::Cell &cell) const
{
    (void) splat;

    BucketState::CellState &cs = state.getCellState(cell);
    if (cs.blockId == BAD_BLOCK)
    {
        return true; // this cell is too coarse, so refine recursively
    }
    else
    {
        assert(cs.blockId < state.childCur.size());
        state.childCur[cs.blockId].append(id);
        return false;
    }
}

/**
 * Pick the smallest possible power of 2 size for a microblock.
 * The limitation is thta there must be at most @a maxSplit microblocks.
 */
static internal::Cell::size_type chooseMicroSize(
    const internal::Cell::size_type dims[3], std::size_t maxSplit)
{
    internal::Cell::size_type microSize = 1;
    std::size_t microBlocks = 1;
    for (int i = 0; i < 3; i++)
        microBlocks = mulSat(microBlocks, std::size_t(divUp(dims[i], microSize)));
    while (microBlocks > maxSplit)
    {
        microSize *= 2;
        microBlocks = 1;
        for (int i = 0; i < 3; i++)
            microBlocks = mulSat(microBlocks, std::size_t(divUp(dims[i], microSize)));
    }
    return microSize;
}

/**
 * Recursive implementation of @ref bucket.
 *
 * @param first,last      Range of ranges to process for this spatial region.
 * @param numSplats       Number of splats encoded into [@a first, @a last).
 * @param params          User parameters.
 * @param recursionDepth  Number of higher-level @ref bucketRecurse invocations on the stack.
 * @param totalRanges     Number of ranges held in memory across all levels of the recursion.
 */
static void bucketRecurse(RangeConstIterator first,
                          RangeConstIterator last,
                          Range::index_type numSplats,
                          const Grid &grid,
                          const BucketParameters &params,
                          unsigned int recursionDepth,
                          Range::index_type totalRanges)
{
    Statistics::getStatistic<Statistics::Peak<unsigned int> >("bucket.depth.peak").set(recursionDepth);
    Statistics::getStatistic<Statistics::Peak<Range::index_type> >("bucket.totalRanges.peak").set(totalRanges);

    internal::Cell::size_type dims[3];
    for (int i = 0; i < 3; i++)
        dims[i] = grid.numCells(i);
    internal::Cell::size_type maxDim = std::max(std::max(dims[0], dims[1]), dims[2]);

    if (numSplats <= params.maxSplats && maxDim <= params.maxCells)
    {
        params.process(params.splats, numSplats, first, last, grid);
    }
    else if (maxDim == 1)
    {
        throw DensityError(numSplats); // can't subdivide a 1x1x1 cell
    }
    else
    {
        /* Pick a microblock size such that we don't exceed maxSplit
         * microblocks. If the currently cell is bigger than maxCells
         * in any direction we use a power of two times maxCells, otherwise
         * we use a power of 2.
         */
        internal::Cell::size_type microSize;
        if (maxDim > params.maxCells)
        {
            // number of maxCells-sized blocks
            internal::Cell::size_type subDims[3];
            for (int i = 0; i < 3; i++)
                subDims[i] = divUp(dims[i], params.maxCells);
            microSize = params.maxCells * chooseMicroSize(subDims, params.maxSplit);
        }
        else
            microSize = chooseMicroSize(dims, params.maxSplit);

        /* Levels in octree-like structure */
        int macroLevels = 1;
        while (microSize << (macroLevels - 1) < maxDim)
            macroLevels++;

        std::vector<Range> childRanges;
        std::vector<std::tr1::uint64_t> savedOffset;
        std::vector<internal::Cell> savedPicked;
        std::vector<Range::index_type> savedNumSplats;
        std::size_t numPicked;
        /* Open a scope so that we can destroy the BucketState later */
        {
            BucketState state(params, grid, microSize, macroLevels);
            /* Create histogram */
            internal::forEachSplatCell(params.splats, first, last, grid, state.microSize, state.macroLevels, CountSplat(state));
            /* Select cells to bucket splats into */
            internal::forEachCell(state.dims, state.microSize, state.macroLevels, PickCells(state));
            /* Do the bucketing.
             */
            // Add sentinel for easy extraction of subranges
            state.pickedOffset.push_back(state.nextOffset);
            childRanges.resize(state.nextOffset);
            numPicked = state.picked.size();
            state.childCur.reserve(numPicked);
            for (std::size_t i = 0; i < numPicked; i++)
                state.childCur.push_back(internal::RangeCollector<std::vector<Range>::iterator>(
                        childRanges.begin() + state.pickedOffset[i]));
            internal::forEachSplatCell(params.splats, first, last, grid, state.microSize, state.macroLevels, BucketSplats(state));
            for (std::size_t i = 0; i < numPicked; i++)
                state.childCur[i].flush();

            /* Bucketing is complete but we have a lot of memory allocated that we
             * don't need for the recursive step. Copy it outside this scope then
             * close the scope to destroy state.
             */
            savedPicked.swap(state.picked);
            savedOffset.swap(state.pickedOffset);
            savedNumSplats.reserve(numPicked);
            for (std::size_t i = 0; i < numPicked; i++)
                savedNumSplats.push_back(state.getCellState(savedPicked[i]).counter.countSplats());
        }

        /* Now recurse into the chosen cells */
        for (std::size_t i = 0; i < numPicked; i++)
        {
            const internal::Cell &cell = savedPicked[i];
            const internal::Cell::size_type *lower = cell.getLower();
            internal::Cell::size_type upper[3];
            std::copy(cell.getUpper(), cell.getUpper() + 3, upper);
            // Clip the cell to the grid
            for (int j = 0; j < 3; j++)
            {
                upper[j] = std::min(upper[j], dims[j]);
                assert(lower[j] < upper[j]);
            }
            Grid childGrid = grid.subGrid(
                lower[0], upper[0], lower[1], upper[1], lower[2], upper[2]);
            bucketRecurse(childRanges.begin() + savedOffset[i],
                          childRanges.begin() + savedOffset[i + 1],
                          savedNumSplats[i],
                          childGrid,
                          params,
                          recursionDepth + 1,
                          totalRanges + childRanges.size());
        }
    }
}

struct MakeGrid
{
    typedef Splat value_type;

    bool first;
    float low[3];
    float bboxMin[3];
    float bboxMax[3];

    MakeGrid() : first(true) {}
    const Splat &operator()(const Splat &splat);
};

const Splat &MakeGrid::operator()(const Splat &splat)
{
    const float radius = splat.radius;
    if (first)
    {
        for (unsigned int i = 0; i < 3; i++)
        {
            low[i] = splat.position[i];
            bboxMin[i] = low[i] - radius;
            bboxMax[i] = low[i] + radius;
        }
        first = false;
    }
    else
    {
        for (unsigned int j = 0; j < 3; j++)
        {
            float p = splat.position[j];
            low[j] = std::min(low[j], p);
            bboxMin[j] = std::min(bboxMin[j], p - radius);
            bboxMax[j] = std::max(bboxMax[j], p + radius);
        }
    }
    return splat;
}

} // namespace

void bucket(const SplatVector &splats,
            const Grid &bbox,
            Range::index_type maxSplats,
            int maxCells,
            std::size_t maxSplit,
            const Processor &process)
{
    Range::index_type numSplats = splats.size();
    std::vector<Range> root(1, Range(0, numSplats));

    BucketParameters params(splats, process, maxSplats, maxCells, maxSplit);
    params.maxSplats = maxSplats;
    params.maxCells = maxCells;
    params.maxSplit = maxSplit;
    bucketRecurse(root.begin(), root.end(), numSplats, bbox, params, 0, root.size());
}

void loadSplats(const boost::ptr_vector<FastPly::Reader> &files,
                float spacing,
                bool sort,
                SplatVector &splats,
                Grid &grid)
{
    Statistics::Timer timer("loadSplats.time");

    Range::index_type numSplats = 0;
    BOOST_FOREACH(const FastPly::Reader &reader, files)
    {
        numSplats += reader.numVertices();
    }
    if (numSplats == 0)
        throw std::length_error("Must be at least one splat");

    const unsigned int block_size = SplatVector::block_size;
    typedef internal::FilesStream<boost::ptr_vector<FastPly::Reader>::const_iterator> files_type;
    typedef stxxl::stream::transform<MakeGrid, files_type> peek_type;
    typedef CompareSplatsMorton comparator_type;
    typedef stxxl::stream::sort<peek_type, comparator_type, block_size> sort_type;

    MakeGrid state;

    splats.resize(numSplats);
    files_type filesStream(files.begin(), files.end());
    peek_type peekStream(state, filesStream);
    if (sort)
    {
        // TODO: make the memory use tunable
        sort_type sortStream(peekStream, comparator_type(), 256 * 1024 * 1024);
        // A pass-through transformation that will extract the bounding box as we go
        stxxl::stream::materialize(sortStream, splats.begin(), splats.end());
    }
    else
    {
        stxxl::stream::materialize(peekStream, splats.begin(), splats.end());
    }

    int extents[3][2];
    for (unsigned int i = 0; i < 3; i++)
    {
        float l = (state.bboxMin[i] - state.low[i]) / spacing;
        float h = (state.bboxMax[i] - state.low[i]) / spacing;
        extents[i][0] = RoundDown::convert(l);
        extents[i][1] = RoundUp::convert(h);
    }

    grid = Grid(state.low, spacing,
                extents[0][0], extents[0][1], extents[1][0], extents[1][1], extents[2][0], extents[2][1]);
}

} // namespace Bucket
