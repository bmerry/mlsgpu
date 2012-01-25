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
#include <boost/array.hpp>
#include <boost/multi_array.hpp>
#include <boost/foreach.hpp>
#include <boost/ref.hpp>
#include <boost/numeric/conversion/converter.hpp>
#include "splat.h"
#include "bucket.h"
#include "bucket_internal.h"
#include "errors.h"
#include "statistics.h"
#include "timer.h"

typedef boost::numeric::converter<
    int,
    float,
    boost::numeric::conversion_traits<int, float>,
    boost::numeric::def_overflow_handler,
    boost::numeric::Ceil<float> > RoundUp;
typedef boost::numeric::converter<
    int,
    float,
    boost::numeric::conversion_traits<int, float>,
    boost::numeric::def_overflow_handler,
    boost::numeric::Floor<float> > RoundDown;

namespace Bucket
{

Range::Range() :
    scan(std::numeric_limits<scan_type>::max()),
    size(0),
    start(std::numeric_limits<index_type>::max())
{
}

Range::Range(scan_type scan, index_type splat) :
    scan(scan),
    size(1),
    start(splat)
{
}

Range::Range(scan_type scan, index_type start, size_type size)
    : scan(scan), size(size), start(start)
{
    MLSGPU_ASSERT(size == 0 || start <= std::numeric_limits<index_type>::max() - size + 1, std::out_of_range);
}

bool Range::append(scan_type scan, index_type splat)
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

void RangeCounter::append(Range::scan_type scan, Range::index_type splat)
{
    splats++;
    /* On the first call, the append will succeed (empty range), but we still
     * need to set ranges to 1 since this is the first real range.
     */
    if (ranges == 0 || !current.append(scan, splat))
    {
        current = Range(scan, splat);
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

bool splatCellIntersect(const Splat &splat, const internal::Cell &cell, const Grid &grid)
{
    const Cell::size_type *lower = cell.getLower();
    const Cell::size_type *upper = cell.getUpper();

    float lo[3], hi[3];
    grid.getVertex(lower[0], lower[1], lower[2], lo);
    grid.getVertex(upper[0], upper[1], upper[2], hi);

    // Bounding box test. We don't bother with an exact sphere-box test
    for (int i = 0; i < 3; i++)
        if (splat.position[i] + splat.radius < lo[i]
            || splat.position[i] - splat.radius > hi[i])
            return false;
    return true;
}

} // namespace internal

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

/// Contains static information used to process a cell.
struct BucketParameters
{
    /// Input files holding the raw splats
    const boost::ptr_vector<FastPly::Reader> &files;
    const Grid &grid;                   ///< Bounding box for the entire region
    const Processor &process;           ///< Processing function
    Range::index_type maxSplats;        ///< Maximum splats permitted for processing
    unsigned int maxCells;              ///< Maximum cells along any dimension
    std::size_t maxSplit;               ///< Maximum fan-out for recursion

    BucketParameters(const boost::ptr_vector<FastPly::Reader> &files, const Grid &grid,
                     const Processor &process, Range::index_type maxSplats,
                     unsigned int maxCells, std::size_t maxSplit)
        : files(files), grid(grid), process(process),
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
    /// Size (in grid cells) of the region being processed
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

    BucketState(const BucketParameters &params, const internal::Cell::size_type dims[3],
                internal::Cell::size_type microSize, int macroLevels);

    /// Retrieves a reference to an octree node
    CellState &getCellState(const internal::Cell &cell);
    /// Retrieves a reference to an octree node
    const CellState &getCellState(const internal::Cell &cell) const;
};

BucketState::BucketState(
    const BucketParameters &params, const internal::Cell::size_type dims[3],
    internal::Cell::size_type microSize, int macroLevels)
    : params(params), microSize(microSize), macroLevels(macroLevels),
    cellStates(macroLevels), nextOffset(0)
{
    this->dims[0] = dims[0];
    this->dims[1] = dims[1];
    this->dims[2] = dims[2];

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
 * Function object for use with @ref Bucket::internal::forEachSplat that enters the splat
 * into all corresponding counters in the tree.
 */
class CountSplat
{
private:
    BucketState &state;

    /**
     * Functor for @ref Bucket::internal::forEachCell that enters a single splat into the counters in the
     * hierarchy.
     */
    class CountOneSplat
    {
    private:
        BucketState &state;
        Range::scan_type scan;
        Range::index_type id;
        const Splat &splat;

    public:
        CountOneSplat(BucketState &state, Range::scan_type scan, Range::index_type id, const Splat &splat)
            : state(state), scan(scan), id(id), splat(splat) {}

        bool operator()(const internal::Cell &cell) const;
    };

public:
    CountSplat(BucketState &state) : state(state) {};

    void operator()(Range::scan_type scan, Range::index_type id, const Splat &splat) const;
};

void CountSplat::operator()(Range::scan_type scan, Range::index_type id, const Splat &splat) const
{
    CountOneSplat helper(state, scan, id, splat);
    internal::forEachCell(state.dims, state.microSize, state.macroLevels, helper);
}

bool CountSplat::CountOneSplat::operator()(const internal::Cell &cell) const
{
    if (!splatCellIntersect(splat, cell, state.params.grid))
        return false;

    // Add to the counters
    state.getCellState(cell).counter.append(scan, id);

    // Recurse into children, unless we've reached microblock level
    return cell.getLevel() > 0;
}

/**
 * Functor for @ref Bucket::internal::forEachCell that chooses which cells to make blocks
 * out of. A cell is chosen if it contains few enough splats and is
 * small enough, or if it is a microblock. Otherwise it is split.
 *
 * 
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
 * Functor for @ref Bucket::internal::forEachSplat that places splat information into the allocated buckets.
 */
class BucketSplats
{
private:
    BucketState &state;

    /**
     * Functor for @ref Bucket::internal::forEachCell that enters one splat into the relevant cells.
     */
    class BucketOneSplat
    {
    private:
        BucketState &state;
        Range::scan_type scan;
        Range::index_type id;
        const Splat &splat;

    public:
        BucketOneSplat(BucketState &state, Range::scan_type scan, Range::index_type id, const Splat &splat)
            : state(state), scan(scan), id(id), splat(splat) {}

        bool operator()(const internal::Cell &cell) const;
    };
public:
    BucketSplats(BucketState &state) : state(state) {}

    void operator()(Range::scan_type scan, Range::index_type id, const Splat &splat) const;
};

bool BucketSplats::BucketOneSplat::operator()(const internal::Cell &cell) const
{
    if (!splatCellIntersect(splat, cell, state.params.grid))
        return false;

    BucketState::CellState &cs = state.getCellState(cell);
    if (cs.blockId == BAD_BLOCK)
    {
        return true; // this cell is too coarse, so refine recursively
    }
    else
    {
        assert(cs.blockId < state.childCur.size());
        state.childCur[cs.blockId].append(scan, id);
        return false;
    }
}

void BucketSplats::operator()(Range::scan_type scan, Range::index_type id, const Splat &splat) const
{
    BucketOneSplat helper(state, scan, id, splat);
    internal::forEachCell(state.dims, state.microSize, state.macroLevels, helper);
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

static void bucketRecurse(RangeConstIterator first,
                          RangeConstIterator last,
                          Range::index_type numSplats,
                          const BucketParameters &params)
{
    internal::Cell::size_type dims[3];
    for (int i = 0; i < 3; i++)
        dims[i] = params.grid.numCells(i);
    internal::Cell::size_type maxDim = std::max(std::max(dims[0], dims[1]), dims[2]);

    if (numSplats <= params.maxSplats && maxDim <= params.maxCells)
    {
        params.process(params.files, numSplats, first, last, params.grid);
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
            BucketState state(params, dims, microSize, macroLevels);
            /* Create histogram */
            internal::forEachSplat(params.files, first, last, CountSplat(state));
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
            internal::forEachSplat(params.files, first, last, BucketSplats(state));
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
                internal::Cell::size_type limit = params.grid.numCells(j);
                upper[j] = std::min(upper[j], limit);
                assert(lower[j] < upper[j]);
            }
            Grid childGrid = params.grid.subGrid(
                lower[0], upper[0], lower[1], upper[1], lower[2], upper[2]);
            BucketParameters childParams(params.files, childGrid, params.process,
                                         params.maxSplats, params.maxCells, params.maxSplit);
            bucketRecurse(childRanges.begin() + savedOffset[i],
                          childRanges.begin() + savedOffset[i + 1],
                          savedNumSplats[i],
                          childParams);
        }
    }
}

/* Create a root bucket will all splats in it */
static Range::index_type makeRoot(
    const boost::ptr_vector<FastPly::Reader> &files,
    std::vector<Range> &root)
{
    Range::index_type numSplats = 0;
    root.clear();
    root.reserve(files.size());
    for (size_t i = 0; i < files.size(); i++)
    {
        const Range::index_type vertices = files[i].numVertices();
        numSplats += vertices;
        Range::index_type start = 0;
        while (start < vertices)
        {
            Range::size_type size = std::numeric_limits<Range::size_type>::max();
            if (start + size > vertices)
                size = vertices - start;
            root.push_back(Range(i, start, size));
            start += size;
        }
    }
    return numSplats;
}

struct MakeGrid
{
    bool first;
    float low[3];
    float bboxMin[3];
    float bboxMax[3];

    MakeGrid() : first(true) {}
    void operator()(Range::scan_type scan, Range::index_type id, const Splat &splat);
};

void MakeGrid::operator()(Range::scan_type scan, Range::index_type id, const Splat &splat)
{
    (void) scan;
    (void) id;

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
}

} // namespace

void bucket(const boost::ptr_vector<FastPly::Reader> &files,
            const Grid &bbox,
            Range::index_type maxSplats,
            int maxCells,
            std::size_t maxSplit,
            const Processor &process)
{
    /* Create a root bucket will all splats in it */
    std::vector<Range> root;
    Range::index_type numSplats = makeRoot(files, root);

    BucketParameters params(files, bbox, process, maxSplats, maxCells, maxSplit);
    params.maxSplats = maxSplats;
    params.maxCells = maxCells;
    params.maxSplit = maxSplit;
    bucketRecurse(root.begin(), root.end(), numSplats, params);
}

Grid makeGrid(const boost::ptr_vector<FastPly::Reader> &files,
              float spacing)
{
    Timer timer;

    std::vector<Range> root;
    Range::index_type numSplats = makeRoot(files, root);
    if (numSplats == 0)
        throw std::length_error("Must be at least one splat");

    MakeGrid state;
    internal::forEachSplat(files, root.begin(), root.end(), boost::ref(state));

    int extents[3][2];
    for (unsigned int i = 0; i < 3; i++)
    {
        float l = (state.bboxMin[i] - state.low[i]) / spacing;
        float h = (state.bboxMax[i] - state.low[i]) / spacing;
        extents[i][0] = RoundDown::convert(l);
        extents[i][1] = RoundUp::convert(h);
    }

    Statistics::getStatistic<Statistics::Variable>("makeGrid.time").add(timer.getElapsed());
    return Grid(state.low, spacing,
                extents[0][0], extents[0][1], extents[1][0], extents[1][1], extents[2][0], extents[2][1]);
}

} // namespace Bucket
