/**
 * @file
 *
 * Implementations of template members of @ref bucket.h and @ref bucket_internal.h
 */

#ifndef BUCKET_IMPL_H
#define BUCKET_IMPL_H
#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <boost/multi_array.hpp>
#include "bucket.h"
#include "bucket_internal.h"
#include "statistics.h"
#include "misc.h"
#if HAVE_STXXL
# include <stxxl.h>
#endif

namespace Bucket
{

namespace internal
{

template<typename OutputIterator>
RangeCollector<OutputIterator>::RangeCollector(iterator_type out)
    : current(), out(out)
{
}

template<typename OutputIterator>
RangeCollector<OutputIterator>::~RangeCollector()
{
    flush();
}

template<typename OutputIterator>
OutputIterator RangeCollector<OutputIterator>::append(Range::scan_type scan, Range::index_type splat)
{
    if (!current.append(scan, splat))
    {
        *out++ = current;
        current = Range(scan, splat);
    }
    return out;
}

template<typename OutputIterator>
OutputIterator RangeCollector<OutputIterator>::flush()
{
    if (current.size > 0)
    {
        *out++ = current;
        current = Range();
    }
    return out;
}


/**
 * Implementation detail of @ref forEachCell. Do not call this directly.
 *
 * @param dims      See @ref forEachCell.
 * @param cell      Current cell to process recursively.
 * @param func      See @ref forEachCell.
 */
template<typename Func>
void forEachCell_r(const Cell::size_type dims[3], const Cell &cell, const Func &func)
{
    if (func(cell))
    {
        if (cell.getLevel() > 0)
        {
            for (unsigned int i = 0; i < 8; i++)
            {
                Cell child = cell.child(i);
                if (child.getLower()[0] < dims[0]
                    && child.getLower()[1] < dims[1]
                    && child.getLower()[2] < dims[2])
                    forEachCell_r(dims, child, func);
            }
        }
    }
}

template<typename Func>
void forEachCell(const Cell::size_type dims[3], Cell::size_type microSize, unsigned int levels, const Func &func)
{
    MLSGPU_ASSERT(levels >= 1U
                  && levels <= (unsigned int) std::numeric_limits<Cell::size_type>::digits, std::invalid_argument);
    int level = levels - 1;
    MLSGPU_ASSERT((dims[0] - 1) >> level < microSize, std::invalid_argument);
    MLSGPU_ASSERT((dims[1] - 1) >> level < microSize, std::invalid_argument);
    MLSGPU_ASSERT((dims[2] - 1) >> level < microSize, std::invalid_argument);

    Cell::size_type size = microSize << level;
    forEachCell_r(dims, Cell(0, 0, 0, size, size, size, level), func);
}

template<typename Func>
void forEachCell(const Grid &grid, Cell::size_type microSize, unsigned int levels, const Func &func)
{
    const Cell::size_type dims[3];
    for (int i = 0; i < 3; i++)
        dims[i] = grid.numCells(i);
    forEachCell(dims, microSize, levels, func);
}

template<typename CollectionSet, typename Func>
void forEachSplat(
    const CollectionSet &splats,
    RangeConstIterator first,
    RangeConstIterator last,
    const Func &func)
{
    for (RangeConstIterator it = first; it != last; ++it)
    {
        const Range &range = *it;
        assert(range.scan < splats.size());
        splats[range.scan].forEach(range.start, range.start + range.size,
                                   boost::bind(func, range.scan, _1, _2));
    }
}

template<typename Func>
class ForEachSplatCell
{
private:
    const Grid &grid;
    Cell::size_type dims[3];
    Cell::size_type microSize;
    unsigned int levels;
    const Func &func;

    class PerSplat
    {
    private:
        Range::scan_type scan;
        Range::index_type id;
        const Splat &splat;
        const Func &func;
        float lower[3];      ///< Splat lower bound converted to grid coordinates
        float upper[3];      ///< Splat upper bound converted to grid coordinates

    public:
        PerSplat(const Grid &grid, Range::scan_type scan, Range::index_type id, const Splat &splat, const Func &func)
            : scan(scan), id(id), splat(splat), func(func)
        {
            float lo[3], hi[3];
            for (int i = 0; i < 3; i++)
            {
                lo[i] = splat.position[i] - splat.radius;
                hi[i] = splat.position[i] + splat.radius;
            }
            grid.worldToVertex(lo, lower);
            grid.worldToVertex(hi, upper);
        }

        bool operator()(const Cell &cell) const
        {
            for (int i = 0; i < 3; i++)
                if (upper[i] < cell.getLower()[i] || lower[i] > cell.getUpper()[i])
                    return false;
            return func(scan, id, splat, cell);
        }
    };

public:
    typedef void result_type;

    ForEachSplatCell(const Grid &grid, Cell::size_type microSize, unsigned int levels, const Func &func)
        : grid(grid), microSize(microSize), levels(levels), func(func)
    {
        for (int i = 0; i < 3; i++)
            dims[i] = grid.numCells(i);
    }

    void operator()(Range::scan_type scan, Range::index_type id, const Splat &splat) const
    {
        PerSplat p(grid, scan, id, splat, func);
        forEachCell(dims, microSize, levels, p);
    }
};

template<typename CollectionSet, typename Func>
void forEachSplatCell(
    const CollectionSet &splats,
    RangeConstIterator first,
    RangeConstIterator last,
    const Grid &grid, Cell::size_type microSize, unsigned int levels,
    const Func &func)
{
    ForEachSplatCell<Func> f(grid, microSize, levels, func);
    forEachSplat(splats, first, last, f);
}

/// Contains static information used to process a cell.
struct BucketParameters
{
    Range::index_type maxSplats;        ///< Maximum splats permitted for processing
    unsigned int maxCells;              ///< Maximum cells along any dimension
    std::size_t maxSplit;               ///< Maximum fan-out for recursion

    BucketParameters(Range::index_type maxSplats,
                     unsigned int maxCells, std::size_t maxSplit)
        : maxSplats(maxSplats), maxCells(maxCells), maxSplit(maxSplit) {}
};

/**
 * Dynamic state that is updated as part of processing a cell.
 */
struct BucketState
{
    static const std::size_t BAD_BLOCK = std::size_t(-1);

    struct Child
    {
        Cell cell;
        std::vector<Range> ranges;
        RangeCollector<std::back_insert_iterator<std::vector<Range> > > collector;

        Child() : collector(std::back_inserter(ranges)) {}
        Child(const Cell &cell) : cell(cell), collector(std::back_inserter(ranges)) {}
        Child(const Child &c)
            : cell(c.cell), ranges(c.ranges), collector(std::back_inserter(ranges)) {}
        Child &operator=(const Child &c)
        {
            cell = c.cell;
            ranges = c.ranges;
            return *this;
        }
    };

    const BucketParameters &params;
    /// Grid covering just the region being processed
    const Grid &grid;
    /**
     * Size (in grid cells) of the region being processed.
     * This is just a cache of grid.numCells for ease of passing to @ref forEachCell.
     */
    Cell::size_type dims[3];
    /// Side length of a microblock
    Cell::size_type microSize;
    /// Number of levels in the octree of counters.
    int macroLevels;
    /**
     * Octree of splat counts. Each element of the vector is one level of the
     * octree.  Element zero contains the finest level, higher elements the
     * coarser levels.
     *
     * During the initial counting phase, each entry represents a delta to
     * be added to the sum of the children. Elements other than the leaves
     * will thus typically be negative. A correction pass applies the summation
     * up the tree.
     */
    std::vector<boost::multi_array<std::tr1::int64_t, 3> > cellCounts;

    /**
     * Index of the chosen cell for each leaf (BAD_BLOCK if empty).
     */
    boost::multi_array<std::size_t, 3> cellBlocks;

    /// Number of leaf cells skipped by @ref PickCells for being empty
    std::tr1::uint64_t skippedCells;

    /**
     * The blocks and ranges for the next level of the hierarchy.
     */
    std::vector<Child> children;

    /// Constructor
    BucketState(const BucketParameters &params, const Grid &grid,
                Cell::size_type microSize, int macroLevels);

    /**
     * Get the (inclusive) range of indices in the base level of @ref cellCounts
     * covered a splat.
     */
    void getSplatCells(const Splat &splat, Cell::size_type lo[3], Cell::size_type hi[3]);

    /// The number of splats that land in a given power-of-two cell
    std::tr1::int64_t getCellCount(const Cell &cell) const;

    void upsweepCounts();
};

/**
 * Function object for use with @ref Bucket::internal::forEachSplat that enters the splat
 * into all corresponding counters in the tree.
 */
class CountSplat
{
private:
    BucketState &state;

public:
    typedef void result_type;
    CountSplat(BucketState &state) : state(state) {};

    void operator()(Range::scan_type scan, Range::index_type id, const Splat &splat) const;
};

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
    bool operator()(const Cell &cell) const;
};

/**
 * Functor for @ref Bucket::internal::forEachSplat that places splat information into
 * bucket ranges.
 */
class BucketSplats
{
private:
    BucketState &state;

public:
    typedef void result_type;
    BucketSplats(BucketState &state) : state(state) {}

    void operator()(Range::scan_type scan, Range::index_type id, const Splat &splat) const;
};

Cell::size_type chooseMicroSize(
    const Cell::size_type dims[3], std::size_t maxSplit);

/**
 * Recursive implementation of @ref bucket.
 *
 * @param splats          Backing store of splats to process.
 * @param process         Function to call for each final bucket.
 * @param first,last      Range of ranges to process for this spatial region.
 * @param numSplats       Number of splats encoded into [@a first, @a last).
 * @param grid            Sub-grid on which the recursion is being done.
 * @param params          User parameters.
 * @param recursionDepth  Number of higher-level @ref bucketRecurse invocations on the stack.
 * @param totalRanges     Number of ranges held in memory across all levels of the recursion.
 */
template<typename CollectionSet>
void bucketRecurse(
    const CollectionSet &splats,
    RangeConstIterator first,
    RangeConstIterator last,
    Range::index_type numSplats,
    const Grid &grid,
    const BucketParameters &params,
    const typename ProcessorType<CollectionSet>::type &process,
    const Recursion &recursionState)
{
    Statistics::getStatistic<Statistics::Peak<unsigned int> >("bucket.depth.peak").set(recursionState.depth);
    Statistics::getStatistic<Statistics::Peak<Range::index_type> >("bucket.totalRanges.peak").set(recursionState.totalRanges);

    Cell::size_type dims[3];
    for (int i = 0; i < 3; i++)
        dims[i] = grid.numCells(i);
    Cell::size_type maxDim = std::max(std::max(dims[0], dims[1]), dims[2]);

    if (numSplats <= params.maxSplats && maxDim <= params.maxCells)
    {
        boost::unwrap_ref(process)(splats, numSplats, first, last, grid, recursionState);
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
        Cell::size_type microSize;
        if (maxDim > params.maxCells)
        {
            // number of maxCells-sized blocks
            Cell::size_type subDims[3];
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

        std::size_t numPicked;
        std::vector<BucketState::Child> savedChildren;
        std::vector<Range::index_type> savedNumSplats;
        std::tr1::uint64_t cellsDone = recursionState.cellsDone;
        /* Open a scope so that we can destroy the BucketState later */
        {
            BucketState state(params, grid, microSize, macroLevels);
            /* Create histogram */
            forEachSplat(splats, first, last, CountSplat(state));
            state.upsweepCounts();
            /* Select cells to bucket splats into */
            forEachCell(state.dims, state.microSize, state.macroLevels, PickCells(state));
            /* Do the bucketing. */
            forEachSplat(splats, first, last, BucketSplats(state));
            for (std::size_t i = 0; i < state.children.size(); i++)
                state.children[i].collector.flush();

            /* Bucketing is complete but we have a lot of memory allocated that we
             * don't need for the recursive step. Copy it outside this scope then
             * close the scope to destroy state.
             */
            numPicked = state.children.size();
            savedNumSplats.reserve(numPicked);
            for (std::size_t i = 0; i < numPicked; i++)
                savedNumSplats.push_back(state.getCellCount(state.children[i].cell));
            savedChildren.swap(state.children);

            /* Any cells we skipped are automatically finished */
            cellsDone += state.skippedCells;
        }

        /* Now recurse into the chosen cells */
        for (std::size_t i = 0; i < numPicked; i++)
        {
            const BucketState::Child &child = savedChildren[i];
            const Cell &cell = child.cell;
            const Cell::size_type *lower = cell.getLower();
            Cell::size_type upper[3];
            std::copy(cell.getUpper(), cell.getUpper() + 3, upper);
            // Clip the cell to the grid
            for (int j = 0; j < 3; j++)
            {
                upper[j] = std::min(upper[j], dims[j]);
                assert(lower[j] < upper[j]);
            }
            Grid childGrid = grid.subGrid(
                lower[0], upper[0], lower[1], upper[1], lower[2], upper[2]);
            Recursion childRecursion = recursionState;
            childRecursion.depth++;
            childRecursion.totalRanges += child.ranges.size();
            childRecursion.cellsDone = cellsDone;
            bucketRecurse(splats,
                          child.ranges.begin(),
                          child.ranges.end(),
                          savedNumSplats[i],
                          childGrid,
                          params,
                          process,
                          childRecursion);
            cellsDone += childGrid.numCells();
        }
    }
}

/**
 * Function object for accumulating a bounding box of splats.
 *
 * The interface is designed to be used with @c stxxl::stream::transform so
 * that it passes through its inputs unchanged, or with @ref
 * Collection::forEach.
 */
struct MakeGrid
{
    typedef Splat value_type;

    bool first;
    float low[3];
    float bboxMin[3];
    float bboxMax[3];

    MakeGrid() : first(true) {}
    const Splat &operator()(const Splat &splat);
    void operator()(unsigned int index, const Splat &splat);

    /// Generate the grid from the accumulated data
    Grid makeGrid(float spacing) const;
};

} // namespace internal

template<typename CollectionSet>
void bucket(const CollectionSet &splats,
            const Grid &bbox,
            Range::index_type maxSplats,
            int maxCells,
            std::size_t maxSplit,
            const typename ProcessorType<CollectionSet>::type &process,
            const Recursion &recursionState)
{
    Range::index_type numSplats = 0;
    std::vector<Range> root;
    root.reserve(splats.size());
    for (typename CollectionSet::size_type i = 0; i < splats.size(); i++)
    {
        root.push_back(Range(i, 0, splats[i].size()));
        numSplats += splats[i].size();
    }

    internal::BucketParameters params(maxSplats, maxCells, maxSplit);
    Recursion childRecursion = recursionState;
    childRecursion.depth++;
    childRecursion.totalRanges += root.size();
    internal::bucketRecurse(splats, root.begin(), root.end(), numSplats, bbox, params, process, childRecursion);
}

#if HAVE_STXXL
template<typename CollectionSet>
void loadSplats(const CollectionSet &files,
                float spacing,
                bool sort,
                StxxlVectorCollection<Splat>::vector_type &splats,
                Grid &grid)
{
    typedef typename StxxlVectorCollection<Splat>::vector_type SplatVector;
    Statistics::Timer timer("loadSplats.time");

    Range::index_type numSplats = 0;
    for (typename CollectionSet::const_iterator i = files.begin(); i != files.end(); ++i)
    {
        numSplats += i->size();
    }

    const unsigned int block_size = SplatVector::block_size;
    typedef CollectionStream<typename CollectionSet::const_iterator> files_type;
    typedef stxxl::stream::transform<internal::MakeGrid, files_type> peek_type;
    typedef CompareSplatsMorton comparator_type;
    typedef stxxl::stream::sort<peek_type, comparator_type, block_size> sort_type;

    internal::MakeGrid state;

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

    grid = state.makeGrid(spacing);
}
#endif // HAVE_STXXL

template<typename CollectionSet>
void makeGrid(const CollectionSet &files,
              float spacing,
              Grid &grid)
{
    Statistics::Timer timer("makeGrid.time");
    internal::MakeGrid state;

    for (typename CollectionSet::const_iterator i = files.begin(); i != files.end(); ++i)
    {
        i->forEach(0, i->size(), boost::ref(state));
    }
    grid = state.makeGrid(spacing);
}

} // namespace Bucket

#endif /* !BUCKET_IMPL_H */
