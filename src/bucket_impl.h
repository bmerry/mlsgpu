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
#include <boost/array.hpp>
#include <boost/multi_array.hpp>
#include <boost/smart_ptr/scoped_ptr.hpp>
#include <boost/numeric/conversion/converter.hpp>
#include <ostream>
#include <limits>
#include "bucket.h"
#include "bucket_internal.h"
#include "statistics.h"
#include "misc.h"
#include "progress.h"
#include "logging.h"
#include <stxxl.h>

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
OutputIterator RangeCollector<OutputIterator>::append(
    Range::scan_type scan, Range::index_type first, Range::index_type last)
{
    if (first >= last)
        return out; // some odd corner cases can develop otherwise

    if (current.size > 0)
    {
        if (current.scan == scan && current.start + current.size >= first
            && last >= current.start)
        {
            // Ranges overlap or adjoin
            first = std::min(first, current.start);
            last = std::max(last, current.start + current.size);
        }
        else
            flush();
    }
    current.scan = scan;
    while (true)
    {
        if (last - first <= std::numeric_limits<Range::size_type>::max())
        {
            current.start = first;
            current.size = last - first;
            break;
        }
        else
        {
            current.start = first;
            current.size = std::numeric_limits<Range::size_type>::max();
            first += current.size;
            *out++ = current;
        }
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
 * Implementation detail of @ref forEachNode. Do not call this directly.
 *
 * @param dims      See @ref forEachNode.
 * @param node      Current node to process recursively.
 * @param func      See @ref forEachNode.
 */
template<typename Func>
void forEachNode_r(const Node::size_type dims[3], const Node &node, const Func &func)
{
    if (func(node))
    {
        if (node.getLevel() > 0)
        {
            for (unsigned int i = 0; i < 8; i++)
            {
                Node child = node.child(i);
                const boost::array<Node::size_type, 3> &coords = child.getCoords();
                for (unsigned int j = 0; j < 3; j++)
                {
                    if ((coords[j] << child.getLevel()) >= dims[j])
                        goto skip;
                }
                forEachNode_r(dims, child, func);
skip:;
            }
        }
    }
}

template<typename Func>
void forEachNode(const Node::size_type dims[3], unsigned int levels, const Func &func)
{
    MLSGPU_ASSERT(levels >= 1U
                  && levels <= (unsigned int) std::numeric_limits<Node::size_type>::digits, std::invalid_argument);
    int level = levels - 1;
    MLSGPU_ASSERT(dims[0] <= Node::size_type(1) << level, std::invalid_argument);
    MLSGPU_ASSERT(dims[1] <= Node::size_type(1) << level, std::invalid_argument);
    MLSGPU_ASSERT(dims[2] <= Node::size_type(1) << level, std::invalid_argument);

    forEachNode_r(dims, Node(0, 0, 0, level), func);
}

/// Contains static information used to process a region.
struct BucketParameters
{
    Range::index_type maxSplats;        ///< Maximum splats permitted for processing
    Grid::size_type maxCells;           ///< Maximum cells along any dimension
    bool maxCellsHint;                  ///< If true, @ref maxCells is merely a microblock size hint
    std::size_t maxSplit;               ///< Maximum fan-out for recursion
    ProgressDisplay *progress;          ///< Progress display to update for empty cells

    BucketParameters(Range::index_type maxSplats,
                     Grid::size_type maxCells, bool maxCellsHint, std::size_t maxSplit,
                     ProgressDisplay *progress)
        : maxSplats(maxSplats), maxCells(maxCells), maxCellsHint(maxCellsHint),
        maxSplit(maxSplit), progress(progress) {}
};

/**
 * Dynamic state that is updated as part of processing a region.
 */
struct BucketState
{
    static const std::size_t BAD_REGION = std::size_t(-1);

    /**
     * A child block. The copy constructor and assignment operator are
     * provided so that these can be stored in an STL vector. Copying
     * the ranges to grow the vector would be expensive, but in actual
     * use the containing vector is sized first before splats are
     * inserted.
     */
    struct Subregion
    {
        Node node;
        Range::size_type splatsSeen;   ///< Number of splats added to ranges
        Range::size_type numSplats;    ///< Expected number of splats
        std::vector<Range> ranges;
        RangeCollector<std::back_insert_iterator<std::vector<Range> > > collector;

        Subregion() : collector(std::back_inserter(ranges)) {}
        Subregion(const Node &node, Range::size_type numSplats)
            : node(node), splatsSeen(0), numSplats(numSplats), collector(std::back_inserter(ranges)) {}
        Subregion(const Subregion &c)
            : node(c.node), splatsSeen(c.splatsSeen), numSplats(c.numSplats),
            ranges(c.ranges), collector(std::back_inserter(ranges)) {}
        Subregion &operator=(const Subregion &c)
        {
            node = c.node;
            splatsSeen = c.splatsSeen;
            numSplats = c.numSplats;
            ranges = c.ranges;
            return *this;
        }
    };

    const BucketParameters &params;
    /// Grid covering the region being processed
    const Grid &grid;
    /// Size in microblocks of the region being processed.
    Grid::size_type dims[3];

    /// Side length of a microblock
    Grid::size_type microSize;
    /// Number of levels in the octree of counters.
    int macroLevels;
    /**
     * Octree of splat counts. Each element of the vector is one level of the
     * octree.  Element zero contains the finest level, higher elements the
     * coarser levels.
     *
     * During the initial counting phase, each entry represents a delta to
     * be added to the sum of the children. Elements other than the leaves
     * will thus typically be negative. @ref upsweepCounts applies the
     * summation up the tree.
     */
    std::vector<boost::multi_array<std::tr1::int64_t, 3> > nodeCounts;

    /**
     * Index of the chosen subregion for each leaf (BAD_REGION if empty).
     */
    boost::multi_array<std::size_t, 3> microRegions;

    /**
     * The nodes and ranges for the next level of the hierarchy.
     */
    std::vector<Subregion> subregions;

    /// Constructor
    BucketState(const BucketParameters &params, const Grid &grid,
                Grid::size_type microSize, int macroLevels);

    /**
     * The number of splats that land in a given node.
     * @see @ref upsweepCounts.
     */
    std::tr1::int64_t getNodeCount(const Node &node) const;

    /**
     * Convert @ref nodeCounts from a delta encoding to plain counts.
     * This should be called after all calls to @ref CountSplat are complete,
     * and before calling @ref getNodeCount.
     */
    void upsweepCounts();

    /**
     * Clamp a range of microblocks to the coverage of nodeCounts, checking for
     * the case of no overlap. If there is no overlap, the output values are
     * undefined.
     *
     * @param[in]  lower, upper  Inclusive range of microblock coordinates.
     * @param[out] lo, hi        Microblock coordinates clamped to the microblock coverage.
     * @return @c true if the range overlapped the coverage, otherwise @c false.
     */
    bool clamp(const boost::array<Grid::difference_type, 3> &lower,
               const boost::array<Grid::difference_type, 3> &upper,
               boost::array<Node::size_type, 3> &lo,
               boost::array<Node::size_type, 3> &hi);
};

/**
 * Function object for use with @ref SplatSet::SimpleSet::forEachRange that enters the splat
 * into all corresponding counters in the tree.
 */
class CountSplat
{
private:
    BucketState &state;

public:
    typedef void result_type;
    CountSplat(BucketState &state) : state(state) {};

    void operator()(Range::scan_type scan, Range::index_type first, Range::index_type last,
                    const boost::array<Grid::difference_type, 3> &lower,
                    const boost::array<Grid::difference_type, 3> &upper) const;
};

/**
 * Functor for @ref Bucket::internal::forEachNode that chooses which
 * nodes to turn into regions. A node is chosen if it contains few enough
 * splats and is small enough, or if it is a microblock. Otherwise it is
 * split.
 */
class PickNodes
{
private:
    BucketState &state;

public:
    PickNodes(BucketState &state) : state(state) {}
    bool operator()(const Node &node) const;
};

/**
 * Functor for @ref SplatSet::SimpleSet::forEachRange that places splat information into
 * bucket ranges.
 */
template<typename CollectionSet>
class BucketSplat
{
private:
    BucketState &state;
    const CollectionSet &splats;
    const typename ProcessorType<CollectionSet>::type &process;
    const Recursion &recursionState;

public:
    typedef void result_type;
    BucketSplat(
        BucketState &state,
        const CollectionSet &splats,
        const typename ProcessorType<CollectionSet>::type &process,
        const Recursion &recursionState)
        : state(state), splats(splats), process(process), recursionState(recursionState) {}

    void operator()(Range::scan_type scan,
                    Range::index_type first, Range::index_type last,
                    const boost::array<Grid::difference_type, 3> &lower,
                    const boost::array<Grid::difference_type, 3> &upper) const;
};

template<typename CollectionSet>
void BucketSplat<CollectionSet>::operator()(
    Range::scan_type scan,
    Range::index_type first, Range::index_type last,
    const boost::array<Grid::difference_type, 3> &lower,
    const boost::array<Grid::difference_type, 3> &upper) const
{
    Range::index_type count = last - first;
    boost::array<Node::size_type, 3> lo, hi;
    if (!state.clamp(lower, upper, lo, hi))
        return;

    for (Node::size_type x = lo[0]; x <= hi[0]; x++)
        for (Node::size_type y = lo[1]; y <= hi[1]; y++)
            for (Node::size_type z = lo[2]; z <= hi[2]; z++)
            {
                std::size_t regionId = state.microRegions[x][y][z];
                assert(regionId < state.subregions.size());
                BucketState::Subregion &region = state.subregions[regionId];

                /* Only add once per node */
                const Node::size_type nodeSize = region.node.size();
                const Node::size_type mask = nodeSize - 1;
                if ((x == lo[0] || ((x & mask) == 0))
                    && (y == lo[1] || (y & mask) == 0)
                    && (z == lo[2] || (z & mask) == 0))
                {
                    region.collector.append(scan, first, last);
                    region.splatsSeen += count;
                    if (region.splatsSeen == region.numSplats)
                    {
                        region.collector.flush();

                        // Clip the region to the grid
                        Grid::size_type lower[3], upper[3];
                        region.node.toCells(state.microSize, lower, upper, state.grid);
                        Grid childGrid = state.grid.subGrid(
                            lower[0], upper[0], lower[1], upper[1], lower[2], upper[2]);

                        Recursion childRecursion = recursionState;
                        childRecursion.depth++;
                        childRecursion.totalRanges += region.ranges.size();
                        bucketRecurse(splats,
                                      region.ranges.begin(),
                                      region.ranges.end(),
                                      region.numSplats,
                                      childGrid,
                                      state.params,
                                      process,
                                      childRecursion);
                    }
                }
            }
}

/**
 * Determine the appropriate size for the microblocks.
 * This ignores @a maxCells. Instead, @a dims could be in either units of cells or
 * in units of @a maxCells, in which case we are using how many @a maxCells sized
 * blocks form each microblock.
 */
Grid::size_type chooseMicroSize(
    const Grid::size_type dims[3], std::size_t maxSplit);

/**
 * Recursive implementation of @ref bucket.
 *
 * @param splats          Backing store of splats to process.
 * @param process         Function to call for each final bucket.
 * @param first,last      Range of ranges to process for this spatial region.
 * @param numSplats       Number of splats encoded into [@a first, @a last).
 * @param grid            Sub-grid on which the recursion is being done.
 * @param params          User parameters.
 * @param recursionState  Statistics about what is already held on the stack.
 */
template<typename CollectionSet>
void bucketRecurse(
    const CollectionSet &splats,
    RangeConstIterator first,
    RangeConstIterator last,
    typename CollectionSet::index_type numSplats,
    const Grid &grid,
    const BucketParameters &params,
    const typename ProcessorType<CollectionSet>::type &process,
    const Recursion &recursionState)
{
    Statistics::getStatistic<Statistics::Peak<unsigned int> >("bucket.depth.peak").set(recursionState.depth);
    Statistics::getStatistic<Statistics::Peak<Range::index_type> >("bucket.totalRanges.peak").set(recursionState.totalRanges);

    Grid::size_type cellDims[3];
    for (int i = 0; i < 3; i++)
        cellDims[i] = grid.numCells(i);
    Grid::size_type maxCellDim = std::max(std::max(cellDims[0], cellDims[1]), cellDims[2]);

    if (numSplats <= params.maxSplats &&
        (maxCellDim <= params.maxCells || params.maxCellsHint))
    {
        boost::unwrap_ref(process)(splats, numSplats, first, last, grid, recursionState);
    }
    else if (maxCellDim == 1)
    {
        throw DensityError(numSplats); // can't subdivide a 1x1x1 cell
    }
    else
    {
        /* Pick a microblock size such that we don't exceed maxSplit
         * microblocks. If the current region is bigger than maxCells
         * in any direction we use a power of two times maxCells, otherwise
         * we use a power of 2.
         *
         * TODO: no need for it to be a power of 2?
         */
        Grid::size_type microSize;
        if (maxCellDim > params.maxCells)
        {
            // number of maxCells-sized blocks
            Grid::size_type subDims[3];
            for (unsigned int i = 0; i < 3; i++)
                subDims[i] = divUp(cellDims[i], params.maxCells);
            microSize = params.maxCells * chooseMicroSize(subDims, params.maxSplit);
        }
        else
            microSize = chooseMicroSize(cellDims, params.maxSplit);

        /* Levels in octree structure */
        int macroLevels = 1;
        while (microSize << (macroLevels - 1) < maxCellDim)
            macroLevels++;

        BucketState state(params, grid, microSize, macroLevels);
        /* Create histogram */
        splats.forEachRange(first, last, grid, microSize, CountSplat(state));
        state.upsweepCounts();
        /* Select cells to bucket splats into */
        forEachNode(state.dims, state.macroLevels, PickNodes(state));
        /* Do the bucketing. */
        splats.forEachRange(first, last,
                            grid, microSize,
                            BucketSplat<CollectionSet>(state, splats, process, recursionState));

        /* Check that all regions were completed */
        // TODO: move sublaunches back out of BucketSplat
#ifndef NDEBUG
        for (std::size_t i = 0; i < state.subregions.size(); i++)
            assert(state.subregions[i].splatsSeen == state.subregions[i].numSplats);
#endif
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

    Range::index_type nonFinite;

    MakeGrid() : first(true), nonFinite(0) {}
    const Splat &operator()(const Splat &splat);
    void operator()(unsigned int index, const Splat &splat);

    /// Generate the grid from the accumulated data
    Grid makeGrid(float spacing) const;
};

} // namespace internal

template<typename CollectionSet>
void bucket(const CollectionSet &splats,
            const Grid &region,
            typename CollectionSet::index_type maxSplats,
            Grid::size_type maxCells,
            bool maxCellsHint,
            std::size_t maxSplit,
            const typename ProcessorType<CollectionSet>::type &process,
            ProgressDisplay *progress,
            const Recursion &recursionState)
{
    typename CollectionSet::index_type numSplats = 0;
    std::vector<Range> root;
    root.reserve(splats.getSplats().size());
    for (typename CollectionSet::scan_type i = 0; i < splats.getSplats().size(); i++)
    {
        const typename CollectionSet::index_type size = splats.getSplats()[i].size();
        if (size > 0)
        {
            root.push_back(Range(i, 0, size));
            numSplats += size;
        }
    }

    internal::BucketParameters params(maxSplats, maxCells, maxCellsHint, maxSplit, progress);
    Recursion childRecursion = recursionState;
    childRecursion.depth++;
    childRecursion.totalRanges += root.size();
    internal::bucketRecurse(splats, root.begin(), root.end(), numSplats, region, params, process, childRecursion);
}

} // namespace Bucket

#endif /* !BUCKET_IMPL_H */
