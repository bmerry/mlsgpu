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
#include "progress.h"
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

/// Contains static information used to process a region.
struct BucketParameters
{
    Range::index_type maxSplats;        ///< Maximum splats permitted for processing
    Grid::size_type maxCells;           ///< Maximum cells along any dimension
    std::size_t maxSplit;               ///< Maximum fan-out for recursion
    ProgressDisplay *progress;          ///< Progress display to update for empty cells

    BucketParameters(Range::index_type maxSplats,
                     Grid::size_type maxCells, std::size_t maxSplit,
                     ProgressDisplay *progress)
        : maxSplats(maxSplats), maxCells(maxCells), maxSplit(maxSplit), progress(progress) {}
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
     * Get the (inclusive) range of indices in the base level of
     * @ref nodeCounts covered by the bounding box of a splat.
     *
     * @param      splat        Splat to query
     * @param[out] lo           Indices of first microblock covered by @a splat
     * @param[out] hi           Indices of last microblock covered by @a splat.
     */
    void getSplatMicro(const Splat &splat, Node::size_type lo[3], Node::size_type hi[3]);

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
 * Functor for @ref Bucket::internal::forEachSplat that places splat information into
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

    void operator()(Range::scan_type scan, Range::index_type id, const Splat &splat) const;
};

template<typename CollectionSet>
void BucketSplat<CollectionSet>::operator()(Range::scan_type scan, Range::index_type id,
                                            const Splat &splat) const
{
    Node::size_type lo[3], hi[3];
    state.getSplatMicro(splat, lo, hi);
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
                    region.collector.append(scan, id);
                    region.splatsSeen++;
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
    Range::index_type numSplats,
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

    if (numSplats <= params.maxSplats && maxCellDim <= params.maxCells)
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
        forEachSplat(splats, first, last, CountSplat(state));
        state.upsweepCounts();
        /* Select cells to bucket splats into */
        forEachNode(state.dims, state.macroLevels, PickNodes(state));
        /* Do the bucketing. */
        forEachSplat(splats, first, last, BucketSplat<CollectionSet>(state, splats, process, recursionState));

        /* Check that all regions were completed */
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
            const Grid &bbox,
            Range::index_type maxSplats,
            Grid::size_type maxCells,
            std::size_t maxSplit,
            const typename ProcessorType<CollectionSet>::type &process,
            ProgressDisplay *progress,
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

    internal::BucketParameters params(maxSplats, maxCells, maxSplit, progress);
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
