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
                if (child.getLower()[0] < dims[0]
                    && child.getLower()[1] < dims[1]
                    && child.getLower()[2] < dims[2])
                    forEachNode_r(dims, child, func);
            }
        }
    }
}

template<typename Func>
void forEachNode(const Node::size_type dims[3], Node::size_type microSize, unsigned int levels, const Func &func)
{
    MLSGPU_ASSERT(levels >= 1U
                  && levels <= (unsigned int) std::numeric_limits<Node::size_type>::digits, std::invalid_argument);
    int level = levels - 1;
    MLSGPU_ASSERT((dims[0] - 1) >> level < microSize, std::invalid_argument);
    MLSGPU_ASSERT((dims[1] - 1) >> level < microSize, std::invalid_argument);
    MLSGPU_ASSERT((dims[2] - 1) >> level < microSize, std::invalid_argument);

    Node::size_type size = microSize << level;
    forEachNode_r(dims, Node(0, 0, 0, size, size, size, level), func);
}

template<typename Func>
void forEachNode(const Grid &grid, Node::size_type microSize, unsigned int levels, const Func &func)
{
    const Node::size_type dims[3];
    for (int i = 0; i < 3; i++)
        dims[i] = grid.numCells(i);
    forEachNode(dims, microSize, levels, func);
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
    unsigned int maxCells;              ///< Maximum cells along any dimension
    std::size_t maxSplit;               ///< Maximum fan-out for recursion

    BucketParameters(Range::index_type maxSplats,
                     unsigned int maxCells, std::size_t maxSplit)
        : maxSplats(maxSplats), maxCells(maxCells), maxSplit(maxSplit) {}
};

/**
 * Dynamic state that is updated as part of processing a region.
 */
struct BucketState
{
    static const std::size_t BAD_BLOCK = std::size_t(-1);

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
        std::vector<Range> ranges;
        RangeCollector<std::back_insert_iterator<std::vector<Range> > > collector;

        Subregion() : collector(std::back_inserter(ranges)) {}
        Subregion(const Node &node) : node(node), collector(std::back_inserter(ranges)) {}
        Subregion(const Subregion &c)
            : node(c.node), ranges(c.ranges), collector(std::back_inserter(ranges)) {}
        Subregion &operator=(const Subregion &c)
        {
            node = c.node;
            ranges = c.ranges;
            return *this;
        }
    };

    const BucketParameters &params;
    /// Grid covering the region being processed
    const Grid &grid;
    /**
     * Size (in cells) of the region being processed.
     * This is just a cache of grid.numCells for ease of passing to @ref forEachNode.
     */
    Node::size_type dims[3];
    /// Side length of a microblock
    Node::size_type microSize;
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
     * Index of the chosen subregion for each leaf (BAD_BLOCK if empty).
     */
    boost::multi_array<std::size_t, 3> microRegions;

    /// Number of cells skipped by @ref PickNodes for being empty
    std::tr1::uint64_t skippedCells;

    /**
     * The nodes and ranges for the next level of the hierarchy.
     */
    std::vector<Subregion> subregions;

    /// Constructor
    BucketState(const BucketParameters &params, const Grid &grid,
                Node::size_type microSize, int macroLevels);

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
class BucketSplats
{
private:
    BucketState &state;

public:
    typedef void result_type;
    BucketSplats(BucketState &state) : state(state) {}

    void operator()(Range::scan_type scan, Range::index_type id, const Splat &splat) const;
};

/**
 * Determine the appropriate size for the microblocks.
 * This ignores @a maxCells. Instead, @a dims could be in either units of cells or
 * in units of @a maxCells, in which case we are using how many @a maxCells sized
 * blocks form each microblock.
 */
Node::size_type chooseMicroSize(
    const Node::size_type dims[3], std::size_t maxSplit);

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

    Node::size_type dims[3];
    for (int i = 0; i < 3; i++)
        dims[i] = grid.numCells(i);
    Node::size_type maxDim = std::max(std::max(dims[0], dims[1]), dims[2]);

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
         * microblocks. If the current region is bigger than maxCells
         * in any direction we use a power of two times maxCells, otherwise
         * we use a power of 2.
         *
         * TODO: no need for it to be a power of 2?
         */
        Node::size_type microSize;
        if (maxDim > params.maxCells)
        {
            // number of maxCells-sized blocks
            Node::size_type subDims[3];
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

        std::size_t numRegions;
        std::vector<BucketState::Subregion> savedRegions;
        std::vector<Range::index_type> savedNumSplats;
        std::tr1::uint64_t cellsDone = recursionState.cellsDone;
        /* Open a scope so that we can destroy the BucketState later */
        {
            BucketState state(params, grid, microSize, macroLevels);
            /* Create histogram */
            forEachSplat(splats, first, last, CountSplat(state));
            state.upsweepCounts();
            /* Select cells to bucket splats into */
            forEachNode(state.dims, state.microSize, state.macroLevels, PickNodes(state));
            /* Do the bucketing. */
            forEachSplat(splats, first, last, BucketSplats(state));
            for (std::size_t i = 0; i < state.subregions.size(); i++)
                state.subregions[i].collector.flush();

            /* Bucketing is complete but we have a lot of memory allocated that we
             * don't need for the recursive step. Copy it outside this scope then
             * close the scope to destroy state.
             */
            numRegions = state.subregions.size();
            savedNumSplats.reserve(numRegions);
            for (std::size_t i = 0; i < numRegions; i++)
                savedNumSplats.push_back(state.getNodeCount(state.subregions[i].node));
            savedRegions.swap(state.subregions);

            /* Any cells we skipped are automatically finished */
            cellsDone += state.skippedCells;
        }

        /* Now recurse into the chosen regions */
        for (std::size_t i = 0; i < numRegions; i++)
        {
            const BucketState::Subregion &region = savedRegions[i];
            const Node &node = region.node;
            const Node::size_type *lower = node.getLower();
            Node::size_type upper[3];
            std::copy(node.getUpper(), node.getUpper() + 3, upper);
            // Clip the region to the grid
            for (int j = 0; j < 3; j++)
            {
                upper[j] = std::min(upper[j], dims[j]);
                assert(lower[j] < upper[j]);
            }
            Grid childGrid = grid.subGrid(
                lower[0], upper[0], lower[1], upper[1], lower[2], upper[2]);
            Recursion childRecursion = recursionState;
            childRecursion.depth++;
            childRecursion.totalRanges += region.ranges.size();
            childRecursion.cellsDone = cellsDone;
            bucketRecurse(splats,
                          region.ranges.begin(),
                          region.ranges.end(),
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
