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
    std::tr1::uint64_t maxSplats;       ///< Maximum splats permitted for processing
    Grid::size_type maxCells;           ///< Maximum cells along any dimension
    bool maxCellsHint;                  ///< If true, @ref maxCells is merely a microblock size hint
    std::size_t maxSplit;               ///< Maximum fan-out for recursion
    ProgressDisplay *progress;          ///< Progress display to update for empty cells

    BucketParameters(std::tr1::uint64_t maxSplats,
                     Grid::size_type maxCells, bool maxCellsHint, std::size_t maxSplit,
                     ProgressDisplay *progress)
        : maxSplats(maxSplats), maxCells(maxCells), maxCellsHint(maxCellsHint),
        maxSplit(maxSplit), progress(progress) {}
};

/**
 * Dynamic state that is updated as part of processing a region.
 */
class BucketState
{
public:
    /// Constant parameters for the bucketing process
    const BucketParameters &params;
    /// Grid covering the region being processed
    const Grid grid;
    /// Side length of a microblock
    const Grid::size_type microSize;
    /// Number of levels in the octree of counters.
    const int macroLevels;

    /// Constructor
    BucketState(const BucketParameters &params, const Grid &grid,
                Grid::size_type microSize, int macroLevels);

    /// Enters a blob into all corresponding counters in the tree.
    void countSplats(const SplatSet::BlobInfo &blob);

    /**
     * Convert @ref nodeCounts from a delta encoding to plain counts.
     * This should be called after all calls to @ref CountSplat are complete,
     * and before calling @ref getNodeCount.
     */
    void upsweepCounts();

    /// Places splat information into bucket ranges.
    template<typename Splats>
    void bucketSplats(const SplatSet::BlobInfo &blob,
                      const Splats &splats,
                      const typename ProcessorType<Splats>::type &process,
                      const Recursion &recursionState);

    /**
     * The number of splats that land in a given node.
     * @see @ref upsweepCounts.
     */
    std::tr1::int64_t getNodeCount(const Node &node) const;

    /// Size in microblocks of the region being processed.
    const Grid::size_type *getDims() const { return dims; }

private:
    friend class PickNodes;

    static const std::size_t BAD_REGION = std::size_t(-1);

    /**
     * A child region. It is stored in a vector, so needs a valid copy
     * constructor.  Copying the ranges to grow the vector would be expensive,
     * but in actual use the containing vector is sized first before blobs are
     * inserted.
     */
    struct Subregion
    {
        Node node;
        std::tr1::uint64_t numSplats;  ///< Expected number of splats
        SplatSet::SubsetBase subset;   ///< Just the blob ranges - later put in a full subset object

        Subregion() {}
        Subregion(const Node &node, std::tr1::uint64_t numSplats)
            : node(node), numSplats(numSplats) {}
    };

    /// Size in microblocks of the region being processed.
    Grid::size_type dims[3];

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
 * Function object that places splat information into bucket ranges.
 */
template<typename Splats>
class BucketSplat
{
private:
    BucketState &state;
    const Splats &splats;
    const typename ProcessorType<Splats>::type &process;
    const Recursion &recursionState;

public:
    typedef void result_type;
    BucketSplat(
        BucketState &state,
        const Splats &splats,
        const typename ProcessorType<Splats>::type &process,
        const Recursion &recursionState)
        : state(state), splats(splats), process(process), recursionState(recursionState) {}

    void operator()(const SplatSet::BlobInfo &blob) const;
};

template<typename Splats>
void BucketState::bucketSplats(const SplatSet::BlobInfo &blob,
                               const Splats &splats,
                               const typename ProcessorType<Splats>::type &process,
                               const Recursion &recursionState)
{
    boost::array<Node::size_type, 3> lo, hi;
    if (!clamp(blob.lower, blob.upper, lo, hi))
        return;

    for (Node::size_type x = lo[0]; x <= hi[0]; x++)
        for (Node::size_type y = lo[1]; y <= hi[1]; y++)
            for (Node::size_type z = lo[2]; z <= hi[2]; z++)
            {
                std::size_t regionId = microRegions[x][y][z];
                assert(regionId < subregions.size());
                BucketState::Subregion &region = subregions[regionId];

                /* Only add once per node */
                const Node::size_type nodeSize = region.node.size();
                const Node::size_type mask = nodeSize - 1;
                if ((x == lo[0] || ((x & mask) == 0))
                    && (y == lo[1] || (y & mask) == 0)
                    && (z == lo[2] || (z & mask) == 0))
                {
                    region.subset.addBlob(blob);
                    if (region.subset.numSplats() == region.numSplats)
                    {
                        // Clip the region to the grid
                        Grid::size_type lower[3], upper[3];
                        region.node.toCells(microSize, lower, upper, grid);
                        Grid childGrid = grid.subGrid(
                            lower[0], upper[0], lower[1], upper[1], lower[2], upper[2]);

                        Recursion childRecursion = recursionState;
                        childRecursion.depth++;
                        childRecursion.totalRanges += region.subset.numRanges();

                        typename SplatSet::Traits<Splats>::subset_type subset(splats);
                        subset.swap(region.subset);
                        bucketRecurse(subset,
                                      childGrid,
                                      params,
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

template<typename Splats>
bool bucketCallback(const Splats &, const Grid &,
                    const typename ProcessorType<Splats>::type &,
                    const Recursion &,
                    boost::false_type)
{
    return false;
}

template<typename Splats>
bool bucketCallback(const Splats &splats, const Grid &grid,
                    const typename ProcessorType<Splats>::type &process,
                    const Recursion &recursionState,
                    boost::true_type)
{
    process(splats, grid, recursionState);
    return true;
}

/**
 * Recursive implementation of @ref bucket.
 *
 * @param splats          Backing store of splats to process.
 * @param process         Function to call for each final bucket.
 * @param grid            Sub-grid on which the recursion is being done.
 * @param params          User parameters.
 * @param recursionState  Statistics about what is already held on the stack.
 */
template<typename Splats>
void bucketRecurse(
    const Splats &splats,
    const Grid &grid,
    const BucketParameters &params,
    const typename ProcessorType<Splats>::type &process,
    const Recursion &recursionState)
{
    Statistics::getStatistic<Statistics::Peak<unsigned int> >("bucket.depth.peak").set(recursionState.depth);
    Statistics::getStatistic<Statistics::Peak<std::size_t> >("bucket.totalRanges.peak").set(recursionState.totalRanges);

    Grid::size_type cellDims[3];
    for (int i = 0; i < 3; i++)
        cellDims[i] = grid.numCells(i);
    Grid::size_type maxCellDim = std::max(std::max(cellDims[0], cellDims[1]), cellDims[2]);

    if (splats.maxSplats() <= params.maxSplats
        && (maxCellDim <= params.maxCells || params.maxCellsHint)
        && bucketCallback(splats, grid, process, recursionState,
                          typename SplatSet::Traits<Splats>::is_subset()))
    {
        // The bucketCallback in the if statement did the work
    }
    else if (maxCellDim == 1)
    {
        throw DensityError(splats.maxSplats()); // can't subdivide a 1x1x1 cell
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
        boost::scoped_ptr<SplatSet::BlobStream> blobs(splats.makeBlobStream(grid, microSize));
        while (!blobs->empty())
        {
            state.countSplats(**blobs);
            ++*blobs;
        }
        blobs.reset();

        state.upsweepCounts();
        /* Select cells to bucket splats into */
        forEachNode(state.getDims(), state.macroLevels, PickNodes(state));

        /* Do the bucketing. */
        blobs.reset(splats.makeBlobStream(grid, microSize));
        while (!blobs->empty())
        {
            state.bucketSplats(**blobs, splats, process, recursionState);
            ++*blobs;
        }
    }
}

} // namespace internal

template<typename Splats>
void bucket(const Splats &splats,
            const Grid &region,
            std::tr1::uint64_t maxSplats,
            Grid::size_type maxCells,
            bool maxCellsHint,
            std::size_t maxSplit,
            const typename ProcessorType<Splats>::type &process,
            ProgressDisplay *progress,
            const Recursion &recursionState)
{
    internal::BucketParameters params(maxSplats, maxCells, maxCellsHint, maxSplit, progress);
    internal::bucketRecurse(splats, region, params, process, recursionState);
}

} // namespace Bucket

#endif /* !BUCKET_IMPL_H */
