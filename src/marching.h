/**
 * @file
 *
 * Marching tetrahedra algorithm.
 */

#ifndef MARCHING_H
#define MARCHING_H

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <CL/cl.hpp>
#include <cstddef>
#include <vector>
#include <utility>
#include "tr1_cstdint.h"
#include <boost/function.hpp>
#include <clogs/clogs.h>
#include "grid.h"
#include "mesh.h"
#include "clh.h"

class TestMarching;

/**
 * Marching tetrahedra algorithm implemented in OpenCL.
 * An instance of this class contains buffers to hold intermediate state,
 * and is specialized for a specific OpenCL context and device. An instance
 * also currently specialized to the dimensions of the sampling grid, although
 * that could easily be changed to allow smaller jobs.
 *
 * At present, it has the following features and limitations:
 *  - The values are supplied by a callback object that is called to enqueue work to produce
 *    a number of slices at a time. This allows the algorithm to work even if the entire
 *    grid is too large to fit in memory.
 *  - The outputs are supplied to another functor, which may be called more than once for
 *    a single volume. This allows the algorithm to work even if the output is too large
 *    to hold in GPU memory.
 *  - Where a non-finite value is present in the field, all adjacent cells are discarded.
 *    This can lead to boundaries in the surface.
 *  - Shared vertices at cell boundaries are welded together, except for those where the
 *    volume was subdivided into separate calls to the output functor.
 *  - Boundary ("external") vertices have associated vertex keys that uniquely identify
 *    them and allow them to be welded in a post-processing step.
 *
 * The implementation operates on @em swathes of slices at a time. For each
 * swathe, it:
 *  -# Identifies the cells that are not empty (not entirely inside, entirely
 *     outside, or containing non-finite values) and counts the vertices and indices
 *     they produce.
 *  -# Compacts those cells into an array to allow more efficient iteration over them.
 *  -# Performs a scan of these counts to allocate positions in the array.
 *     This scan is seeded with the total number already produced.
 *  -# Generates the vertices and indices.
 *
 * The user specifies the maximum memory to use for intermediate vertices and indices.
 * If this amount is exceeded, the existing data is shipped out and collection is
 * restarted. It is also possible that a single swathe has too much data, in which case
 * it is split into separate slices. However, slices are never split.
 */
class Marching
{
    friend class TestMarching;
public:
    enum
    {
        MAX_CELL_VERTICES = 13 ///< Maximum vertices generated per cell
    };
    enum
    {
        MAX_CELL_INDICES = 36  ///< Maximum triangles generated per cell
    };
    enum
    {
        /// Bytes of storage required for all internal data for a worst-case cell
        MAX_CELL_BYTES = MAX_CELL_VERTICES * (2 * sizeof(cl_uint) + 2 * sizeof(cl_float4) + 2 * sizeof(cl_ulong))
            + MAX_CELL_INDICES * sizeof(cl_uint)
    };
    enum
    {
        NUM_CUBES = 256        ///< Number of possible vertex codes for a cube (2^vertices)
    };
    enum
    {
        NUM_EDGES = 19         ///< Number of edges in each cube
    };
    enum
    {
        NUM_TETRAHEDRA = 6     ///< Number of tetrahedra in each cube
    };
    enum
    {
        /// Number of bits in fixed-point xyz fields in a vertex key (including fractional bits)
        KEY_AXIS_BITS = 21
    };
    enum
    {
        /// Logarithm base 2 of @ref MAX_DIMENSION.
        MAX_DIMENSION_LOG2 = 13
    };
    enum
    {
        /// Logarithm base 2 of @ref MAX_GLOBAL_DIMENSION.
        MAX_GLOBAL_DIMENSION_LOG2 = KEY_AXIS_BITS - 1
    };

    enum
    {
        /**
         * Maximum size that is legal to pass to the constructor for @a maxWidth,
         * @a maxHeight or @a maxDepth.  This does not guarantee that it will
         * be possible to allocate sufficient memory, but asking for more will
         * fail without even trying.
         *
         * The current value is chosen on the basis that it is the minimum value of
         * @c CL_DEVICE_IMAGE2D_MAX_WIDTH. It could be raised if necessary, but for
         * current usage there is little point.
         */
        MAX_DIMENSION = 1U << MAX_DIMENSION_LOG2
    };

    enum
    {
        /**
         * Maximum size that is legal for global coordinates (after biasing
         * with an offset).
         */
        MAX_GLOBAL_DIMENSION = (1U << MAX_GLOBAL_DIMENSION_LOG2) - 1
    };

    enum
    {
        /// Total bytes held in @ref countTable.
        COUNT_TABLE_BYTES = 256 * sizeof(cl_uchar2)
    };
    enum
    {
        /// Total bytes held in @ref startTable.
        START_TABLE_BYTES = 257 * sizeof(cl_ushort2)
    };
    enum
    {
        /// Total bytes held in @ref dataTable.
        DATA_TABLE_BYTES = 8192 * sizeof(cl_uchar)
    };
    enum
    {
        /// Total bytes held in @ref keyTable.
        KEY_TABLE_BYTES = 2432 * sizeof(cl_uint3)
    };

    /**
     * Contains data necessary for accessing slices in a packed image.
     * The @a width and @a height may be less than the actual allocated
     * sizes, but describe the dimensions that contain payload.
     * A point at coordinates (@a x, @a y, @a z) in the volume is stored
     * at location (@a x, @a z * @c zStride + @c zBias). Note that in
     * most cases @a zBias will be negative.
     */
    struct ImageParams
    {
        Grid::size_type width;
        Grid::size_type height;
        cl_uint zStride;
        cl_int zBias;
    };

    /**
     * Augments image parameters with a range of slices to process in the
     * image. The @c zFirst and @c zLast can be considered to be either
     * a closed interval of corners or a half-open interval of cells.
     */
    struct Swathe : public ImageParams
    {
        Grid::size_type zFirst;
        Grid::size_type zLast;
    };

    /**
     * An interface for classes that supply the signed distance function
     * for @ref Marching.
     */
    class Generator
    {
    public:
        virtual ~Generator() {}

        /**
         * Returns alignment requirements. The X and Y alignment requirements
         * are used to pad up image allocations. The Z alignment requirement
         * restricts the starting Z value for enqueuing. The value must be
         * constant per axis.
         *
         * @return A 3-element array of X, Y, Z alignment values
         *
         * @see @ref enqueue
         */
        virtual const Grid::size_type *alignment() const = 0;

        /**
         * Enqueue CL work to compute the signed distance function.
         *
         * This function is not required to be threadsafe or to be safe for concurrent
         * execution of the CL commands.
         *
         * @param queue                 The command queue to use.
         * @param distance              Output storage for the signed distance function
         * @param swathe                Swathe of values to produce
         * @param events                Events to wait for (may be @c NULL).
         * @param[out] event            Event signaled on completion (may be @c NULL).
         *
         * @pre
         * - @a swathe.width and @a swathe.height are positive.
         * - @a swathe.zFirst &lt;= @a swathe.zLast
         * - The X size of @a distance is at least <code>roundUp</code>(@a swathe.width, #alignment (0)).
         * - The Y size of @a distance is at least
         *   @a swathe.zStride * roundUp(@a zLast + 1, #alignment (2)) + @a zBias
         * - @a swathe.zStride is at least <code>roundUp</code>(@a swathe.height, #alignment (1))
         * - @a swathe.zFirst is a multiple of #alignment (2).
         * @post
         * - The signed distances for @a z values in the swathe will be stored
         *   in @a distance (see @ref Marching::ImageParams for details).
         * - The pixels at lower @a z coordinates are unaffected.
         * - Other pixels in the image have undefined values.
         */
        virtual void enqueue(
            const cl::CommandQueue &queue,
            const cl::Image2D &distance,
            const Swathe &swathe,
            const std::vector<cl::Event> *events,
            cl::Event *event) = 0;
    };

private:
    /**
     * Structure to hold the various values read back from the device at
     * various times.  This is allocated in a @ref CLH::PinnedMemory so that only
     * one page needs to be pinned.
     */
    struct Readback
    {
        cl_uint compacted;
        cl_uint2 elementCounts;
        cl_uint numWelded;
        cl_uint firstExternal;
    };

    /**
     * The vertices incident on each edge. It is important that the vertex indices
     * are in order in each edge.
     */
    static const unsigned char edgeIndices[NUM_EDGES][2];

    /**
     * The vertices of each tetrahedron in a cube. The vertices must be wound
     * consistently such that the first three appear counter-clockwise when
     * viewed from the fourth in a right-handed coordinate system.
     */
    static const unsigned char tetrahedronIndices[NUM_TETRAHEDRA][4];

    /**
     * The maximum number of cell corners (not cells) in the grid, excluding
     * alignment padding.
     */
    Grid::size_type maxWidth, maxHeight, maxDepth;

    /**
     * The number of slices to process in one go.
     */
    Grid::size_type maxSwathe;

    /**
     * Space allocated to hold intermediate vertices and indices.
     */
    std::size_t vertexSpace, indexSpace;

    /**
     * Buffer of uchar2 values, indexed by cube code. The two elements are
     * the number of vertices and indices generated by the cell.
     */
    cl::Buffer countTable;
    /**
     * Buffer of ushort2 values, indexed by cube code. The two elements are
     * the positions of the index array and vertex array in @ref dataTable.
     * It has one extra element at the end so that the element range for
     * the last cube code can be found.
     */
    cl::Buffer startTable;
    /**
     * Buffer of uchar values, which are either indices to be emitted
     * (after biasing), or vertices represented as an edge ID. The range
     * of vertices or indices for a particular cube code is determined by
     * two adjacent elements of @ref countTable.
     */
    cl::Buffer dataTable;
    /**
     * Buffer of cl_uint3 values, in ranges indexed by startTable. It
     * contains offsets added to the 3 parts of the cell key to get a vertex
     * key for each vertex generated in a cell (prior to computing the three
     * fields into a single ulong). The cell key is simply the vertex
     * key for the vertex at minimum-x/y/z corner.
     *
     * Each value is in .1 fixed-point format.
     */
    cl::Buffer keyTable;

    /**
     * Buffer of uint2 values, indexed by compacted cell ID. Initially they are
     * the number of vertices and indices generated by each cell;
     * after a scan they are the positions to write the vertex/indices.
     * There is an extra element at the end to allow for a scan that
     * gives the total.
     */
    cl::Buffer viCount;

    /**
     * Buffer of uint3 values, indexed by compacted cell ID. They are the 3D
     * coordinates of the corresponding original cell, with the Z coordinate
     * indexing @ref image rather than the full volume.
     */
    cl::Buffer cells;

    /**
     * Buffer containing 1 uint, the number of generated non-empty cells.
     */
    cl::Buffer numOccupied;

    /**
     * Number of vertices and indices produced for each slice. Each element
     * is a uint2, and is indexed relative to the local volume.
     */
    cl::Buffer viHistogram;

    /**
     * Intermediate unwelded vertices. These are @c cl_float4 values, with the
     * w component holding a bit-cast of the original index before sorting by key.
     */
    cl::Buffer unweldedVertices;

    /**
     * Welded vertices. These are tightly packed @c float3 values.
     */
    cl::Buffer weldedVertices;

    /**
     * Indices. Before welding, these are local (0-based) and index the @ref unweldedVertices
     * array. During welding, these are rewritten to refer to welded vertices, and are
     * offset by the number of vertices previously shipped out so that they index the
     * virtual array of all emitted vertices.
     */
    cl::Buffer indices;

    /**
     * Sort keys corresponding to @ref unweldedVertices.
     *
     * There is an additional sentinel at the end with value @c ULONG_MAX.
     */
    cl::Buffer unweldedVertexKeys;

    /**
     * Sort keys corresponding to @ref weldedVertices. Only defined for external vertices.
     */
    cl::Buffer weldedVertexKeys;

    /**
     * Indicator for whether each unwelded vertex is unique. It is then scanned
     * to produce a remap table for compacting vertices. It also has one extra
     * element at the end to allow the total number of welded vertices to be
     * read back.
     */
    cl::Buffer vertexUnique;

    /**
     * Remapping table used to map unwelded vertex indices to welded vertex
     * indices. It combines the sorting by key with the compaction.
     */
    cl::Buffer indexRemap;

    /**
     * Single @c cl_uint value containing the index in @c weldedVertices
     * of the first external vertex.
     */
    cl::Buffer firstExternal;

    /**
     * The image holding slices of the signed distance function.
     */
    cl::Image2D image;

    /**
     * The number of y steps between slices in the backing image.
     */
    Grid::size_type zStride;

    /**
     * @name
     * @{
     * Temporary buffers used during sorting.
     */
    cl::Buffer tmpVertexKeys, tmpVertices;
    /** @} */

    cl::Kernel genOccupiedKernel;           ///< Kernel compiled from @ref genOccupied.
    cl::Kernel generateElementsKernel;      ///< Kernel compiled from @ref generateElements.
    cl::Kernel countUniqueVerticesKernel;   ///< Kernel compiled from @ref countUniqueVertices.
    cl::Kernel compactVerticesKernel;       ///< Kernel compiled from @ref compactVerticesKernel.
    cl::Kernel reindexKernel;               ///< Kernel compiled from @ref reindexKernel.
    cl::Kernel copySliceKernel;             ///< Kernel compiled from @ref copySliceKernel (for driver bug workaround).

    /**
     * @name
     * @{
     * Statistics measuring time spent in kernels with corresponding names.
     */
    Statistics::Variable &genOccupiedKernelTime;
    Statistics::Variable &generateElementsKernelTime;
    Statistics::Variable &countUniqueVerticesKernelTime;
    Statistics::Variable &compactVerticesKernelTime;
    Statistics::Variable &reindexKernelTime;
    Statistics::Variable &copySliceTime;    ///< Time for slice copy, either with kernel or with @c clEnqueueCopyImage
    Statistics::Variable &zeroTime;         ///< Time to zero out buffers
    Statistics::Variable &readbackTime;     ///< Time to read back metadata

    /** @} */

    clogs::Scan scanUint;                   ///< Scanner to scan @c cl_uint values.
    clogs::Scan scanElements;               ///< Scanner to scan @ref viCount.
    clogs::Radixsort sortVertices;          ///< Sorts vertices by keys for welding.

    /// Pinned memory for doing readbacks
    CLH::PinnedMemory<Readback> readback;

    /// Pinned memory for reading back @ref viHistogram
    CLH::PinnedMemory<cl_uint2> viReadback;

    /**
     * Finds the edge incident on vertices v0 and v1.
     *
     * @pre edge (v0, v1) is one of the existing edges
     */
    static unsigned int findEdgeByVertexIds(unsigned int v0, unsigned int v1);

    /**
     * Determines the parity of a permutation.
     *
     * The permutation can contain any unique values - they do not need to be
     * 0..n-1. It is considered to be the permutation that would map the
     * sorted sequence to the given sequence.
     *
     * @param first, last The range to measure (forward iterators)
     * @retval 0 if the permutation contains an odd number of swaps
     * @retval 1 if the permutation contains an even number of swaps.
     */
    template<typename Iterator>
    static unsigned int permutationParity(Iterator first, Iterator last);

    /**
     * Populate the static tables describing how to slice up cells.
     */
    void makeTables(const cl::Context &context);

public:
    /**
     * Checks whether a device is suitable for use with this class. At the time
     * of writing, the only requirement is that images are supported.
     *
     * @throw CLH::invalid_device if the device cannot be used.
     */
    static void validateDevice(const cl::Device &device);

    /**
     * Estimates the device memory required for particular values of the
     * constructor arguments. This is intended to fairly accurately reflect
     * memory allocated in buffers and images, but excludes all overheads for
     * fragmentation, alignment, parameters, programs, command buffers etc.
     *
     * @param device, maxWidth, maxHeight, maxDepth, maxSwathe, meshMemory, alignment  Parameters that would be passed to the constructor.
     *
     * @return The required resources.
     *
     * @pre @a maxWidth, @a maxHeight and @a maxDepth do not exceed @ref MAX_DIMENSION.
     */
    static CLH::ResourceUsage resourceUsage(
        const cl::Device &device,
        Grid::size_type maxWidth, Grid::size_type maxHeight, Grid::size_type maxDepth,
        Grid::size_type maxSwathe,
        std::size_t meshMemory,
        const Grid::size_type alignment[3]);

    /**
     * The function type to pass to @ref generate for receiving output data.
     * When invoked, this function must enqueue commands to retrieve the data
     * from the supplied buffers. It must wait for the given events before
     * accessing the data, and it must return an event that will be signaled
     * when it is safe for the caller to overwrite the supplied buffers (if it
     * operates synchronously, it should just return an already-signaled user
     * event).
     *
     * Calls to this function are serialized, but the work that it enqueues may
     * proceed in parallel.
     *
     * The command-queue is provided for convenience. If the event returned is
     * not associated with the same command-queue, the callee is responsible for
     * ensuring that the work completes within finite time.
     *
     * The vertices and indices returned will correspond. The indices will be
     * local i.e. an index of 0 indicates the initial element of @a vertices,
     * regardless of how many previous vertices have been passed to the same
     * functor. The vertices are in tightly packed cl_float triplets
     * (x,y,z) while the indices are in tightly packed cl_uint index triplets.
     *
     * The vertices are partitioned into internal and external vertices, with the
     * internal ones first. @a numInternalVertices indicates the position of the split.
     * For the external vertices, @a vertexKeys gives the keys, which can be used by
     * the caller to weld external vertices together. @a vertexKeys is indexed in the
     * same way as @a vertices, but the keys for internal vertices are undefined.
     */
    typedef boost::function<void(const cl::CommandQueue &,
                                 const DeviceKeyMesh &mesh,
                                 const std::vector<cl::Event> *events,
                                 cl::Event *event)> OutputFunctor;

    /**
     * Constructor. Note that it must be possible to allocate an OpenCL 2D image of
     * dimensions @a width by @a height, so they should be constrained appropriately.
     *
     * @param context        OpenCL context used to allocate buffers.
     * @param device         Device for which kernels are to be compiled.
     * @param maxWidth, maxHeight, maxDepth Maximum X, Y, Z dimensions (in corners) of the provided sampling grid.
     * @param maxSwathe      Maximum number of slices to process in one go (in cells)
     * @param meshMemory     Bytes of memory to allocate for mesh data (including internal data)
     * @param alignment      Alignment values that would be returned by @ref Generator::alignment.
     *
     * @pre
     * - @a maxWidth, @a maxHeight, @a maxDepth are between 2 and @ref MAX_DIMENSION.
     * - @a maxSwathe is at least @a alignment[2]
     * - @a meshMemory &gt;= (@a maxWidth - 1) * (@a maxHeight - 1) * @ref MAX_CELL_BYTES
     */
    Marching(const cl::Context &context, const cl::Device &device,
             Grid::size_type maxWidth, Grid::size_type maxHeight, Grid::size_type maxDepth,
             Grid::size_type maxSwathe,
             std::size_t meshMemory,
             const Grid::size_type alignment[3]);

    /**
     * Generate an isosurface.
     *
     * @note Because this function needs to read back intermediate results
     * before enqueuing more work, this is not purely an enqueuing operation.
     * It will block until all of the work has completed. To hide latency, it is
     * necessary to have something happening on another CPU thread.
     *
     * The region that is processed is assumed to be at an offset of @a
     * keyOffset within some larger grid. To accommodate this, vertex keys for
     * external vertices are offset by @a keyOffset. The output vertices are
     * also transformed into the global grid coordinate systems using the formula
     * \f$v_{\text{out}} = v_{\text{in}} - \text{keyOffset}.\f$
     * The interpolation is done in a way that guarantees invariance, provided that the
     * surrounding isovalues are invariant.
     *
     * @param queue          Command queue to enqueue the work to.
     * @param generator      Generates the function (see @ref Generator).
     * @param output         Functor to receive chunks of output (see @ref OutputFunctor).
     * @param size           Number of vertices in each dimension to process.
     * @param keyOffset      XYZ values to add to vertex keys of external vertices.
     * @param events         Previous events to wait for (can be @c NULL).
     *
     * @note @a keyOffset is specified in integer units, not fixed-point.
     *
     * @note @a size is in units of corners, which is one more than the number of cells.
     *
     * @pre
     * - The values of @a size must not exceed the dimensions passed to the
     *   constructor.
     * - The generator's alignment must be compatible with (i.e., divide into) the
     *   alignment values passed to the constructor.
     */
    void generate(const cl::CommandQueue &queue,
                  Generator &generator,
                  const OutputFunctor &output,
                  const Grid::size_type size[3],
                  const cl_uint3 &keyOffset,
                  const std::vector<cl::Event> *events = NULL);

private:
    /**
     * Copy one slice of the image to another.
     *
     * @param queue           Command queue to use for enqueuing work.
     * @param image           Image to operate on.
     * @param zSrc            Slice number for source.
     * @param zTrg            Slice number for target.
     * @param params          Image parameters.
     * @param events          Events to wait for before starting (may be @c NULL).
     * @param[out] event      Event signalled on completion (may be @c NULL).
     *
     * @note @a src and @a trg are image-relative i.e., the @c zBias in
     * @a params does not apply.
     */
    void copySlice(
        const cl::CommandQueue &queue,
        const cl::Image2D &image,
        Grid::size_type zSrc,
        Grid::size_type zTrg,
        const ImageParams &params,
        const std::vector<cl::Event> *events,
        cl::Event *event);

    /**
     * Determine which cells in a slice need to be processed further,
     * and produce per-cell counts of vertices and indices.
     * This function may wait for previous events, but operates
     * synchronously. On input, two images contain adjacent slices of
     * samples of the function. On output, @ref cells contains a list
     * of x,y pairs giving the coordinates of the cells that will generate
     * geometry, @ref viCount contains the vertex and index counts per
     * cell, and @ref numOccupied contains the number of cells.
     *
     * @param queue           Command queue to use for enqueuing work.
     * @param swathe          Swathe of data to process
     * @param events          Events to wait for before starting (may be @c NULL).
     *
     * @return The number of cells that need further processing.
     *
     * @note @a firstSlice and @a lastSlice reference corners, so only
     * @a lastSlice - @a firstSlice cell-slices are processed.
     */
    std::size_t generateCells(
        const cl::CommandQueue &queue,
        const Swathe &swathe,
        const std::vector<cl::Event> *events);

    /**
     * Post-process a batch of geometry and send it to the output functor.
     * This function operates asynchronously, with an event returned to
     * indicate completion. It handles welding of shared vertices into
     * a single vertex and the corresponding reindexing, as well as
     * offsetting indices to be relative to the global set of vertices.
     *
     * The input vertices are in @ref unweldedVertices and @ref indices.
     * The welded vertices are placed in @ref weldedVertices, and the indices
     * are updated in-place. As a side effect, @ref vertexUnique and
     * @ref indexRemap are clobbered.
     *
     * @param queue           Command queue to use for enqueuing work.
     * @param keyOffset       Value added to keys (see @ref generate for details).
     * @param sizes           Number of vertices and indices in input.
     * @param zMax            Maximum potential z value of vertices (not cells).
     * @param output          Functor to which the welded geometry is passed.
     * @param events          Events to wait for before starting (may be @c NULL).
     * @param[out] event      Event signalled on completion (may be @c NULL).
     */
    void shipOut(const cl::CommandQueue &queue,
                 const cl_uint3 &keyOffset,
                 const cl_uint2 &sizes,
                 cl_uint zMax,
                 const OutputFunctor &output,
                 const std::vector<cl::Event> *events,
                 cl::Event *event);

    /**
     * Recursively process a swathe. This function will call @ref shipOut as
     * necessary to make space in the vertex and index buffers. It will first
     * attempt to process the whole swathe in one go, but failing that it will
     * use a slice histogram to split it into maximal pieces.
     *
     * @param queue           Command queue to use for enqueuing work.
     * @param output          Passed to @ref shipOut.
     * @param swathe          The range of slices to process.
     * @param keyOffset       Passed to @ref shipOut.
     * @param localSize       Work group size, matching the dynamic local memory allocation.
     * @param[in,out] offsets Positions in vertex and index buffers to start appending.
     * @param[in,out] zTop    Z value for corners at top of last shipped-out data
     * @param events          Events to wait for before starting (may be @c NULL).
     * @param[out] event      Event signalled on completion (may be @c NULL).
     *
     * @pre
     * - All kernel arguments for @ref generateElements have been set, except for
     *   @a top.
     */
    Grid::size_type addSlices(
        const cl::CommandQueue &queue,
        const OutputFunctor &output,
        const Swathe &swathe,
        const cl_uint3 &keyOffset,
        std::size_t localSize,
        cl_uint2 &offsets, cl_uint &zTop,
        const std::vector<cl::Event> *events,
        cl::Event *event);
};

#endif /* !MARCHING_H */
