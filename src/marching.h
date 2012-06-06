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
 * The implementation operates on a slice at a time. For each slice, it:
 *  -# Identifies the cells that are not empty (not entirely inside, entirely
 *     outside, or containing non-finite values).
 *  -# Compacts those cells into an array to allow more efficient iteration over them.
 *  -# Extracts the number of vertices and indices each (compacted) cell will produce.
 *  -# Performs of these counts to allocate positions in the array. These scans are
 *     seeded with the total number already produced.
 *  -# Generates the vertices and indices.
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
     * An interface for classes that supply the signed distance function
     * for @ref Marching.
     */
    class Generator
    {
    public:
        virtual ~Generator() {}

        /**
         * Return the ideal number of slices this class would like to process at a
         * time. This is just a hint, since there may not be that many available.
         */
        virtual Grid::size_type slicesHint() const = 0;

        /**
         * Allocate storage for holding a range of slices. The image must allow
         * CL to read from it. The layout is that slices are stacked up along
         * the Y dimension.
         *
         * The caller may choose to pad the image to a larger size if it
         * desires. This may be useful if it uses a fixed workgroup size which
         * would otherwise cause it to access outside the image dimensions.
         *
         * @return An image of dimensions at least @a width by @a height * @a depth.
         */
        virtual cl::Image2D allocateSlices(Grid::size_type width, Grid::size_type height, Grid::size_type depth) const = 0;

        /**
         * Enqueue CL work to compute the signed distance function.
         *
         * This function is not required to be threadsafe or to be safe for concurrent
         * execution of the CL commands.
         *
         * @param queue                 The command queue to use.
         * @param distance              Output storage for the signed distance function
         * @param size                  The dimensions of the entire volume.
         * @param zFirst, zLast         Half-open range of Z values to process.
         * @param[out] zStride          Y step between slices.
         * @param events                Events to wait for (may be @c NULL).
         * @param[out] event            Event signaled on completion (may be @c NULL).
         *
         * @pre
         * - All elements of @a size are positive.
         * - 0 &lt;= @a zFirst < @a zLast &lt;= @a size[2].
         * - @a distance was allocated using @ref allocateSlices with dimensions at
         *      least @a size[0], @a size[1], (@a zLast - @a zFirst).
         * - @a zFirst is a multiple of @ref slicesHint().
         * @post
         * - The signed distance for point (x, y, z) in the volume will be stored
         *   in @a distance at coordinates x, y + (z - zFirst) * zStride.
         * - @a zStride is at least @a size[1].
         */
        virtual void enqueue(
            const cl::CommandQueue &queue,
            const cl::Image2D &distance,
            const Grid::size_type size[3],
            Grid::size_type zFirst, Grid::size_type zLast,
            Grid::size_type &zStride,
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
     * The number of cell corners (not cells) in the grid.
     */
    Grid::size_type maxWidth, maxHeight, maxDepth;

    /**
     * Space allocated to hold intermediate vertices and indices.
     *
     * @todo Make these tunable.
     */
    std::size_t vertexSpace, indexSpace;

    cl::Context context;   ///< OpenCL context used to allocate buffers

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
     * Buffer of uint2 values, indexed by compacted cell ID. They are the 2D
     * coordinates of the corresponding original cell.
     */
    cl::Buffer cells;

    /**
     * Buffer of uint values, indexed by uncompacted cell ID. Initially it is a
     * boolean array of cells that are non-empty, which is then scanned to
     * produce the forward compaction mapping. Uncompacted cell IDs are y-major.
     */
    cl::Buffer occupied;

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
     * Sort keys corresponding from @ref unweldedVertices.
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
     * The images holding two slices of the signed distance function.
     */
    cl::Image2D backingImages[2];

    /**
     * @name
     * @{
     * Temporary buffers used during sorting.
     */
    cl::Buffer tmpVertexKeys, tmpVertices;
    /** @} */

    cl::Kernel countOccupiedKernel;         ///< Kernel compiled from @ref countOccupied.
    cl::Kernel compactKernel;               ///< Kernel compiled from @ref compact.
    cl::Kernel countElementsKernel;         ///< Kernel compiled from @ref countElements.
    cl::Kernel generateElementsKernel;      ///< Kernel compiled from @ref generateElements.
    cl::Kernel countUniqueVerticesKernel;   ///< Kernel compiled from @ref countUniqueVertices.
    cl::Kernel compactVerticesKernel;       ///< Kernel compiled from @ref compactVerticesKernel.
    cl::Kernel reindexKernel;               ///< Kernel compiled from @ref reindexKernel.

    clogs::Scan scanUint;                   ///< Scanner to scan @c cl_uint values.
    clogs::Scan scanElements;               ///< Scanner to scan @ref viCount.
    clogs::Radixsort sortVertices;          ///< Sorts vertices by keys for welding.

    /// Pinned memory for doing readbacks
    CLH::PinnedMemory<Readback> readback;

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
    void makeTables();

public:
    /**
     * Checks whether a device is suitable for use with this class. At the time
     * of writing, the only requirement is that images are supported.
     */
    static bool validateDevice(const cl::Device &device);

    /**
     * Returns the maximum number of vertices that may be passed in a call
     * to the output function.
     */
    static std::tr1::uint64_t getMaxVertices(Grid::size_type maxWidth, Grid::size_type maxHeight);

    /**
     * Returns the maximum number of triangles that may be passed in a call
     * to the output function.
     */
    static std::tr1::uint64_t getMaxTriangles(Grid::size_type maxWidth, Grid::size_type maxHeight);

    /**
     * Estimates the device memory required for particular values of the
     * constructor arguments. This is intended to fairly accurately reflect
     * memory allocated in buffers and images, but excludes all overheads for
     * fragmentation, alignment, parameters, programs, command buffers etc.
     *
     * It does @em not account for the memory allocated for the slice data. For that
     * you need to query the specific sub-class of @ref Generator.
     *
     * @param device, maxWidth, maxHeight, maxDepth  Parameters that would be passed to the constructor.
     * @return The required resources.
     *
     * @pre @a maxWidth, @a maxHeight and @a maxDepth do not exceed @ref MAX_DIMENSION.
     */
    static CLH::ResourceUsage resourceUsage(
        const cl::Device &device,
        Grid::size_type maxWidth, Grid::size_type maxHeight, Grid::size_type maxDepth);

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
     * Apart from O(1) overheads for tables etc, the total OpenCL memory
     * allocated is:
     *  - 20 * (@a width - 1) * (@a height - 1) bytes in buffers.
     *  - two calls to @ref Generator::allocateSlices.
     *
     * @param context        OpenCL context used to allocate buffers.
     * @param device         Device for which kernels are to be compiled.
     * @param generator      Generator used to allocate the slice data and pad the
     *                       dimensions. It need not be the same one passed to
     *                       @ref generate, as long as they have the same metadata
     *                       and can interchange image allocations.
     * @param maxWidth, maxHeight, maxDepth Maximum X, Y, Z dimensions (in corners) of the provided sampling grid.
     *
     * @pre
     * - @a maxWidth, @a maxHeight and @a maxDepth are between 2 and @ref MAX_DIMENSION.
     */
    Marching(const cl::Context &context, const cl::Device &device,
             const Generator &generator,
             Grid::size_type maxWidth, Grid::size_type maxHeight, Grid::size_type maxDepth);

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
     * @param generator      Generates the function (see @ref MarchingGenerator).
     * @param output         Functor to receive chunks of output (see @ref OutputFunctor).
     * @param size           Number of vertices in each dimension to process.
     * @param keyOffset      XYZ values to add to vertex keys of external vertices.
     * @param events         Previous events to wait for (can be @c NULL).
     *
     * @note @a keyOffset is specified in integer units, not fixed-point.
     *
     * @note @a size is in units of corners, which is one more than the number of cells.
     *
     * @pre The values of @a size must not exceed the dimensions passed to the
     * constructor.
     */
    void generate(const cl::CommandQueue &queue,
                  Generator &generator,
                  const OutputFunctor &output,
                  const Grid::size_type size[3],
                  const cl_uint3 &keyOffset,
                  const std::vector<cl::Event> *events = NULL);

private:
    /**
     * Represents a single slice within a multi-slice image. Such
     * images pack slices vertically, so @a yOffset is the number of
     * pixels to skip vertically to find the slice in question.
     */
    struct Slice
    {
        cl::Image2D image;
        cl_uint yOffset;
    };

    /**
     * Determine which cells in a slice need to be processed further.
     * This function may wait for previous events, but operates
     * synchronously. On input, two images contain adjacent slices of
     * samples of the function. On output, @ref cells contains a list
     * of x,y pairs giving the coordinates of the cells that will generate
     * geometry, and @ref occupied has been clobbered.
     *
     * @param queue           Command queue to use for enqueuing work.
     * @param sliceA,sliceB   Images containing isofunction values.
     * @param width,height    Dimensions of the image portions that are populated.
     * @param events          Events to wait for before starting (may be @c NULL).
     *
     * @return The number of cells that need further processing.
     *
     * @todo It need not be totally synchronous (compaction is independent).
     */
    std::size_t generateCells(const cl::CommandQueue &queue,
                              const Slice &sliceA,
                              const Slice &sliceB,
                              Grid::size_type width, Grid::size_type height,
                              const std::vector<cl::Event> *events);

    /**
     * Count the number of (unwelded) vertices and indices that will be generated
     * by each (compacted) cell. This function may wait for previous work,
     * but does its own work synchronously and so does not return an event.
     * On output, @ref viCount contains pairs of offsets into the vertex
     * and index outputs that indicate where each cell should emit its
     * geometry.
     *
     * @param queue           Command queue to use for enqueuing work.
     * @param sliceA,sliceB   Images containing isofunction values.
     * @param compacted       Number of cells in @ref cells.
     * @param events          Events to wait for before starting (may be @c NULL).
     * @return The total number of vertices and indices that will be generated.
     */
    cl_uint2 countElements(const cl::CommandQueue &queue,
                           const Slice &sliceA,
                           const Slice &sliceB,
                           std::size_t compacted,
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
     * @param event           Event to wait for before returning (may be @c NULL).
     */
    void shipOut(const cl::CommandQueue &queue,
                 const cl_uint3 &keyOffset,
                 const cl_uint2 &sizes,
                 cl_uint zMax,
                 const OutputFunctor &output,
                 const std::vector<cl::Event> *events,
                 cl::Event *event);
};

#endif /* !MARCHING_H */
