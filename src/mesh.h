/**
 * @file
 *
 * Data structures for storing the output of @ref Marching.
 *
 * The classes in this file are @ref MeshBase, an abstract base class, and
 * several concrete instantiations of it. They differ in terms of
 *  - the number of passes needed
 *  - whether they support welding of external vertices
 *  - the amount of temporary memory required.
 */

#ifndef MESH_H
#define MESH_H

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <CL/cl.h>

#include <string>
#include <vector>
#include <map>
#include <string>
#include <iosfwd>
#include <boost/array.hpp>
#include <boost/noncopyable.hpp>
#include <boost/thread/mutex.hpp>
#include <tr1/unordered_map>
#include "marching.h"
#include "src/fast_ply.h"

/**
 * Enumeration of the supported mesh types
 */
enum MeshType
{
    SIMPLE_MESH,
    WELD_MESH,
    BIG_MESH,
    STXXL_MESH
};

/**
 * Wrapper around @ref MeshType for use with @ref Choice.
 */
class MeshTypeWrapper
{
public:
    typedef MeshType type;
    static std::map<std::string, MeshType> getNameMap();
};

/**
 * Abstract base class for output collectors for @ref Marching.
 *
 * The basic procedure for using one of these classes is:
 * -# Instantiate it.
 * -# Uses @ref numPasses to determine how many passes are required.
 * -# For each pass, call @ref outputFunctor to obtain a functor, then
 *    make as many calls to @ref Marching::generate as desired using this
 *    functor. Each call should set @a keyOffset so that vertex keys line up.
 *    Each pass must generate exactly the same geometry, but the chunks may
 *    be generated in different order. The functor may be called from
 *    multiple threads simultaneously.
 * -# Call @ref finalize.
 * -# If file output is desired, call @ref write.
 */
class MeshBase
{
protected:
#if UNIT_TESTS
    /**
     * Internal implementation of manifold testing.
     * @param numVertices The number of vertices.
     * @param triangles   Indices indexing the range [0, @a numVertices).
     */
    static bool isManifold(std::size_t numVertices, const std::vector<boost::array<cl_uint, 3> > &triangles);
#endif

    /**
     * Mutex used by subclasses to serialize their output functors.
     */
    boost::mutex mutex;

public:
    /// Virtual destructor to allow destruction via base class pointer
    virtual ~MeshBase() {}

    /// Number of passes required.
    virtual unsigned int numPasses() const = 0;

    /**
     * Retrieves a functor to be passed to @ref Marching::generate in a
     * specific pass.
     * Multi-pass classes may do finalization on a previous pass before
     * returning the functor, so this function should only be called for
     * pass @a pass once pass @a pass - 1 has completed. It must also
     * only be called once per pass.
     *
     * @pre @a pass is less than @ref numPasses().
     */
    virtual Marching::OutputFunctor outputFunctor(unsigned int pass) = 0;

    /**
     * Perform any final processing once the last pass has completed.
     *
     * @param progressStream If non-NULL and finalization does significant work,
     * a progress meter will be displayed.
     * @pre All passes have been executed.
     */
    virtual void finalize(std::ostream *progressStream = NULL) { (void) progressStream; }

#if UNIT_TESTS
    /**
     * Determine whether the mesh is manifold (possibly with boundary).
     *
     * This is intended for test code, and so is not necessarily efficient.
     * It also need not be implemented by all classes. If unimplemented, it
     * must return false (the default implementation provides this).
     *
     * A mesh is considered non-manifold if it has out-of-range indices or
     * isolated vertices (not part of any triangle).
     *
     * @pre @ref finalize() has already been called.
     */
    virtual bool isManifold() const { return false; }
#endif

    /**
     * Writes the data to file. The writer passed in must not yet have been opened
     * (this function will do that). The caller may optionally have set comments on it.
     *
     * @param writer          Writer to write to (unopened).
     * @param filename        Filename to write to.
     * @param progressStream  If non-NULL, a log stream for a progress meter
     * @throw std::ios_base::failure on I/O failure (including failure to open the file).
     *
     * @pre @ref finalize() has been called.
     */
    virtual void write(FastPly::WriterBase &writer, const std::string &filename,
                       std::ostream *progressStream = NULL) const = 0;
};

/**
 * Output collector for @ref Marching that does not do any welding of
 * external vertices. It simply collects all the vertices into one vector
 * and indices into another.
 */
class SimpleMesh : public MeshBase
{
private:
    /// Storage for vertices
    std::vector<boost::array<cl_float, 3> > vertices;

    /// Storage for indices.
    std::vector<boost::array<cl_uint, 3> > triangles;

    /// Function called by the functor.
    void add(const cl::CommandQueue &queue,
             const cl::Buffer &vertices,
             const cl::Buffer &vertexKeys,
             const cl::Buffer &indices,
             std::size_t numVertices,
             std::size_t numInternalVertices,
             std::size_t numIndices,
             cl::Event *event);

public:
    virtual unsigned int numPasses() const { return 1; }
    virtual Marching::OutputFunctor outputFunctor(unsigned int pass);

#if UNIT_TESTS
    virtual bool isManifold() const;
#endif

    virtual void write(FastPly::WriterBase &writer, const std::string &filename,
                       std::ostream *progressStream = NULL) const;
};

/**
 * Mesh that operates in one pass, and implements welding of external
 * vertices.
 *
 * Internal and external vertices are partitioned into separate arrays,
 * and external vertex keys are kept to allow final welding. In
 * intermediate stages, indices are encoded as either an index into
 * the internal vector, or as the bit-inverse (~) of an index into
 * the external vector. The external indices are rewritten during
 * @ref finalize().
 */
class WeldMesh : public MeshBase
{
private:
    /// Storage for internal vertices
    std::vector<boost::array<cl_float, 3> > internalVertices;
    /// Storage for external vertices
    std::vector<boost::array<cl_float, 3> > externalVertices;
    /// Storage for external vertex keys
    std::vector<cl_ulong> externalKeys;
    /// Storage for indices
    std::vector<boost::array<cl_uint, 3> > triangles;

    /// Implementation of the functor
    void add(const cl::CommandQueue &queue,
             const cl::Buffer &vertices,
             const cl::Buffer &vertexKeys,
             const cl::Buffer &indices,
             std::size_t numVertices,
             std::size_t numInternalVertices,
             std::size_t numIndices,
             cl::Event *event);

public:
    virtual unsigned int numPasses() const { return 1; }
    virtual Marching::OutputFunctor outputFunctor(unsigned int pass);
    virtual void finalize(std::ostream *progressStream = NULL);

#if UNIT_TESTS
    virtual bool isManifold() const;
#endif

    virtual void write(FastPly::WriterBase &writer, const std::string &filename,
                       std::ostream *progressStream = NULL) const;
};

namespace detail
{

/**
 * An internal base class for @ref BigMesh and @ref StxxlMesh, implementing
 * algorithms common to both.
 *
 * External vertices are entered into a hash table that maps their keys to
 * their final indices in the output. When a new chunk of data comes in,
 * new external vertices are entered into the key map and known ones are
 * discarded. The indices are then rewritten using the key map.
 */
class KeyMapMesh : public MeshBase
{
protected:
    typedef std::tr1::unordered_map<cl_ulong, cl_uint> map_type;

    /// Maps external vertex keys to external indices
    map_type keyMap;

    /**
     * @name
     * @{
     * Temporary buffers.
     * These are stored in the object so that memory can be recycled if
     * possible, rather than thrashing the allocator.
     */
    std::vector<boost::array<cl_float, 3> > tmpVertices;
    std::vector<cl_ulong> tmpVertexKeys;
    std::vector<boost::array<cl_uint, 3> > tmpTriangles;
    std::vector<cl_uint> tmpIndexTable;
    /** @} */

    /**
     * Load data from the OpenCL buffers into host memory.
     * The supplied vectors are resized to the appropriate
     * size for the data that is being loaded.
     *
     * @post
     * - The read into hKeys is complete.
     * - The read into hVertices will be complete when @a verticesEvent is signalled.
     * - The read into hTriangles will be complete when @a trianglesEvent is signalled.
     * - The events will complete is finish time (i.e., the queue will have been flushed).
     */
    void loadData(const cl::CommandQueue &queue,
                  const cl::Buffer &dVertices,
                  const cl::Buffer &dVertexKeys,
                  const cl::Buffer &dIndices,
                  std::vector<boost::array<cl_float, 3> > &hVertices,
                  std::vector<cl_ulong> &hVertexKeys,
                  std::vector<boost::array<cl_uint, 3> > &hTriangles,
                  std::size_t numVertices,
                  std::size_t numInternalVertices,
                  std::size_t numTriangles,
                  cl::Event *verticesEvent,
                  cl::Event *trianglesEvent) const;

    /**
     * Add external vertex keys to the key map and computes an index rewrite table.
     * The index rewrite table maps local external indices for a block to their
     * final values.
     *
     * @param vertexOffset    The final index for the first external vertex in the block.
     * @param hKeys           Keys of the external vertices.
     * @param[out] indexTable The index remapping table.
     */
    std::size_t updateKeyMap(
        cl_uint vertexOffset,
        const std::vector<cl_ulong> &hKeys,
        std::vector<cl_uint> &indexTable);

    /**
     * Writes indices in place from being block-relative to the final form.
     * @param priorVertices        First vertex in the block (internal or external).
     * @param numInternalVertices  Number of internal vertices in the block.
     * @param indexTable           External index rewrite table computed by @ref updateKeyMap.
     * @param[in,out] triangles    Vertex indices.
     */
    void rewriteTriangles(
        cl_uint priorVertices,
        std::size_t numInternalVertices,
        const std::vector<cl_uint> &indexTable,
        std::vector<boost::array<cl_uint, 3> > &triangles) const;
};

} // namespace detail

/**
 * Two-pass collector that can handle very large meshes by writing
 * the geometry to file as it is produced. It requires an out-of-order
 * writer, and requires the writer to be provided up front.
 *
 * The two passes are:
 * 1. Counting, and assigning the key mapping to detect duplicate external
 *    vertices.
 * 2. Write the data.
 *
 * Unlike @ref WeldMesh, the external vertices are written out as they come in
 * (immediately after the internal vertices for the corresponding chunk), which
 * avoids the need to buffer them up until the end. The only unbounded memory is
 * for the key map.
 */
class BigMesh : public detail::KeyMapMesh
{
private:
    typedef FastPly::WriterBase::size_type size_type;

    FastPly::WriterBase &writer;
    const std::string filename;

    size_type nVertices;    ///< Number of vertices seen in first pass
    size_type nTriangles;   ///< Number of triangles seen in first pass

    size_type nextVertex;   ///< Number of vertices written so far
    size_type nextTriangle; ///< Number of triangles written so far

    /// Implementation of the first-pass functor
    void count(const cl::CommandQueue &queue,
               const cl::Buffer &vertices,
               const cl::Buffer &vertexKeys,
               const cl::Buffer &indices,
               std::size_t numVertices,
               std::size_t numInternalVertices,
               std::size_t numIndices,
               cl::Event *event);

    /// Implementation of the second-pass functor
    void add(const cl::CommandQueue &queue,
             const cl::Buffer &vertices,
             const cl::Buffer &vertexKeys,
             const cl::Buffer &indices,
             std::size_t numVertices,
             std::size_t numInternalVertices,
             std::size_t numIndices,
             cl::Event *event);

public:
    virtual unsigned int numPasses() const { return 2; }

    /**
     * Constructor. Unlike the in-core mesh types, the file information must
     * be passed to the constructor so that results can be streamed into it.
     *
     * The file will be created on the second pass.
     */
    BigMesh(FastPly::WriterBase &writer, const std::string &filename);

    virtual Marching::OutputFunctor outputFunctor(unsigned int pass);

    /**
     * Completes writing. The parameters must have the same values given to the constructor.
     */
    virtual void write(FastPly::WriterBase &writer, const std::string &filename,
                       std::ostream *progressStream = NULL) const;
};

#include <stxxl.h>

/**
 * Mesh class that uses the same algorithm as @ref BigMesh, but stores
 * the data in STXXL containers before concatenating them rather than
 * using multiple passes. It thus trades storage requirements against
 * performance, at least when @ref BigMesh is compute-bound.
 */
class StxxlMesh : public detail::KeyMapMesh
{
private:
    typedef FastPly::WriterBase::size_type size_type;

    typedef stxxl::VECTOR_GENERATOR<boost::array<float, 3> >::result vertices_type;
    typedef stxxl::VECTOR_GENERATOR<boost::array<cl_uint, 3> >::result triangles_type;
    vertices_type vertices;
    triangles_type triangles;

    /// Implementation of the functor
    void add(const cl::CommandQueue &queue,
             const cl::Buffer &vertices,
             const cl::Buffer &vertexKeys,
             const cl::Buffer &indices,
             std::size_t numVertices,
             std::size_t numInternalVertices,
             std::size_t numIndices,
             cl::Event *event);

    /// Function object that accepts incoming vertices and writes them to a writer.
    class VertexBuffer : public boost::noncopyable
    {
    private:
        FastPly::WriterBase &writer;
        size_type nextVertex;
        std::vector<boost::array<float, 3> > buffer;
    public:
        typedef void result_type;

        VertexBuffer(FastPly::WriterBase &writer, size_type capacity);
        void operator()(const boost::array<float, 3> &vertex);
        void flush();
    };

    /// Function object that accepts incoming triangles and writes them to a writer.
    class TriangleBuffer : public boost::noncopyable
    {
    private:
        FastPly::WriterBase &writer;
        size_type nextTriangle;
        std::vector<boost::array<std::tr1::uint32_t, 3> > buffer;
    public:
        typedef void result_type;

        TriangleBuffer(FastPly::WriterBase &writer, size_type capacity);
        void operator()(const boost::array<std::tr1::uint32_t, 3> &triangle);
        void flush();
    };

public:
    virtual unsigned int numPasses() const { return 1; }
    virtual Marching::OutputFunctor outputFunctor(unsigned int pass);
    virtual void write(FastPly::WriterBase &writer, const std::string &filename,
                       std::ostream *progressStream = NULL) const;
};

/**
 * Factory function to create a mesh of the specified type.
 */
MeshBase *createMesh(MeshType type, FastPly::WriterBase &writer, const std::string &filename);

#endif /* !MESH_H */
