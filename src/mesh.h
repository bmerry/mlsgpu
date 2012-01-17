/**
 * @file
 *
 * Data structures for storing the output of @ref Marching.
 *
 * The classes in this file implement a common concept. The documentation
 * can be found under @ref SimpleMesh.
 */

#ifndef MESH_H
#define MESH_H

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <CL/cl.h>

#include <string>
#include <vector>
#include <boost/array.hpp>
#include <tr1/unordered_map>
#include "marching.h"
#include "src/fast_ply.h"

/**
 * Output collector for @ref Marching that does not do any welding of
 * external vertices. It simply collects all the vertices into one vector
 * and indices into another.
 *
 * This is one of several classes implementing the same interface.
 * The basic procedure for using these classes is:
 * -# Instantiate it.
 * -# Uses @ref numPasses to determine how many passes are required.
 * -# For each pass, call @ref outputFunctor to obtain a functor, then
 *    make as many calls to @ref Marching::generate as desired using this
 *    functor. Each call should set @a indexOffset to @ref numVertices(),
 *    and also set @a keyOffset so that vertex keys line up. Each pass
 *    must generate exactly the same vertices.
 * -# Call @ref finalize.
 * -# If file output is desired, call @ref write.
 */
class SimpleMesh
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
    /// Number of passes required.
    static const unsigned int numPasses = 1;

    /// Number of vertices captured so far.
    std::size_t numVertices() const;
    /// Number of triangles captured so far.
    std::size_t numTriangles() const;

    /**
     * Retrieves a functor to be passed to @ref Marching::generate in a
     * specific pass.
     * Multi-pass classes may do finalization on a previous pass before
     * returning the functor, so this function should only be called for
     * pass @a pass once pass @a pass - 1 has completed.
     *
     * @pre @a pass is less than @ref numPasses().
     */
    Marching::OutputFunctor outputFunctor(unsigned int pass);

    /**
     * Perform any final processing once the last pass has completed.
     *
     * @pre All passes have been executed.
     */
    void finalize() {}

#if UNIT_TESTS
    /**
     * Determine whether the mesh is manifold (possibly with boundary).
     *
     * This is intended for test code, and so is not necessarily efficient.
     *
     * @pre @ref finalize() has already been called.
     */
    bool isManifold() const;

    /**
     * Internal implementation. It is public so that other classes can make
     * use of it, but it is not intended to be called directly.
     */
    static bool isManifold(std::size_t numVertices, const std::vector<boost::array<cl_uint, 3> > &triangles);
#endif

    /**
     * Writes the data to file. The writer passed in must not yet have been opened
     * (this function will do that). The caller may optionally have set comments on it.
     *
     * @throw std::ios_base::failure on I/O failure (including failure to open the file).
     *
     * @pre @ref finalize() has been called.
     */
    void write(FastPly::WriterBase &writer, const std::string &filename) const;
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
 *
 * See @ref SimpleMesh for documentation of the public interface.
 */
class WeldMesh
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
    static const unsigned int numPasses = 1;

    typedef std::size_t size_type;

    size_type numVertices() const;
    size_type numTriangles() const;

    Marching::OutputFunctor outputFunctor(unsigned int pass);
    void finalize();

#if UNIT_TESTS
    bool isManifold() const;
#endif

    void write(FastPly::WriterBase &writer, const std::string &filename) const;
};

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
 *
 * See @ref SimpleMesh for documentation of the public interface.
 */
class BigMesh
{
public:
    typedef FastPly::WriterBase::size_type size_type;

private:
    FastPly::WriterBase &writer;
    const std::string filename;

    /// Maps external vertex keys to external indices
    std::tr1::unordered_map<cl_ulong, cl_uint> keyMap;

    size_type nVertices;  ///< Number of vertices seen in first pass
    size_type nTriangles; ///< Number of triangles seen in first pass

    size_type nextVertex;   ///< Number of vertices written so far
    size_type nextTriangle; ///< Number of triangles written so far

    size_type inVertices;   ///< Running total of vertices passed in to second pass

    /**
     * @name
     * @{
     * Temporary buffers for reading data from OpenCL.
     * These are stored in the object so that memory can be recycled if
     * possible, rather than thrashing the allocator.
     */
    std::vector<cl_ulong> tmpKeys;
    std::vector<boost::array<cl_float, 3> > tmpVertices;
    std::vector<boost::array<cl_uint, 3> > tmpTriangles;
    /** @} */

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
    static const unsigned int numPasses = 2;

    /**
     * Constructor. Unlike the in-core mesh types, the file information must
     * be passed to the constructor so that results can be streamed into it.
     *
     * The file will be created on the second pass.
     */
    BigMesh(FastPly::WriterBase &writer, const std::string &filename);

    size_type numVertices() const;
    size_type numTriangles() const;

    Marching::OutputFunctor outputFunctor(unsigned int pass);
    void finalize();

    /**
     * Completes writing. The parameters must have the same values given to the constructor.
     */
    void write(FastPly::WriterBase &writer, const std::string &filename) const;
};

#endif /* MESH_H */
