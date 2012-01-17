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
#include "marching.h"

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

    /**
     * Writes the data to file. The writer passed in must not yet have been opened
     * (this function will do that). The caller may optionally have set comments on it.
     *
     * @throw std::ios_base::failure on I/O failure (including failure to open the file).
     *
     * @pre @ref finalize() has been called.
     */
    template<typename Writer>
    void write(Writer &writer, const std::string &filename) const;
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

    std::size_t numVertices() const;
    std::size_t numTriangles() const;

    Marching::OutputFunctor outputFunctor(unsigned int pass);
    void finalize();

    template<typename Writer>
    void write(Writer &writer, const std::string &filename) const;
};

template<typename Writer>
void SimpleMesh::write(Writer &writer, const std::string &filename) const
{
    writer.setNumVertices(vertices.size());
    writer.setNumTriangles(triangles.size());
    writer.open(filename);
    writer.writeVertices(0, vertices.size(), &vertices[0][0]);
    writer.writeTriangles(0, triangles.size(), &triangles[0][0]);
}

template<typename Writer>
void WeldMesh::write(Writer &writer, const std::string &filename) const
{
    writer.setNumVertices(internalVertices.size() + externalVertices.size());
    writer.setNumTriangles(triangles.size());
    writer.open(filename);
    writer.writeVertices(0, internalVertices.size(), &internalVertices[0][0]);
    writer.writeVertices(internalVertices.size(), externalVertices.size(), &externalVertices[0][0]);
    writer.writeTriangles(0, triangles.size(), &triangles[0][0]);
}

#endif /* MESH_H */
