/**
 * @file
 *
 * Data structures for storing the output of @ref Marching.
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
 * Abstract base class for meshes.
 *
 * The basic procedure for using an instance of this class is:
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
class MeshBase
{
public:
    /// Number of passes required.
    virtual unsigned int numPasses() const { return 1; }
    /// Number of vertices captured so far.
    virtual std::size_t numVertices() const = 0;
    /// Number of triangles captured so far.
    virtual std::size_t numTriangles() const = 0;

    /**
     * Retrieves a functor to be passed to @ref Marching::generate in a
     * specific pass.
     * Multi-pass subclasses may do finalization on a previous pass before
     * returning the functor, so this function should only be called for
     * pass @a pass once pass @a pass - 1 has completed.
     *
     * @pre @a pass is less than @ref numPasses().
     */
    virtual Marching::OutputFunctor outputFunctor(unsigned int pass) = 0;

    /**
     * Perform any final processing once the last pass has completed.
     *
     * @pre All passes have been executed.
     */
    virtual void finalize() {}

    /**
     * Writes the data to file.
     *
     * @throw std::ios_base::failure on I/O failure (including failure to open the file).
     *
     * @pre @ref finalize() has been called.
     */
    virtual void write(const std::string &filename) const = 0;

    virtual ~MeshBase() {}
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
    virtual std::size_t numVertices() const;
    virtual std::size_t numTriangles() const;

    virtual Marching::OutputFunctor outputFunctor(unsigned int pass);
    virtual void write(const std::string &filename) const;
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
    virtual std::size_t numVertices() const;
    virtual std::size_t numTriangles() const;

    virtual Marching::OutputFunctor outputFunctor(unsigned int pass);
    virtual void finalize();
    virtual void write(const std::string &filename) const;
};

#endif /* MESH_H */
