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
 */
class MeshBase
{
public:
    virtual unsigned int numPasses() const { return 1; }
    virtual std::size_t numVertices() const = 0;
    virtual std::size_t numTriangles() const = 0;

    virtual Marching::OutputFunctor outputFunctor(unsigned int pass) = 0;
    virtual void finalize() {}
    virtual void write(const std::string &filename) const = 0;

    virtual ~MeshBase() {}
};

/**
 * Output collector for @ref Marching that does not do any welding of
 * external vertices. It simply
 */
class SimpleMesh : public MeshBase
{
private:
    std::vector<boost::array<cl_float, 3> > vertices;
    std::vector<boost::array<cl_uint, 3> > triangles;

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
 */
class WeldMesh : public MeshBase
{
private:
    std::vector<boost::array<cl_float, 3> > internalVertices;
    std::vector<boost::array<cl_float, 3> > externalVertices;
    std::vector<cl_ulong> externalKeys;
    std::vector<boost::array<cl_uint, 3> > triangles;

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
