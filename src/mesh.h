/**
 * @file
 *
 * Structure to encapsulate mesh data on a device.
 */

#ifndef MESH_H
#define MESH_H

#if HAVE_CONFIG_H
# include <config.h>
#endif

#include <CL/cl.hpp>
#include <cstddef>
#include <vector>
#include <boost/array.hpp>

struct DeviceMesh
{
    cl::Buffer vertices;
    cl::Buffer triangles;

    std::size_t numVertices;
    std::size_t numTriangles;

    DeviceMesh() : vertices(), triangles(), numVertices(0), numTriangles(0) {}
    DeviceMesh(const cl::Context &context, cl_mem_flags flags, std::size_t numVertices, std::size_t numTriangles);
};

struct DeviceKeyMesh : public DeviceMesh
{
    cl::Buffer vertexKeys;
    std::size_t numInternalVertices;

    DeviceKeyMesh() : DeviceMesh(), vertexKeys(), numInternalVertices(0) {}
    DeviceKeyMesh(const cl::Context &context, cl_mem_flags flags,
                  std::size_t numVertices, std::size_t numInternalVertices, std::size_t numTriangles);
};

struct HostMesh
{
    std::vector<boost::array<cl_float, 3> > vertices;
    std::vector<boost::array<cl_uint, 3> > triangles;
};

struct HostKeyMesh : public HostMesh
{
    std::vector<cl_ulong> vertexKeys;
};

void enqueueReadMesh(const cl::CommandQueue &queue,
                     const DeviceKeyMesh &dMesh, HostKeyMesh &hMesh,
                     const std::vector<cl::Event> *events,
                     cl::Event *verticesEvent,
                     cl::Event *vertexKeysEvent,
                     cl::Event *trianglesEvent);

#endif /* !MESH_H */
