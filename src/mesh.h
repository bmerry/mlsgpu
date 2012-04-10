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
#include "allocator.h"

/**
 * Encapsulates a mesh consisting of vertices and triangles in OpenCL buffers.
 *
 * The buffers are not required to be fully utilized; separate counts indicate
 * how much is allocated. It is also valid for @a numTriangles or @a numVertices
 * to be zero; consumers of device mesh objects must deal with this gracefully.
 *
 * It is unspecified whether a @c DeviceMesh is the unique owner of its memory
 * or whether it is shared via OpenCL's reference-counting mechanism.
 */
struct DeviceMesh
{
    /**
     * Buffer containing the vertices, which are tightly-packed @c cl_float
     * xyz triplets (not @c cl_float3).
     */
    cl::Buffer vertices;
    /**
     * Buffer containing the triangle indices, which are triplets of @c cl_uint.
     */
    cl::Buffer triangles;

    std::size_t numVertices;   ///< Number of vertices.
    std::size_t numTriangles;  ///< Number of triangles.

    /**
     * Default constructor. In this state the buffers are unallocated.
     */
    DeviceMesh() : vertices(), triangles(), numVertices(0), numTriangles(0) {}

    /**
     * Constructor. The buffers are allocated with just enough space to hold
     * the specified number of vertices and triangles. It is legal for
     * @a numVertices or @a numTriangles to be zero.
     *
     * @param context       Context used to allocate buffers.
     * @param flags         Memory flags passed to create the buffer.
     * @param numVertices   The number of vertices to allocate.
     * @param numTriangles  The number of triangles to allocate.
     */
    DeviceMesh(const cl::Context &context, cl_mem_flags flags, std::size_t numVertices, std::size_t numTriangles);
};

/**
 * Refinement of @ref DeviceMesh in which every vertex has an associated key
 * and the vertices are partitioned in @em internal (at the front) and @em
 * external (at the back).
 */
struct DeviceKeyMesh : public DeviceMesh
{
    cl::Buffer vertexKeys;                 ///< Vertex keys
    std::size_t numInternalVertices;       ///< Number of internal vertices

    /**
     * Default constructor. In this state the buffers are unallocated.
     */
    DeviceKeyMesh() : DeviceMesh(), vertexKeys(), numInternalVertices(0) {}

    /**
     * Constructor. The buffers are allocated with just enough space to hold
     * the specified number of vertices and triangles. It is legal for
     * @a numVertices or @a numTriangles to be zero.
     */
    DeviceKeyMesh(const cl::Context &context, cl_mem_flags flags,
                  std::size_t numVertices, std::size_t numInternalVertices, std::size_t numTriangles);
};

/**
 * A host-memory counterpart to @ref DeviceKeyMesh. However, unlike a @ref
 * DeviceKeyMesh, the host holds keys @em only for external vertices. Thus,
 * the number of internal vertices can be determined as
 * <code>vertices.size() - vertexKeys.size()</code>, and
 * <code>vertexKeys[i]</code> corresponds to <code>vertices[i +
 * numInternalVertices]</code>.
 */
struct HostKeyMesh
{
    Statistics::Container::vector<boost::array<cl_float, 3> > vertices;
    Statistics::Container::vector<boost::array<cl_uint, 3> > triangles;
    Statistics::Container::vector<cl_ulong> vertexKeys;

    HostKeyMesh() :
        vertices("mem.HostMesh::vertices"),
        triangles("mem.HostMesh::triangles"),
        vertexKeys("mem.HostKeyMesh::vertexKeys") {}
};

/**
 * Transfer mesh data from the device to the host. Each of the three buffers
 * is optionally transferred; to skip transfer, pass @c NULL for the
 * corresponding event (if you want the transfer but don't care about the
 * event, you will need to pass one anyway).
 *
 * For each transferred property, the corresponding element of @a hMesh
 * are discarded. Properties that are not transferred as preserved in @a
 * hMesh.
 *
 * @param queue          Queue in which to enqueue the transfers.
 * @param dMesh          Source of the copy.
 * @param hMesh          Target of the copy.
 * @param events         Events to wait for before starting the copy.
 * @param verticesEvent  Event signaled when the vertex copy is complete
 *                       (@c NULL to disable the copy).
 * @param vertexKeysEvent Event signaled when the vertex key copy is complete
 *                       (@c NULL to disable the copy).
 * @param trianglesEvent Event signaled when the triangle copy is complete
 *                       (@c NULL to disable the copy).
 *
 * @pre @a dMesh has allocated buffers, and @a numVertices and @a numTriangles
 * are positive.
 */
void enqueueReadMesh(const cl::CommandQueue &queue,
                     const DeviceKeyMesh &dMesh, HostKeyMesh &hMesh,
                     const std::vector<cl::Event> *events,
                     cl::Event *verticesEvent,
                     cl::Event *vertexKeysEvent,
                     cl::Event *trianglesEvent);

#endif /* !MESH_H */
