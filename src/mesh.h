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

class MeshSizes
{
private:
    std::size_t numVertices_;
    std::size_t numTriangles_;
    std::size_t numInternalVertices_;

public:
    MeshSizes() : numVertices_(0), numTriangles_(0), numInternalVertices_(0) {}

    MeshSizes(std::size_t numVertices, std::size_t numTriangles, std::size_t numInternalVertices)
        : numVertices_(numVertices),
        numTriangles_(numTriangles),
        numInternalVertices_(numInternalVertices)
    {
    }

    bool operator==(const MeshSizes &b) const
    {
        return numVertices_ == b.numVertices_
            && numTriangles_ == b.numTriangles_
            && numInternalVertices_ == b.numInternalVertices_;
    }

    void assign(std::size_t numVertices, std::size_t numTriangles, std::size_t numInternalVertices)
    {
        numVertices_ = numVertices;
        numTriangles_ = numTriangles;
        numInternalVertices_ = numInternalVertices;
    }

    std::size_t numVertices() const { return numVertices_; }
    std::size_t numTriangles() const { return numTriangles_; }
    std::size_t numInternalVertices() const { return numInternalVertices_; }
    std::size_t numExternalVertices() const { return numVertices_ - numInternalVertices_; }

    /**
     * Number of bytes that need to be allocated for @ref HostKeyMesh::HostKeyMesh.
     */
    std::size_t getHostBytes() const
    {
        return 3 * sizeof(cl_float) * numVertices_
            +  3 * sizeof(cl_uint) * numTriangles_
            +  sizeof(cl_ulong) * numExternalVertices();
    }
};

/**
 * Encapsulates a mesh consisting of vertices, triangles and vertex keys in
 * OpenCL buffers. Every vertex has an associated key and the vertices are
 * partitioned in @em internal (at the front) and @em external (at the back).
 * Vertex keys are present for all vertices but only meaningful for the
 * external vertices.
 *
 * The buffers are not required to be fully utilized; separate counts indicate
 * how much is allocated. It is also valid for @a numTriangles or @a numVertices
 * to be zero; consumers of device mesh objects must deal with this gracefully.
 *
 * It is unspecified whether a @c DeviceMesh is the unique owner of its memory
 * or whether it is shared via OpenCL's reference-counting mechanism.
 */
struct DeviceKeyMesh : public MeshSizes
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
    /**
     * Buffer containing vertex keys, which are @c cl_ulong values.
     */
    cl::Buffer vertexKeys;                 ///< Vertex keys

    DeviceKeyMesh() {}

    /**
     * Constructor. The buffers are allocated with just enough space to hold
     * the specified number of vertices and triangles. It is legal for
     * any of the sizes to be zero.
     */
    DeviceKeyMesh(const cl::Context &context, cl_mem_flags flags, const MeshSizes &sizes);
};

/**
 * A host-memory counterpart to @ref DeviceKeyMesh. However, unlike a @ref
 * DeviceKeyMesh, the host holds keys @em only for external vertices. Thus,
 * <code>vertexKeys[i]</code> corresponds to <code>vertices[i +
 * numInternalVertices]</code>.
 */
struct HostKeyMesh : public MeshSizes
{
    boost::array<cl_float, 3> *vertices;
    boost::array<cl_uint, 3> *triangles;
    cl_ulong *vertexKeys;

    HostKeyMesh() :
        vertices(NULL), triangles(NULL), vertexKeys(NULL) {}

    /**
     * Construct from an existing pool of memory.
     *
     * @pre @a ptr is @c cl_ulong aligned.
     */
    HostKeyMesh(void *ptr, const MeshSizes &sizes);
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
