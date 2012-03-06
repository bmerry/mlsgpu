/**
 * @file
 *
 * Implementation of @ref mesh.h.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif

#ifndef __CL_ENABLE_EXCEPTIONS
# define __CL_ENABLE_EXCEPTIONS
#endif

#include <CL/cl.hpp>
#include <vector>
#include "mesh.h"
#include "clh.h"

DeviceMesh::DeviceMesh(
    const cl::Context &context, cl_mem_flags flags,
    std::size_t numVertices, std::size_t numTriangles)
:
    vertices(context, flags, numVertices * (3 * sizeof(cl_float))),
    triangles(context, flags, numTriangles * (3 * sizeof(cl_uint))),
    numVertices(numVertices),
    numTriangles(numTriangles)
{
}

DeviceKeyMesh::DeviceKeyMesh(
    const cl::Context &context, cl_mem_flags flags,
    std::size_t numVertices, std::size_t numInternalVertices, std::size_t numTriangles)
:
    DeviceMesh(context, flags, numVertices, numTriangles),
    vertexKeys(context, flags, numVertices * sizeof(cl_ulong)),
    numInternalVertices(numInternalVertices)
{
}

void enqueueReadMesh(const cl::CommandQueue &queue,
                     const DeviceKeyMesh &dMesh, HostKeyMesh &hMesh,
                     const std::vector<cl::Event> *events,
                     cl::Event *verticesEvent,
                     cl::Event *vertexKeysEvent,
                     cl::Event *trianglesEvent)
{
    const std::size_t numExternalVertices = dMesh.numVertices - dMesh.numInternalVertices;
    hMesh.vertices.resize(dMesh.numVertices);
    hMesh.vertexKeys.resize(numExternalVertices);
    hMesh.triangles.resize(dMesh.numTriangles);

    if (trianglesEvent != NULL)
        queue.enqueueReadBuffer(dMesh.triangles, CL_FALSE,
                                0, dMesh.numTriangles * (3 * sizeof(cl_uint)),
                                &hMesh.triangles[0][0],
                                events, trianglesEvent);

    if (numExternalVertices > 0 && vertexKeysEvent != NULL)
    {
        queue.enqueueReadBuffer(dMesh.vertexKeys, CL_FALSE,
                                dMesh.numInternalVertices * sizeof(cl_ulong),
                                numExternalVertices * sizeof(cl_ulong),
                                &hMesh.vertexKeys[0],
                                events, vertexKeysEvent);
        /* Start this transfer going while we queue up the following ones */
        queue.flush();
    }
    else
        CLH::doneEvent(queue, vertexKeysEvent);

    if (verticesEvent != NULL)
        queue.enqueueReadBuffer(dMesh.vertices, CL_FALSE,
                                0, dMesh.numVertices * (3 * sizeof(cl_float)),
                                &hMesh.vertices[0][0],
                                events, verticesEvent);
    queue.flush();
}
