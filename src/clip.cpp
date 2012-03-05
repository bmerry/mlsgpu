/**
 * @file
 *
 * Marching output functor for boundary clipping.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif

#ifndef __CL_ENABLE_EXCEPTIONS
# define __CL_ENABLE_EXCEPTIONS
#endif

#include <CL/cl.hpp>
#include <boost/function.hpp>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include "errors.h"
#include "marching.h"
#include "clip.h"
#include "statistics.h"
#include "clh.h"

Clip::Clip(const cl::Context &context, const cl::Device &device,
           std::size_t maxVertices, std::size_t maxTriangles)
:
    maxVertices(maxVertices), maxTriangles(maxTriangles),
    distances(context, CL_MEM_READ_WRITE, maxVertices * sizeof(cl_float)),
    vertexCompact(context, CL_MEM_READ_WRITE, (maxVertices + 1) * sizeof(cl_uint)),
    triangleCompact(context, CL_MEM_READ_WRITE, (maxTriangles + 1) * sizeof(cl_uint)),
    compactScan(context, device, clogs::TYPE_UINT),
    outVertices(context, CL_MEM_READ_WRITE, maxVertices * sizeof(cl_float3)),
    outVertexKeys(context, CL_MEM_READ_WRITE, maxVertices * sizeof(cl_ulong)),
    outIndices(context, CL_MEM_READ_WRITE, maxTriangles * (3 * sizeof(cl_uint)))
{
    std::vector<cl::Device> devices(1, device);
    program = CLH::build(context, devices, "kernels/clip.cl");
    vertexInitKernel = cl::Kernel(program, "vertexInit");
    classifyKernel = cl::Kernel(program, "classify");
    triangleCompactKernel = cl::Kernel(program, "triangleCompact");
    vertexCompactKernel = cl::Kernel(program, "vertexCompact");
}

void Clip::setDistanceFunctor(const DistanceFunctor &distanceFunctor)
{
    this->distanceFunctor = distanceFunctor;
}

void Clip::setOutput(const Marching::OutputFunctor &output)
{
    this->output = output;
}

void Clip::operator()(
    const cl::CommandQueue &queue,
    const cl::Buffer &vertices,
    const cl::Buffer &vertexKeys,
    const cl::Buffer &indices,
    std::size_t numVertices,
    std::size_t numInternalVertices,
    std::size_t numIndices,
    cl::Event *event)
{
    const std::size_t numTriangles = numIndices / 3;
    MLSGPU_ASSERT(numVertices <= maxVertices, std::length_error);
    MLSGPU_ASSERT(numTriangles <= maxTriangles, std::length_error);

    std::vector<cl::Event> wait;

    cl::Event distanceEvent;
    distanceFunctor(queue, distances, vertices, numVertices, NULL, &distanceEvent);

    /* TODO:
     * - Pretty much every call needs to use an explicit work group size
     *   and have some manner to handle the leftover bits.
     * - Move setArg into constructor where feasible
     * - If interpolation is being done, probably need to split welding
     *   out of Marching and re-use it here.
     * - Change interface to pass in wait events, to avoid waitForEvents
     *   below.
     */

    /*** Classify and compact vertices and indices ***/

    cl::Event vertexInitEvent;
    vertexInitKernel.setArg(0, vertexCompact);
    queue.enqueueNDRangeKernel(vertexInitKernel,
                               cl::NullRange,
                               cl::NDRange(numVertices),
                               cl::NullRange,
                               NULL, &vertexInitEvent);

    wait.resize(2);
    wait[0] = distanceEvent;
    wait[1] = vertexInitEvent;
    cl::Event classifyEvent;
    classifyKernel.setArg(0, triangleCompact);
    classifyKernel.setArg(1, vertexCompact);
    classifyKernel.setArg(2, indices);
    classifyKernel.setArg(3, distances);
    queue.enqueueNDRangeKernel(classifyKernel,
                               cl::NullRange,
                               cl::NDRange(numTriangles),
                               cl::NullRange,
                               &wait, &classifyEvent);

    /*** Compact vertices and their keys ***/

    wait.resize(1);
    wait[0] = classifyEvent;
    cl::Event vertexScanEvent;
    compactScan.enqueue(queue, vertexCompact, numVertices + 1, NULL, &wait, &vertexScanEvent);

    wait.resize(1);
    wait[0] = vertexScanEvent;
    cl_uint vertexCount = 0, internalVertexCount = 0;
    cl::Event vertexCountEvent, internalVertexCountEvent;
    queue.enqueueReadBuffer(vertexCompact, CL_FALSE, numInternalVertices * sizeof(cl_uint), sizeof(cl_uint),
                            &internalVertexCount, &wait, &internalVertexCountEvent);
    queue.enqueueReadBuffer(vertexCompact, CL_FALSE, numVertices * sizeof(cl_uint), sizeof(cl_uint),
                            &vertexCount, &wait, &vertexCountEvent);

    wait.resize(1);
    wait[0] = vertexScanEvent;
    cl::Event vertexCompactEvent;
    vertexCompactKernel.setArg(0, outVertices);
    vertexCompactKernel.setArg(1, outVertexKeys);
    vertexCompactKernel.setArg(2, vertexCompact);
    vertexCompactKernel.setArg(3, vertices);
    vertexCompactKernel.setArg(4, vertexKeys);
    queue.enqueueNDRangeKernel(vertexCompactKernel,
                               cl::NullRange,
                               cl::NDRange(numVertices),
                               cl::NullRange,
                               &wait, &vertexCompactEvent);

    /*** Compact triangles, while also rewriting the indices ***/

    wait.resize(1);
    wait[0] = classifyEvent;
    cl::Event triangleScanEvent;
    compactScan.enqueue(queue, triangleCompact, numTriangles + 1, NULL, &wait, &triangleScanEvent);

    wait.resize(1);
    wait[0] = triangleScanEvent;
    cl::Event triangleCountEvent;
    cl_uint triangleCount = 0;
    queue.enqueueReadBuffer(triangleCompact, CL_FALSE, numTriangles * sizeof(cl_uint), sizeof(cl_uint),
                            &triangleCount, &wait, &triangleCountEvent);

    wait.resize(2);
    wait[0] = triangleScanEvent;
    wait[1] = vertexScanEvent;
    cl::Event triangleCompactEvent;
    triangleCompactKernel.setArg(0, outIndices);
    triangleCompactKernel.setArg(1, triangleCompact);
    triangleCompactKernel.setArg(2, indices);
    triangleCompactKernel.setArg(3, vertexCompact);
    queue.enqueueNDRangeKernel(triangleCompactKernel,
                               cl::NullRange,
                               cl::NDRange(numTriangles),
                               cl::NullRange,
                               &wait, &triangleCompactEvent);

    // Some of these happen-after others and so some steps are redundant, but
    // checking for all of them is safe.
    wait.resize(5);
    wait[0] = internalVertexCountEvent;
    wait[1] = vertexCountEvent;
    wait[2] = triangleCountEvent;
    wait[3] = vertexCompactEvent;
    wait[4] = triangleCompactEvent;
    cl::Event::waitForEvents(wait);

    if (vertexCount > 0)
    {
        output(queue, outVertices, outVertexKeys, outIndices,
               vertexCount, internalVertexCount, triangleCount * 3, event);
    }
    else if (event != NULL)
    {
        cl::UserEvent done(queue.getInfo<CL_QUEUE_CONTEXT>());
        done.setStatus(CL_COMPLETE);
        *event = done;
    }
}
