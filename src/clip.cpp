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

Clip::Clip(const cl::Context &context, const cl::Device &device,
           std::size_t maxVertices, std::size_t maxTriangles)
:
    maxVertices(maxVertices), maxTriangles(maxTriangles),
    discriminant(context, CL_MEM_READ_WRITE, maxVertices * sizeof(cl_float3)),
    compact(context, CL_MEM_READ_WRITE, std::max(maxVertices, maxTriangles) * sizeof(cl_uint)),
    compactScan(context, device, clogs::TYPE_UINT)
{
    // TODO: compile program and extract kernels
}

void Clip::setDistance(const DistanceFunctor &distance)
{
    this->distance = distance;
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
    distance(queue, discriminant, vertices, NULL, &distanceEvent);

    /* TODO: pretty much every call needs to use an explicit work group size
     * and have some manner to handle the leftover bits.
     */

    // TODO: set up kernel arguments
    // TODO: vertices should be classified during triangle compaction?
    // TODO: set up functions to return allocated memory
    // TODO: if interpolation is being done, probably need to split welding
    // out of Marching and re-use it here.

    /*** Classify and compact vertices and their keys ***/

    wait.resize(1);
    wait[0] = distanceEvent;
    cl::Event vertexClassifyEvent;
    queue.enqueueNDRangeKernel(vertexClassifyKernel,
                               cl::NullRange,
                               cl::NDRange(numVertices),
                               cl::NullRange,
                               &wait, &vertexClassifyEvent);

    wait.resize(1);
    wait[0] = vertexClassifyEvent;
    cl::Event vertexScanEvent;
    compactScan.enqueue(queue, compact, numVertices + 1, NULL, &wait, &vertexScanEvent);

    wait.resize(1);
    wait[0] = vertexScanEvent;
    cl_uint vertexCount = 0, internalVertexCount = 0;
    cl::Event vertexCountEvent, internalVertexCountEvent;
    queue.enqueueReadBuffer(compact, CL_FALSE, numInternalVertices * sizeof(cl_uint), sizeof(cl_uint),
                            &internalVertexCount, &wait, &internalVertexCountEvent);
    queue.enqueueReadBuffer(compact, CL_FALSE, numVertices * sizeof(cl_uint), sizeof(cl_uint),
                            &vertexCount, &wait, &vertexCountEvent);

    wait.resize(1);
    wait[0] = vertexScanEvent;
    cl::Event vertexCompactEvent;
    queue.enqueueNDRangeKernel(vertexCompactKernel,
                               cl::NullRange,
                               cl::NDRange(numVertices),
                               cl::NullRange,
                               &wait, &vertexCompactEvent);

    /*** Classify and compact triangles, while also rewriting the indices ***/

    wait.resize(3);
    wait[0] = vertexClassifyEvent;
    wait[1] = internalVertexCountEvent;
    wait[2] = vertexCountEvent;
    cl::Event triangleClassifyEvent;
    queue.enqueueNDRangeKernel(triangleClassifyKernel,
                               cl::NullRange,
                               cl::NDRange(numTriangles),
                               cl::NullRange,
                               &wait, &triangleClassifyEvent);

    wait.resize(1);
    wait[0] = triangleClassifyEvent;
    cl::Event triangleScanEvent;
    compactScan.enqueue(queue, compact, numTriangles + 1, NULL, &wait, &triangleScanEvent);

    wait.resize(1);
    wait[0] = triangleScanEvent;
    cl::Event triangleCountEvent;
    cl_uint triangleCount = 0;
    queue.enqueueReadBuffer(compact, CL_FALSE, numTriangles * sizeof(cl_uint), sizeof(cl_uint),
                            &triangleCount, &wait, &triangleCountEvent);

    wait.resize(1);
    wait[0] = triangleScanEvent;
    cl::Event triangleCompactEvent;
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

    // TODO: arrange to pass these into the following functor:
    wait[3] = vertexCompactEvent;
    wait[4] = triangleCompactEvent;
    cl::Event::waitForEvents(wait);

    output(queue, outVertices, outVertexKeys, outIndices,
           vertexCount, internalVertexCount, triangleCount * 3, event);
}
