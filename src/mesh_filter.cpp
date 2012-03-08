/**
 * @file
 *
 * Filter chains for post-processing on-device meshes.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif

#ifndef __CL_ENABLE_EXCEPTIONS
# define __CL_ENABLE_EXCEPTIONS
#endif

#include <CL/cl.hpp>
#include <vector>
#include <algorithm>
#include <boost/function.hpp>
#include <boost/foreach.hpp>
#include "mesh_filter.h"
#include "errors.h"
#include "grid.h"
#include "clh.h"

void MeshFilterChain::operator()(
    const cl::CommandQueue &queue,
    const DeviceKeyMesh &mesh,
    const std::vector<cl::Event> *events,
    cl::Event *event) const
{
    DeviceKeyMesh meshes[2]; // for ping-pong
    DeviceKeyMesh *inMesh = &meshes[0];
    DeviceKeyMesh *outMesh = &meshes[1];
    *inMesh = mesh;

    std::vector<cl::Event> wait(1);
    cl::Event last;
    BOOST_FOREACH(const MeshFilter &filter, filters)
    {
        filter(queue, *inMesh, events, &last, *outMesh);
        wait[0] = last;
        events = &wait;
        std::swap(inMesh, outMesh);
    }
    output(queue, *inMesh, events, event);
}

ScaleBiasFilter::ScaleBiasFilter(const cl::Context &context)
{
    cl::Program program = CLH::build(context, "kernels/scale_bias.cl");
    kernel = cl::Kernel(program, "scaleBiasVertices");
    setScaleBias(1.0f, 0.0f, 0.0f, 0.0f);
}

void ScaleBiasFilter::setScaleBias(float scale, float x, float y, float z)
{
    scaleBias.x = x;
    scaleBias.y = y;
    scaleBias.z = z;
    scaleBias.w = scale;
    kernel.setArg(1, scaleBias);
}

void ScaleBiasFilter::setScaleBias(const Grid &grid)
{
    grid.getVertex(0, 0, 0, scaleBias.s);
    scaleBias.w = grid.getSpacing();
    kernel.setArg(1, scaleBias);
}

void ScaleBiasFilter::operator()(
    const cl::CommandQueue &queue,
    const DeviceKeyMesh &inMesh,
    const std::vector<cl::Event> *events,
    cl::Event *event,
    DeviceKeyMesh &outMesh) const
{
    // TODO: pick a work group size
    kernel.setArg(0, inMesh.vertices);
    CLH::enqueueNDRangeKernel(queue,
                              kernel,
                              cl::NullRange,
                              cl::NDRange(inMesh.numVertices),
                              cl::NullRange,
                              events, event);
    outMesh = inMesh;
}
