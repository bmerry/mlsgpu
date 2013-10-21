/*
 * mlsgpu: surface reconstruction from point clouds
 * Copyright (C) 2013  University of Cape Town
 *
 * This file is part of mlsgpu.
 *
 * mlsgpu is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

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
    : kernelTime(Statistics::getStatistic<Statistics::Variable>("kernel.scaleBias.time"))
{
    cl::Program program = CLH::build(context, "kernels/scale_bias.cl");
    kernel = cl::Kernel(program, "scaleBiasVertices");
    setScaleBias(1.0f, 0.0f, 0.0f, 0.0f);
}

void ScaleBiasFilter::setScaleBias(float scale, float x, float y, float z)
{
    scaleBias.s[0] = x;
    scaleBias.s[1] = y;
    scaleBias.s[2] = z;
    scaleBias.s[3] = scale;
    kernel.setArg(1, scaleBias);
}

void ScaleBiasFilter::setScaleBias(const Grid &grid)
{
    grid.getVertex(0, 0, 0, scaleBias.s);
    scaleBias.s[3] = grid.getSpacing();
    kernel.setArg(1, scaleBias);
}

void ScaleBiasFilter::operator()(
    const cl::CommandQueue &queue,
    const DeviceKeyMesh &inMesh,
    const std::vector<cl::Event> *events,
    cl::Event *event,
    DeviceKeyMesh &outMesh) const
{
    /* The "if" test is because AMD APP SDK 2.6 barfs if inMesh.vertices is
     * NULL, even though this is legal according to the CL spec. Note: don't
     * try to move the enqueue inside the if test. Even though no CL work will
     * be generated, the enqueue will still populate the event.
     */
    if (inMesh.numVertices() > 0)
        kernel.setArg(0, inMesh.vertices);
    CLH::enqueueNDRangeKernelSplit(queue,
                                   kernel,
                                   cl::NullRange,
                                   cl::NDRange(inMesh.numVertices()),
                                   cl::NullRange,
                                   events, event, &kernelTime);
    outMesh = inMesh;
}
