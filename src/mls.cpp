/**
 * @file
 *
 * Moving least squares implementation.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif

#ifndef __CL_ENABLE_EXCEPTIONS
# define __CL_ENABLE_EXCEPTIONS
#endif

#include <CL/cl.hpp>
#include <stdexcept>
#include "errors.h"
#include "mls.h"

const std::size_t MlsFunctor::wgs[2] = {16, 16};

MlsFunctor::MlsFunctor(const cl::Context &context)
{
    std::map<std::string, std::string> defines;
    defines["WGS_X"] = boost::lexical_cast<std::string>(wgs[0]);
    defines["WGS_Y"] = boost::lexical_cast<std::string>(wgs[1]);
    program = CLH::build(context, "kernels/mls.cl", defines);
    kernel = cl::Kernel(program, "processCorners");
}

void MlsFunctor::set(const Grid &grid, const SplatTreeCL &tree, unsigned int subsamplingShift)
{
    cl_float3 gridBias3;
    grid.getVertex(0, 0, 0, gridBias3.s);

    cl_float gridScale = grid.getSpacing();
    cl_float2 gridBias;
    for (unsigned int i = 0; i < 2; i++)
        gridBias.s[i] = gridBias3.s[i];

    zScale = gridScale;
    zBias = gridBias3.s[2];

    kernel.setArg(1, tree.getSplats());
    kernel.setArg(2, tree.getCommands());
    kernel.setArg(3, tree.getStart());
    kernel.setArg(4, gridScale);
    kernel.setArg(5, gridBias);
    kernel.setArg(6, 3 * subsamplingShift);

    dims[0] = (grid.numVertices(0) + wgs[0] - 1) / wgs[0] * wgs[0];
    dims[1] = (grid.numVertices(1) + wgs[1] - 1) / wgs[1] * wgs[1];
}

void MlsFunctor::operator()(
    const cl::CommandQueue &queue,
    const cl::Image2D &slice,
    cl_uint z,
    const std::vector<cl::Event> *events,
    cl::Event *event) const
{
    MLSGPU_ASSERT(slice.getImageInfo<CL_IMAGE_WIDTH>() >= dims[0], std::length_error);
    MLSGPU_ASSERT(slice.getImageInfo<CL_IMAGE_HEIGHT>() >= dims[1], std::length_error);

    cl_float zWorld = z * zScale + zBias;
    kernel.setArg(0, slice);
    kernel.setArg(7, cl_int(z));
    kernel.setArg(8, zWorld);
    queue.enqueueNDRangeKernel(kernel,
                               cl::NullRange,
                               cl::NDRange(dims[0], dims[1]),
                               cl::NDRange(wgs[0], wgs[1]),
                               events, event);
}
