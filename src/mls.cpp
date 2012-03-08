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
#include <boost/math/constants/constants.hpp>
#include "errors.h"
#include "mls.h"
#include "clh.h"

const std::size_t MlsFunctor::wgs[2] = {16, 16};

MlsFunctor::MlsFunctor(const cl::Context &context)
{
    std::map<std::string, std::string> defines;
    defines["WGS_X"] = boost::lexical_cast<std::string>(wgs[0]);
    defines["WGS_Y"] = boost::lexical_cast<std::string>(wgs[1]);
    program = CLH::build(context, "kernels/mls.cl", defines);
    kernel = cl::Kernel(program, "processCorners");
    boundaryKernel = cl::Kernel(program, "measureBoundaries");

    setBoundaryLimit(1.0f);
}

void MlsFunctor::set(const Grid::size_type size[3], const Grid::difference_type offset[3],
                     const SplatTreeCL &tree, unsigned int subsamplingShift)
{
    MLSGPU_ASSERT(size[0] % wgs[0] == 0, std::invalid_argument);
    MLSGPU_ASSERT(size[1] % wgs[1] == 0, std::invalid_argument);

    cl_int3 offset3 = {{ offset[0], offset[1], offset[2] }};

    kernel.setArg(1, tree.getSplats());
    kernel.setArg(2, tree.getCommands());
    kernel.setArg(3, tree.getStart());
    kernel.setArg(4, 3 * subsamplingShift);
    kernel.setArg(5, offset3);

    boundaryKernel.setArg(2, tree.getSplats());
    boundaryKernel.setArg(3, tree.getCommands());
    boundaryKernel.setArg(4, tree.getStart());
    boundaryKernel.setArg(5, 3 * subsamplingShift);
    boundaryKernel.setArg(6, offset3);

    dims[0] = size[0];
    dims[1] = size[1];
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

    kernel.setArg(0, slice);
    kernel.setArg(6, cl_int(z));
    queue.enqueueNDRangeKernel(kernel,
                               cl::NullRange,
                               cl::NDRange(dims[0], dims[1]),
                               cl::NDRange(wgs[0], wgs[1]),
                               events, event);
}

void MlsFunctor::setBoundaryLimit(float limit)
{
    // This is computed theoretically based on the weight function, and assuming a
    // uniform distribution of samples and a straight boundary
    const float boundaryScale = (sqrt(6) * 512) / (693 * boost::math::constants::pi<float>());

    boundaryKernel.setArg(7, 1.0f / (boundaryScale * limit));
}

void MlsFunctor::operator()(
    const cl::CommandQueue &queue,
    const cl::Buffer &distance,
    const cl::Buffer &vertices,
    std::size_t numVertices,
    const std::vector<cl::Event> *events,
    cl::Event *event) const
{
    MLSGPU_ASSERT(distance.getInfo<CL_MEM_SIZE>() >= numVertices * sizeof(cl_float), std::length_error);
    MLSGPU_ASSERT(vertices.getInfo<CL_MEM_SIZE>() >= numVertices * (3 * sizeof(cl_float)), std::length_error);

    boundaryKernel.setArg(0, distance);
    boundaryKernel.setArg(1, vertices);
    // TODO: pick a useful work group size
    CLH::enqueueNDRangeKernelSplit(queue,
                                   boundaryKernel,
                                   cl::NullRange,
                                   cl::NDRange(numVertices),
                                   cl::NullRange,
                                   events, event);
}
