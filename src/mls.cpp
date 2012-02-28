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
