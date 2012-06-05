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
#include "misc.h"

std::map<std::string, MlsShape> MlsShapeWrapper::getNameMap()
{
    std::map<std::string, MlsShape> ans;
    ans["plane"] = MLS_SHAPE_PLANE;
    ans["sphere"] = MLS_SHAPE_SPHERE;
    return ans;
}

const std::size_t MlsFunctor::wgs[3] = {8, 8, 8};

MlsFunctor::MlsFunctor(const cl::Context &context, MlsShape shape)
    : context(context)
{
    std::map<std::string, std::string> defines;
    defines["WGS_X"] = boost::lexical_cast<std::string>(wgs[0]);
    defines["WGS_Y"] = boost::lexical_cast<std::string>(wgs[1]);
    defines["WGS_Z"] = boost::lexical_cast<std::string>(wgs[2]);
    defines["FIT_SPHERE"] = shape == MLS_SHAPE_SPHERE ? "1" : "0";
    defines["FIT_PLANE"] = shape == MLS_SHAPE_PLANE ? "1" : "0";

    cl::Program program = CLH::build(context, "kernels/mls.cl", defines);
    kernel = cl::Kernel(program, "processCorners");
    boundaryKernel = cl::Kernel(program, "measureBoundaries");

    setBoundaryLimit(1.0f);
}

void MlsFunctor::set(const Grid::difference_type offset[3],
                     const SplatTreeCL &tree, unsigned int subsamplingShift)
{
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
}

cl::Image2D MlsFunctor::allocateSlices(
    Grid::size_type width, Grid::size_type height, Grid::size_type depth) const
{
    width = roundUp(width, wgs[0]);
    height = roundUp(height, wgs[1]);
    depth = roundUp(depth, wgs[2]);

    return cl::Image2D(context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT),
                       width, height * depth);
}

void MlsFunctor::enqueue(
    const cl::CommandQueue &queue,
    const cl::Image2D &distance,
    const Grid::size_type size[3],
    Grid::size_type zFirst, Grid::size_type zLast,
    Grid::size_type &zStride,
    const std::vector<cl::Event> *events,
    cl::Event *event)
{
    Grid::size_type width = roundUp(size[0], wgs[0]);
    Grid::size_type height = roundUp(size[1], wgs[1]);

    MLSGPU_ASSERT(distance.getImageInfo<CL_IMAGE_WIDTH>() >= width, std::length_error);
    MLSGPU_ASSERT(distance.getImageInfo<CL_IMAGE_HEIGHT>() >= height * (zLast - zFirst), std::length_error);
    MLSGPU_ASSERT(zFirst < zLast, std::invalid_argument);

    kernel.setArg(0, distance);

    std::vector<cl::Event> wait;
    cl::Event last;
    if (events != NULL)
        wait = *events;
    for (Grid::size_type z = zFirst; z < zLast; z++)
    {
        kernel.setArg(6, cl_int(z));
        kernel.setArg(7, cl_int((z - zFirst) * height));
        queue.enqueueNDRangeKernel(kernel,
                                   cl::NullRange,
                                   cl::NDRange(wgs[0] * wgs[1], width / wgs[0], height / wgs[1]),
                                   cl::NDRange(wgs[0] * wgs[1], 1, 1),
                                   &wait, &last);
        wait.resize(1);
        wait[0] = last;
    }
    zStride = height;
    if (event != NULL)
        *event = last;
}

void MlsFunctor::setBoundaryLimit(float limit)
{
    // This is computed theoretically based on the weight function, and assuming a
    // uniform distribution of samples and a straight boundary
    const float boundaryScale = (sqrt(6.0f) * 512) / (693 * boost::math::constants::pi<float>());

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
    CLH::enqueueNDRangeKernelSplit(queue,
                                   boundaryKernel,
                                   cl::NullRange,
                                   cl::NDRange(numVertices),
                                   cl::NullRange,
                                   events, event);
}
