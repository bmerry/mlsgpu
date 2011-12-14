/**
 * @file
 *
 * Implementation of @ref SplatTreeCL.
 */

#ifndef __CL_ENABLE_EXCEPTIONS
# define __CL_ENABLE_EXCEPTIONS
#endif

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <CL/cl.hpp>
#include <boost/smart_ptr/scoped_ptr.hpp>
#include <cstddef>
#include "splat_tree_cl.h"
#include "splat.h"
#include "grid.h"
#include "clh.h"

SplatTree::command_type *SplatTreeCL::Buffer::allocate(const cl::Context &context, const cl::Device &device, std::size_t size)
{
    buffer = cl::Buffer(context, CL_MEM_READ_ONLY, size * sizeof(command_type));
    mapping.reset(new CLH::BufferMapping(buffer, device, CL_MAP_WRITE, 0, size * sizeof(command_type)));
    return static_cast<command_type *>(mapping->get());
}

#if USE_IMAGES
SplatTree::command_type *SplatTreeCL::Image3D::allocate(const cl::Context &context, const cl::Device &device,
                                                        std::size_t width, std::size_t height, std::size_t depth,
                                                        std::size_t &rowPitch, std::size_t &slicePitch)
{
    image = cl::Image3D(context, CL_MEM_READ_ONLY,
                        cl::ImageFormat(CL_R, CL_SIGNED_INT32),
                        width, height, depth);
    cl::size_t<3> origin;
    origin[0] = 0; origin[1] = 0; origin[2] = 0;
    cl::size_t<3> region;
    region[0] = width;
    region[1] = height;
    region[2] = depth;
    mapping.reset(new CLH::ImageMapping(image, device, CL_MAP_WRITE, origin, region, &rowPitch, &slicePitch));
    // CL specifies pitches in bytes, but we need it in elements
    assert(rowPitch % sizeof(command_type) == 0);
    assert(slicePitch % sizeof(command_type) == 0);
    rowPitch /= sizeof(command_type);
    slicePitch /= sizeof(command_type);
    return static_cast<command_type *>(mapping->get());
}
#endif

SplatTree::command_type *SplatTreeCL::allocateCommands(std::size_t size)
{
    return commands.allocate(context, device, size);
}

SplatTree::command_type *SplatTreeCL::allocateStart(
    std::size_t width, std::size_t height, std::size_t depth,
    std::size_t &rowPitch, std::size_t &slicePitch)
{
#if USE_IMAGES
    return start.allocate(context, device, width, height, depth, rowPitch, slicePitch);
#else
    rowPitch = width;
    slicePitch = width * height;
    return start.allocate(context, device, width * height * depth);
#endif
}

SplatTreeCL::SplatTreeCL(const cl::Context &context, const cl::Device &device,
                         const std::vector<Splat> &splats, const Grid &grid)
    : SplatTree(splats, grid), context(context), device(device)
{
    for (unsigned int i = 0; i < 3; i++)
        dims[i] = grid.numVertices(i);
    code_type size = *std::max_element(dims, dims + 3);
    unsigned int maxLevel = 0;
    while ((1U << maxLevel) < size)
        maxLevel++;

    splats = cl::Buffer(context, CL_MEM_READ_WRITE, splats.size() * sizeof(Splat));
    start = cl::Buffer(context, CL_MEM_READ_WRITE, 

    // copy splats to the GPU
    // writeEntries
    // sort
    // countLevel
    // scan
    // writeLevel
    // transformSplats
}
