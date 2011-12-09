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

SplatTree::command_type *SplatTreeCL::allocateCommands(std::size_t size)
{
    return commands.allocate(context, device, size);
}

SplatTree::command_type *SplatTreeCL::allocateStart(
    std::size_t width, std::size_t height, std::size_t depth,
    std::size_t &rowPitch, std::size_t &slicePitch)
{
    return start.allocate(context, device, width, height, depth, rowPitch, slicePitch);
}

SplatTreeCL::SplatTreeCL(const cl::Context &context, const cl::Device &device,
                         const std::vector<Splat> &splats, const Grid &grid)
    : SplatTree(splats, grid), context(context), device(device)
{
    initialize();
    start.mapping.reset();
    commands.mapping.reset();

    // Prepare the shuffle texture.
    const unsigned int numCoords = 1U << (getNumLevels() - 1);
    std::vector<cl_uint> image(numCoords * 3);
    unsigned int cur = 0;
    const unsigned int mask = ((1U << 30) - 1) / 7;  // 100100100..001 in binary
    for (unsigned int i = 0; i < numCoords; i++)
    {
        // Or in ~mask makes all the intermediate bits 1's, so that carries
        // ripple through to the next interesting bit. Then we take it away
        // again.
        image[i] = cur;
        image[i + numCoords] = cur << 1;
        image[i + 2 * numCoords] = cur << 2;
        cur = ((cur | ~mask) + 1) & mask;
    }
}
