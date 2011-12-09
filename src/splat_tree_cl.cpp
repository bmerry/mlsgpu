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

SplatTree::size_type *SplatTreeCL::Buffer::allocate(const cl::Context &context, const cl::Device &device, size_type size)
{
    buffer = cl::Buffer(context, CL_MEM_READ_ONLY, size * sizeof(size_type));
    mapping.reset(new CLH::BufferMapping(buffer, device, CL_MAP_WRITE, 0, size * sizeof(size_type)));
    return static_cast<size_type *>(mapping->get());
}

SplatTree::size_type *SplatTreeCL::allocateIds(size_type size)
{
    return ids.allocate(context, device, size);
}

SplatTree::size_type *SplatTreeCL::allocateStart(size_type size)
{
    return start.allocate(context, device, size);
}

SplatTree::size_type *SplatTreeCL::allocateLevelStart(size_type size)
{
    (void) size; // prevent compiler warning
    return NULL;
}

SplatTreeCL::SplatTreeCL(const cl::Context &context, const cl::Device &device,
                         const std::vector<Splat> &splats, const Grid &grid)
    : SplatTree(splats, grid), context(context), device(device)
{
    initialize();
    start.mapping.reset();
    ids.mapping.reset();

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
    const cl::ImageFormat format(CL_R, CL_UNSIGNED_INT32);
    shuffle = cl::Image2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, format, numCoords, 3, 0, &image[0]);
}
