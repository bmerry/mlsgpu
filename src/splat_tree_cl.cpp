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

SplatTree::size_type *SplatTreeCL::Buffer::allocate(const cl::Context &context, size_type size)
{
    buffer = cl::Buffer(context, CL_MEM_READ_ONLY, size * sizeof(size_type));
    mapping.reset(new CLH::BufferMapping(buffer, CL_MAP_WRITE, 0, size * sizeof(size_type)));
    return static_cast<size_type *>(mapping->get());
}

SplatTree::size_type *SplatTreeCL::allocateIds(size_type size)
{
    return ids.allocate(context, size);
}

SplatTree::size_type *SplatTreeCL::allocateStart(size_type size)
{
    return start.allocate(context, size);
}

SplatTree::size_type *SplatTreeCL::allocateLevelStart(size_type size)
{
    return levelStart.allocate(context, size);
}

SplatTreeCL::SplatTreeCL(const cl::Context &context, const std::vector<Splat> &splats, const Grid &grid)
    : SplatTree(splats, grid), context(context)
{
    initialize();
    levelStart.mapping.reset();
    start.mapping.reset();
    ids.mapping.reset();
}
