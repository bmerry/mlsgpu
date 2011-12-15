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

SplatTreeCL::SplatTreeCL(const cl::Context &context, std::size_t maxLevels, std::size_t maxSplats)
    : splats(context, CL_MEM_READ_WRITE, sizeof(Splat) * maxSplats),
    start(context, CL_MEM_READ_WRITE, sizeof(cl_uint) << (3 * maxLevels) / 7),
    commands(context, CL_MEM_READ_WRITE, maxSplats * 16 * sizeof(cl_int)),
    entryKeys(context, CL_MEM_READ_WRITE, maxSplats * 8 * sizeof(cl_uint)),
    entryValues(context, CL_MEM_READ_WRITE, maxSplats * 8 * sizeof(cl_uint)),
    maxSplats(0), numSplats(0), levelOffsets(maxLevels)
{
    // TODO: validate that maxSplats and maxLevels aren't too large before attempting to do allocations
    std::size_t pos = 0;
    for (std::size_t i = 0; i < maxLevels; i++)
    {
        levelOffsets[i] = pos;
        pos += 1U << (3 * (maxLevels - i - 1));
    }
}

void SplatTreeCL::enqueueBuild(
    const cl::CommandQueue &queue,
    const Splat *splats, std::size_t numSplats,
    const Grid &grid, bool blockingCopy,
    const std::vector<cl::Event> *events,
    cl::Event *uploadEvent, cl::Event *event)
{
    if (numSplats > maxSplats)
    {
        throw std::length_error("Too many splats");
    }

    cl::Event myUploadEvent;
    queue.enqueueWriteBuffer(this->splats, CL_FALSE, 0, numSplats * sizeof(Splat), splats, events, &myUploadEvent);
    // copy splats to the GPU
    // writeEntries
    // sort
    // countLevel
    // scan
    // writeLevel
    // transformSplats
    if (uploadEvent != NULL)
    {
        *uploadEvent = myUploadEvent;
    }
    if (blockingCopy)
        myUploadEvent.wait();
}
