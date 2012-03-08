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
#include <tr1/cstdint>
#include <limits>
#include <vector>
#include <tr1/cstdint>
#include "splat_tree_cl.h"
#include "splat.h"
#include "grid.h"
#include "clh.h"
#include "errors.h"

/* Definitions of constants */
const std::size_t SplatTreeCL::MAX_LEVELS;
const std::size_t SplatTreeCL::MAX_SPLATS;

bool SplatTreeCL::validateDevice(const cl::Device &device)
{
    if (!device.getInfo<CL_DEVICE_IMAGE_SUPPORT>())
        return false;
    return true;
}

CLH::ResourceUsage SplatTreeCL::resourceUsage(
    const cl::Device &device, const std::size_t maxLevels, const std::size_t maxSplats)
{
    /* Not currently used, although it should be to determine constant overheads in
     * the clogs primitives.
     */
    (void) device;

    MLSGPU_ASSERT(1 <= maxLevels && maxLevels <= MAX_LEVELS, std::length_error);
    MLSGPU_ASSERT(1 <= maxSplats && maxSplats <= MAX_SPLATS, std::length_error);
    const std::tr1::uint64_t maxStart = (std::tr1::uint64_t(1) << (3 * maxLevels)) / 7;

    CLH::ResourceUsage ans;

    // Keep this up to date with the actual allocations below

    // splats = cl::Buffer(context, CL_MEM_READ_WRITE, maxSplats * sizeof(Splat));
    ans.addBuffer(maxSplats * sizeof(Splat));
    // start = cl::Buffer(context, CL_MEM_READ_WRITE, maxStart * sizeof(command_type));
    ans.addBuffer(maxStart * sizeof(command_type));
    // jumpPos = cl::Buffer(context, CL_MEM_READ_WRITE, maxStart * sizeof(command_type));
    ans.addBuffer(maxStart * sizeof(command_type));
    // commands = cl::Buffer(context, CL_MEM_READ_WRITE, maxSplats * 16 * sizeof(command_type));
    ans.addBuffer(maxSplats * 16 * sizeof(command_type));
    // commandMap = cl::Buffer(context, CL_MEM_READ_WRITE, maxSplats * 8 * sizeof(command_type));
    ans.addBuffer(maxSplats * 8 * sizeof(command_type));
    // entryKeys = cl::Buffer(context, CL_MEM_READ_WRITE, (maxSplats * 8) * sizeof(code_type));
    ans.addBuffer((maxSplats * 8) * sizeof(code_type));
    // entryValues = cl::Buffer(context, CL_MEM_READ_WRITE, (maxSplats * 8) * sizeof(command_type));
    ans.addBuffer((maxSplats * 8) * sizeof(command_type));

    // Temporary storage for the sort
    // tmpKeys = cl::Buffer(context, CL_MEM_READ_WRITE, (maxSplats * 8) * sizeof(code_type));
    // tmpValues = cl::Buffer(context, CL_MEM_READ_WRITE, (maxSplats * 8) * sizeof(command_type));
    ans.addBuffer((maxSplats * 8) * sizeof(code_type));
    ans.addBuffer((maxSplats * 8) * sizeof(command_type));

    // TODO: add in constant overheads for the scan and sort primitives

    return ans;
}

SplatTreeCL::SplatTreeCL(const cl::Context &context, std::size_t maxLevels, std::size_t maxSplats)
    : maxSplats(maxSplats), maxLevels(maxLevels), numSplats(0),
    sort(context, context.getInfo<CL_CONTEXT_DEVICES>()[0], clogs::TYPE_UINT, clogs::TYPE_INT),
    scan(context, context.getInfo<CL_CONTEXT_DEVICES>()[0], clogs::TYPE_UINT)
{
    MLSGPU_ASSERT(1 <= maxSplats && maxSplats <= MAX_SPLATS, std::length_error);
    MLSGPU_ASSERT(1 <= maxLevels && maxLevels <= MAX_LEVELS, std::length_error);

    std::size_t maxStart = (std::tr1::uint64_t(1) << (3 * maxLevels)) / 7;
    // If this section is modified, remember to update deviceMemory above
    splats = cl::Buffer(context, CL_MEM_READ_WRITE, maxSplats * sizeof(Splat));
    start = cl::Buffer(context, CL_MEM_READ_WRITE, maxStart * sizeof(command_type));
    jumpPos = cl::Buffer(context, CL_MEM_READ_WRITE, maxStart * sizeof(command_type));
    commands = cl::Buffer(context, CL_MEM_READ_WRITE, maxSplats * 16 * sizeof(command_type));
    commandMap = cl::Buffer(context, CL_MEM_READ_WRITE, maxSplats * 8 * sizeof(command_type));
    entryKeys = cl::Buffer(context, CL_MEM_READ_WRITE, (maxSplats * 8) * sizeof(code_type));
    entryValues = cl::Buffer(context, CL_MEM_READ_WRITE, (maxSplats * 8) * sizeof(command_type));
    tmpKeys = cl::Buffer(context, CL_MEM_READ_WRITE, (maxSplats * 8) * sizeof(code_type));
    tmpValues = cl::Buffer(context, CL_MEM_READ_WRITE, (maxSplats * 8) * sizeof(command_type));
    sort.setTemporaryBuffers(tmpKeys, tmpValues);

    std::map<std::string, std::string> defines;
    defines["MAX_LEVELS"] = boost::lexical_cast<std::string>(maxLevels);
    program = CLH::build(context, "kernels/octree.cl", defines);
    writeEntriesKernel = cl::Kernel(program, "writeEntries");
    countCommandsKernel = cl::Kernel(program, "countCommands");
    writeSplatIdsKernel = cl::Kernel(program, "writeSplatIds");
    fillKernel = cl::Kernel(program, "fill");
    writeStartKernel = cl::Kernel(program, "writeStart");
    writeStartTopKernel = cl::Kernel(program, "writeStartTop");
}

void SplatTreeCL::enqueueWriteEntries(
    const cl::CommandQueue &queue,
    const cl::Buffer &keys,
    const cl::Buffer &values,
    const cl::Buffer &splats,
    command_type numSplats,
    const Grid::difference_type offset[3],
    std::size_t minShift,
    std::size_t maxShift,
    std::vector<cl::Event> *events,
    cl::Event *event)
{
    cl_int3 offset3 = {{ offset[0], offset[1], offset[2] }};

    writeEntriesKernel.setArg(0, keys);
    writeEntriesKernel.setArg(1, values);
    writeEntriesKernel.setArg(2, splats);
    writeEntriesKernel.setArg(3, offset3);
    writeEntriesKernel.setArg(4, cl::__local(sizeof(code_type) * (maxShift + 1)));
    writeEntriesKernel.setArg(5, (cl_uint) minShift);
    writeEntriesKernel.setArg(6, (cl_uint) maxShift);

    CLH::enqueueNDRangeKernel(queue,
                              writeEntriesKernel,
                              cl::NullRange,
                              cl::NDRange(numSplats),
                              cl::NullRange,
                              events, event);
}

void SplatTreeCL::enqueueCountCommands(
    const cl::CommandQueue &queue,
    const cl::Buffer &indicator,
    const cl::Buffer &keys,
    command_type numKeys,
    std::vector<cl::Event> *events,
    cl::Event *event)
{
    countCommandsKernel.setArg(0, indicator);
    countCommandsKernel.setArg(1, keys);
    CLH::enqueueNDRangeKernel(queue,
                              countCommandsKernel,
                              cl::NullRange,
                              cl::NDRange(numKeys - 1),
                              cl::NullRange,
                              events, event);
}

void SplatTreeCL::enqueueWriteSplatIds(
    const cl::CommandQueue &queue,
    const cl::Buffer &commands,
    const cl::Buffer &start,
    const cl::Buffer &jumpPos,
    const cl::Buffer &commandMap,
    const cl::Buffer &keys,
    const cl::Buffer &splatIds,
    command_type numEntries,
    std::vector<cl::Event> *events,
    cl::Event *event)
{
    writeSplatIdsKernel.setArg(0, commands);
    writeSplatIdsKernel.setArg(1, start);
    writeSplatIdsKernel.setArg(2, jumpPos);
    writeSplatIdsKernel.setArg(3, commandMap);
    writeSplatIdsKernel.setArg(4, keys);
    writeSplatIdsKernel.setArg(5, splatIds);
    CLH::enqueueNDRangeKernel(queue,
                              writeSplatIdsKernel,
                              cl::NullRange, cl::NDRange(numEntries), cl::NullRange,
                              events, event);
}

void SplatTreeCL::enqueueFill(
    const cl::CommandQueue &queue,
    const cl::Buffer &buffer,
    std::size_t offset,
    std::size_t elements,
    command_type value,
    std::vector<cl::Event> *events,
    cl::Event *event)
{
    fillKernel.setArg(0, buffer);
    fillKernel.setArg(1, value);
    CLH::enqueueNDRangeKernel(queue,
                              fillKernel,
                              cl::NDRange(offset),
                              cl::NDRange(elements),
                              cl::NullRange,
                              events, event);
}

void SplatTreeCL::enqueueWriteStart(
    const cl::CommandQueue &queue,
    const cl::Buffer &start,
    const cl::Buffer &commands,
    const cl::Buffer &jumpPos,
    code_type curOffset,
    bool havePrev,
    code_type prevOffset,
    code_type numCodes,
    std::vector<cl::Event> *events,
    cl::Event *event)
{
    cl::Kernel &kernel = havePrev ? writeStartKernel : writeStartTopKernel;
    kernel.setArg(0, start);
    kernel.setArg(1, commands);
    kernel.setArg(2, jumpPos);
    kernel.setArg(3, curOffset);
    if (havePrev)
        kernel.setArg(4, prevOffset);

    CLH::enqueueNDRangeKernel(queue,
                              kernel,
                              cl::NullRange,
                              cl::NDRange(numCodes),
                              cl::NullRange,
                              events, event);
}


void SplatTreeCL::enqueueBuild(
    const cl::CommandQueue &queue,
    const Splat *splats, std::size_t numSplats,
    const Grid::size_type size[3], const Grid::difference_type offset[3],
    unsigned int subsamplingShift, bool blockingCopy,
    const std::vector<cl::Event> *events,
    cl::Event *uploadEvent, cl::Event *event)
{
    if (numSplats > maxSplats)
    {
        throw std::length_error("Too many splats");
    }
    Grid::size_type maxSize = Grid::size_type(1U) << (maxLevels + subsamplingShift - 1);
    if (size[0] > maxSize || size[1] > maxSize || size[2] > maxSize)
        throw std::length_error("Grid is too large");
    unsigned int maxShift = maxLevels + subsamplingShift - 1;
    unsigned int minShift = std::min(subsamplingShift, maxShift);
    // TODO: this will always construct a full-size octree, even if size[] only
    // specifies a much smaller space. At a minimum, it should be possible to make
    // levelOffsets more compact.

    this->numSplats = numSplats;
    std::size_t pos = 0;
    levelOffsets.resize(maxShift + 1);
    for (std::size_t i = minShift; i <= maxShift; i++)
    {
        levelOffsets[i] = pos;
        pos += 1U << (3 * (maxShift - i));
    }
    std::size_t numStart = pos;

    std::vector<cl::Event> wait(1);

    // Copy splats to the GPU
    cl::Event myUploadEvent, writeEntriesEvent, sortEvent, countEvent, scanEvent,
        writeSplatIdsEvent, levelEvent, fillJumpPosEvent;
    CLH::enqueueWriteBuffer(queue, this->splats, CL_FALSE, 0, numSplats * sizeof(Splat), splats, events, &myUploadEvent);
    queue.flush(); // Start the copy going while we do remaining queuing.

    const std::size_t numEntries = numSplats * 8;
    wait[0] = myUploadEvent;
    enqueueWriteEntries(queue, entryKeys, entryValues, this->splats, numSplats, offset, minShift, maxShift, &wait, &writeEntriesEvent);
    wait[0] = writeEntriesEvent;
    sort.enqueue(queue, entryKeys, entryValues, numEntries, 3 * (maxShift - minShift) + 1, &wait, &sortEvent);
    wait[0] = sortEvent;
    enqueueCountCommands(queue, commandMap, entryKeys, numEntries, &wait, &countEvent);
    wait[0] = countEvent;
    scan.enqueue(queue, commandMap, numEntries, NULL, &wait, &scanEvent);
    wait[0] = scanEvent;
    enqueueFill(queue, jumpPos, 0, numStart, (command_type) -1, &wait, &fillJumpPosEvent);
    wait[0] = fillJumpPosEvent;
    enqueueWriteSplatIds(queue, commands, start, jumpPos, commandMap, entryKeys, entryValues, numEntries, &wait, &writeSplatIdsEvent);
    wait[0] = writeSplatIdsEvent;

    for (int i = maxShift; i >= int(minShift); i--)
    {
        std::size_t levelSize = std::size_t(1) << (3 * (maxShift - i));
        bool havePrev = (i != int(maxShift));
        enqueueWriteStart(queue, start, commands, jumpPos,
                          levelOffsets[i],
                          havePrev,
                          havePrev ? levelOffsets[i + 1] : 0,
                          levelSize,
                          &wait, &levelEvent);
        wait[0] = levelEvent;
    }

    if (event != NULL)
        *event = wait[0];
    if (uploadEvent != NULL)
        *uploadEvent = myUploadEvent;
    if (blockingCopy)
        myUploadEvent.wait();
}
