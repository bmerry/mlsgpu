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
#include "splat_tree_cl.h"
#include "splat.h"
#include "grid.h"
#include "clh.h"

SplatTreeCL::SplatTreeCL(const cl::Context &context, std::size_t maxLevels, std::size_t maxSplats)
    : maxSplats(maxSplats), maxLevels(maxLevels), numSplats(0),
    sort(context, context.getInfo<CL_CONTEXT_DEVICES>()[0], clcpp::TYPE_UINT, clcpp::TYPE_INT),
    scan(context, context.getInfo<CL_CONTEXT_DEVICES>()[0], clcpp::TYPE_UINT)
{
    if (maxSplats > std::size_t(std::numeric_limits<command_type>::max() / 16))
        throw std::out_of_range("maxSplats is too large");
    if (maxLevels >= std::numeric_limits<code_type>::digits / 3)
        throw std::out_of_range("maxLevels is too large");

    std::size_t maxStart = (std::size_t(1) << (3 * maxLevels)) / 7;
    splats = cl::Buffer(context, CL_MEM_READ_WRITE, maxSplats * sizeof(Splat));
    start = cl::Buffer(context, CL_MEM_READ_WRITE, maxStart * sizeof(command_type));
    commands = cl::Buffer(context, CL_MEM_READ_WRITE, maxSplats * 16 * sizeof(command_type));
    commandMap = cl::Buffer(context, CL_MEM_READ_WRITE, maxSplats * 8 * sizeof(command_type));
    entryKeys = cl::Buffer(context, CL_MEM_READ_WRITE, (maxSplats * 8) * sizeof(code_type));
    entryValues = cl::Buffer(context, CL_MEM_READ_WRITE, (maxSplats * 8) * sizeof(command_type));

    std::map<std::string, std::string> defines;
    defines["MAX_LEVELS"] = boost::lexical_cast<std::string>(maxLevels);
    program = CLH::build(context, "kernels/octree.cl", defines);
    writeEntriesKernel = cl::Kernel(program, "writeEntries");
    countCommandsKernel = cl::Kernel(program, "countCommands");
    writeSplatIdsKernel = cl::Kernel(program, "writeSplatIds");
    fillKernel = cl::Kernel(program, "fill");
    writeStartKernel = cl::Kernel(program, "writeStart");
}

SplatTreeCL::code_type SplatTreeCL::keyOffset(unsigned int level)
{
    return 0x80000000U >> level;
}

void SplatTreeCL::enqueueWriteEntries(
    const cl::CommandQueue &queue,
    const cl::Buffer &keys,
    const cl::Buffer &values,
    const cl::Buffer &splats,
    command_type numSplats,
    const Grid &grid,
    std::size_t numLevels,
    std::vector<cl::Event> *events,
    cl::Event *event)
{
    const float zeros[3] = {0.0f, 0.0f, 0.0f};
    cl_float3 scale, bias;
    cl_float3 invScale, invBias;
    grid.getVertex(0, 0, 0, bias.s);
    grid.worldToVertex(zeros, invBias.s);
    for (unsigned int i = 0; i < 3; i++)
    {
        scale.s[i] = grid.getDirection(i)[i];
        invScale.s[i] = 1.0f / scale.s[i];
    }

    writeEntriesKernel.setArg(0, keys);
    writeEntriesKernel.setArg(1, values);
    writeEntriesKernel.setArg(2, splats);
    writeEntriesKernel.setArg(3, scale);
    writeEntriesKernel.setArg(4, bias);
    writeEntriesKernel.setArg(5, invScale);
    writeEntriesKernel.setArg(6, invBias);
    writeEntriesKernel.setArg(7, cl::__local(sizeof(code_type) * numLevels));
    writeEntriesKernel.setArg(8, (cl_uint) numLevels);

    queue.enqueueNDRangeKernel(writeEntriesKernel,
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
    queue.enqueueNDRangeKernel(countCommandsKernel,
                               cl::NullRange,
                               cl::NDRange(numKeys - 1),
                               cl::NullRange,
                               events, event);
}

void SplatTreeCL::enqueueWriteSplatIds(
    const cl::CommandQueue &queue,
    const cl::Buffer &commands,
    const cl::Buffer &commandMap,
    const cl::Buffer &splatIds,
    command_type numEntries,
    std::vector<cl::Event> *events,
    cl::Event *event)
{
    writeSplatIdsKernel.setArg(0, commands);
    writeSplatIdsKernel.setArg(1, commandMap);
    writeSplatIdsKernel.setArg(2, splatIds);
    queue.enqueueNDRangeKernel(writeSplatIdsKernel,
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
    queue.enqueueNDRangeKernel(fillKernel,
                               cl::NDRange(offset),
                               cl::NDRange(elements),
                               cl::NullRange,
                               events, event);
}

void SplatTreeCL::enqueueWriteStart(
    const cl::CommandQueue &queue,
    const cl::Buffer &start,
    const cl::Buffer &commands,
    const cl::Buffer &commandMap,
    const cl::Buffer &keys,
    code_type numCodes,
    command_type keysLen,
    code_type curOffset,
    code_type prevOffset,
    std::vector<cl::Event> *events,
    cl::Event *event)
{
    const std::size_t M = 255;
    writeStartKernel.setArg(0, start);
    writeStartKernel.setArg(1, commands);
    writeStartKernel.setArg(2, commandMap);
    writeStartKernel.setArg(3, keys);
    writeStartKernel.setArg(4, numCodes);
    writeStartKernel.setArg(5, keysLen);
    writeStartKernel.setArg(6, curOffset);
    writeStartKernel.setArg(7, prevOffset);
    writeStartKernel.setArg(8, cl::__local((M + 1) * sizeof(command_type)));

    unsigned int groups = (numCodes + M - 1) / M;
    queue.enqueueNDRangeKernel(writeStartKernel,
                               cl::NullRange,
                               cl::NDRange(groups * (M + 1)),
                               cl::NDRange(M + 1),
                               events, event);
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
    unsigned int levels = 1;
    while (grid.numVertices(0) > (1U << (levels - 1))
           || grid.numVertices(1) > (1U << (levels - 1))
           || grid.numVertices(2) > (1U << (levels - 1)))
        levels++;
    if (levels > maxLevels)
    {
        throw std::length_error("Grid is too large");
    }

    this->grid = grid;
    this->numSplats = numSplats;
    std::size_t pos = 0;
    levelOffsets.resize(levels);
    for (std::size_t i = 0; i < levels; i++)
    {
        levelOffsets[i] = pos;
        pos += 1U << (3 * (levels - i - 1));
    }

    std::vector<cl::Event> wait(1);

    // Copy splats to the GPU
    cl::Event myUploadEvent, writeEntriesEvent, sortEvent, countEvent, scanEvent,
        writeSplatIdsEvent, levelEvent;
    queue.enqueueWriteBuffer(this->splats, CL_FALSE, 0, numSplats * sizeof(Splat), splats, events, &myUploadEvent);
    queue.flush(); // Start the copy going while we do remaining queuing.

    const std::size_t numEntries = numSplats * 8;
    wait[0] = myUploadEvent;
    enqueueWriteEntries(queue, entryKeys, entryValues, this->splats, numSplats, grid, levels, &wait, &writeEntriesEvent);
    wait[0] = writeEntriesEvent;
    sort.enqueue(queue, entryKeys, entryValues, numEntries, &wait, &sortEvent);
    wait[0] = sortEvent;
    enqueueCountCommands(queue, commandMap, entryKeys, numEntries, &wait, &countEvent);
    wait[0] = countEvent;
    scan.enqueue(queue, commandMap, numEntries, &wait, &scanEvent);
    wait[0] = scanEvent;
    enqueueWriteSplatIds(queue, commands, commandMap, entryValues, numEntries, &wait, &writeSplatIdsEvent);
    // don't update wait[0]: writeSplatIds and writeStart are independent

    // Prime level "-1" with a terminator
    // TODO: use enqueueFillBuffer if CL 1.2 is available
    enqueueFill(queue, start, levelOffsets.back(), 1, (command_type) -1, &wait, &levelEvent);
    for (int i = levels - 1; i >= 0; i--)
    {
        wait[0] = levelEvent;
        enqueueWriteStart(queue, start, commands, commandMap, entryKeys,
                          1U << (3 * (levels - i - 1)), numEntries,
                          levelOffsets[i],
                          levelOffsets[std::min(i + 1, int(levels) - 1)],
                          &wait, &levelEvent);
    }

    wait[0] = writeEntriesEvent;

    if (event != NULL)
        *event = levelEvent;
    if (uploadEvent != NULL)
        *uploadEvent = myUploadEvent;
    if (blockingCopy)
        myUploadEvent.wait();
}
