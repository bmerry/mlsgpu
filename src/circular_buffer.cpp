/**
 * @file
 *
 * Thread-safe circular buffer for pipelining variable-sized data chunks.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif

#include <cstddef>
#include <utility>
#include <algorithm>
#include <cassert>
#include <limits>
#include <string>
#include <boost/noncopyable.hpp>
#include <boost/thread/locks.hpp>
#include "allocator.h"
#include "circular_buffer.h"
#include "errors.h"

std::pair<void *, std::size_t> CircularBuffer::allocate(
    std::size_t elementSize, std::tr1::uintmax_t maxElements)
{
    MLSGPU_ASSERT(elementSize < bufferSize / 2, std::invalid_argument);
    MLSGPU_ASSERT(maxElements > 0, std::invalid_argument);

    boost::unique_lock<boost::mutex> lock(mutex);

    while (bufferHead > bufferTail && bufferHead - bufferTail <= elementSize)
        spaceCondition.wait(lock);
    // At this point we either have enough space, or the free space wraps around
    if (bufferSize - bufferTail < elementSize)
    {
        // Not enough space, so we can be sure the free space wraps around
        bufferTail = 0; // no room at end, so waste that region and start at the beginning
        while (bufferHead <= elementSize)
            spaceCondition.wait(lock);
    }

    std::size_t bytes;
    if (bufferHead > bufferTail)
        bytes = bufferHead - bufferTail - 1;
    else
    {
        bytes = bufferSize - bufferTail;
        if (bufferHead == 0)
            bytes--; // must not completely fill the buffer
    }
    bytes = std::min(bytes, bufferSize / 2);

    std::size_t elements = bytes / elementSize;
    if (elements > maxElements)
        elements = maxElements;
    bytes = elements * elementSize;

    assert(bytes >= elementSize);

    std::pair<void *, std::size_t> ans(buffer + bufferTail, elements);
    bufferTail += bytes;
    if (bufferTail == bufferSize)
        bufferTail = 0;
    assert(bufferTail <= bufferSize);
    return ans;
}

void CircularBuffer::free(void *ptr, std::size_t elementSize, std::size_t elements)
{
    MLSGPU_ASSERT(elements <= std::numeric_limits<std::size_t>::max() / elementSize, std::out_of_range);
    free(ptr, elementSize * elements);
}

void CircularBuffer::free(void *ptr, std::size_t bytes)
{
    MLSGPU_ASSERT(ptr != NULL, std::invalid_argument);
    MLSGPU_ASSERT(bytes > 0, std::invalid_argument);

    boost::lock_guard<boost::mutex> lock(mutex);
    bufferHead = ((char *) ptr - buffer) + bytes;
    if (bufferHead == bufferSize)
        bufferHead = 0;
    // If the buffer is empty, we can continue wherever we like, and
    // going back to the beginning will minimize fragmentation and hopefully
    // also cache pollution.
    if (bufferHead == bufferTail)
        bufferHead = bufferTail = 0;
    spaceCondition.notify_one();
    assert(bufferHead < bufferSize);
}

CircularBuffer::CircularBuffer(const std::string &name, std::size_t size)
    : allocator(Statistics::makeAllocator<Statistics::Allocator<std::allocator<char> > >(name)),
    buffer(NULL), bufferHead(0), bufferTail(0), bufferSize(size)
{
    MLSGPU_ASSERT(size >= 2, std::length_error);
    buffer = allocator.allocate(size);
}

CircularBuffer::~CircularBuffer()
{
    allocator.deallocate(buffer, bufferSize);
}
