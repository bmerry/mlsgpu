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

void *CircularBuffer::allocate(std::size_t bytes)
{
    MLSGPU_ASSERT(bytes > 0, std::invalid_argument);
    MLSGPU_ASSERT(bytes < bufferSize, std::out_of_range);

    boost::unique_lock<boost::mutex> lock(mutex);

    while (bufferHead > bufferTail && bufferHead - bufferTail <= bytes)
        spaceCondition.wait(lock);
    // At this point we either have enough space, or the free space wraps around
    if (bufferSize - bufferTail < bytes)
    {
        // Not enough space, so we can be sure the free space wraps around
        bufferTail = 0; // no room at end, so waste that region and start at the beginning
        while (bufferHead <= bytes)
            spaceCondition.wait(lock);
    }

    void *ans = buffer + bufferTail;
    bufferTail += bytes;
    if (bufferTail == bufferSize)
        bufferTail = 0;
    assert(bufferTail < bufferSize);
    return ans;
}

void *CircularBuffer::allocate(std::size_t elementSize, std::size_t elements)
{
    MLSGPU_ASSERT(elementSize > 0, std::invalid_argument);
    MLSGPU_ASSERT(elements <= (bufferSize - 1) / elementSize, std::out_of_range);
    return allocate(elementSize * elements);
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
