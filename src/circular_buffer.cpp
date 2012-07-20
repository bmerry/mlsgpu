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

    /* This condition is slightly stronger than necessary: in the wraparound
     * case, it's sufficient if there are exactly enough bytes at either the
     * front or the back, but I think this is sufficient to guarantee forward
     * progress.
     */
    while ((bufferHead > bufferTail && bufferHead - bufferTail <= bytes)
           || (bufferHead <= bufferTail && bufferSize - bufferTail <= bytes && bufferHead <= bytes))
        spaceCondition.wait(lock);
    if (bufferHead <= bufferTail && bufferSize - bufferTail <= bytes)
    {
        // no room at the end, so waste that space and start at the front
        bufferTail = 0;
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
    /* If the buffer is empty, we can continue wherever we like, and
     * going back to the beginning will minimize fragmentation and hopefully
     * also cache pollution. More importantly, it guarantees that allocate
     * can make forward progress when asked for a chunk bigger than half
     * the buffer which is guaranteed to overlap the current tail.
     */
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
