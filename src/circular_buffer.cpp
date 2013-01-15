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
#include <list>
#include <boost/noncopyable.hpp>
#include <boost/thread/locks.hpp>
#include "allocator.h"
#include "circular_buffer.h"
#include "errors.h"

std::size_t CircularBufferBase::Allocation::get() const
{
    return *point;
}

CircularBufferBase::Allocation::Allocation(
    std::list<std::size_t>::iterator point)
    : point(point)
{
}

CircularBufferBase::Allocation::Allocation()
    : point(0)
{
}

CircularBufferBase::CircularBufferBase(std::size_t size)
    : bufferSize(size), firstFree(0)
{
    MLSGPU_ASSERT(size > 0, std::invalid_argument);
}

CircularBufferBase::Allocation CircularBufferBase::allocate(std::size_t n)
{
    MLSGPU_ASSERT(n > 0, std::invalid_argument);
    MLSGPU_ASSERT(n <= bufferSize, std::out_of_range);

    boost::lock_guard<boost::mutex> allocLock(allocMutex);
    boost::unique_lock<boost::mutex> lock(mutex);
    std::size_t pos = bufferSize; // sentinel invalid value

retry:
    if (allocPoints.empty())
        pos = 0;
    else
    {
        std::size_t end = allocPoints.front();
        if (firstFree <= end)
        {
            if (end - firstFree >= n)
                pos = firstFree;
        }
        else
        {
            if (bufferSize - firstFree >= n)
                pos = firstFree;
            else if (end >= n)
                pos = 0;
        }
    }

    if (pos == bufferSize)
    {
        spaceCondition.wait(lock);
        goto retry;
    }

    firstFree = pos + n;
    return Allocation(allocPoints.insert(allocPoints.end(), pos));
}

void CircularBufferBase::free(const Allocation &alloc)
{
    boost::lock_guard<boost::mutex> lock(mutex);
    bool first = allocPoints.begin() == alloc.point;
    allocPoints.erase(alloc.point);
    if (first)
        spaceCondition.notify_one();
}

std::size_t CircularBufferBase::size() const
{
    return bufferSize;
}

void *CircularBuffer::Allocation::get() const
{
    return ptr;
}

CircularBuffer::Allocation CircularBuffer::allocate(std::size_t bytes)
{
    Allocation ans;
    ans.base = CircularBufferBase::allocate(bytes);
    ans.ptr = buffer + ans.base.get();
    return ans;
}

CircularBuffer::Allocation CircularBuffer::allocate(std::size_t elementSize, std::size_t elements)
{
    MLSGPU_ASSERT(elementSize > 0, std::invalid_argument);
    MLSGPU_ASSERT(elements <= (size() - 1) / elementSize, std::out_of_range);
    return allocate(elementSize * elements);
}

void CircularBuffer::free(const Allocation &alloc)
{
    CircularBufferBase::free(alloc.base);
}

CircularBuffer::CircularBuffer(const std::string &name, std::size_t size)
    :
    CircularBufferBase(size),
    allocator(Statistics::makeAllocator<Statistics::Allocator<std::allocator<char> > >(name)),
    buffer(NULL)
{
    MLSGPU_ASSERT(size >= 1, std::length_error);
    buffer = allocator.allocate(size);
}

CircularBuffer::~CircularBuffer()
{
    allocator.deallocate(buffer, size());
}
