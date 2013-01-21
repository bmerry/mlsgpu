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
#include "statistics.h"
#include "allocator.h"
#include "errors.h"
#include "timeplot.h"
#include "circular_buffer.h"

std::size_t CircularBufferBase::Allocation::get() const
{
    return *point;
}

CircularBufferBase::Allocation::Allocation(
    Statistics::Container::list<std::size_t>::iterator point)
    : point(point)
{
}

CircularBufferBase::Allocation::Allocation()
    : point(0)
{
}

CircularBufferBase::CircularBufferBase(const std::string &name, std::size_t size)
    : bufferSize(size), firstFree(0), allocPoints(name)
{
    MLSGPU_ASSERT(size > 0, std::invalid_argument);
}

CircularBufferBase::Allocation CircularBufferBase::allocate(
    Timeplot::Worker &tworker, std::size_t n,
    Statistics::Variable *stat)
{
    MLSGPU_ASSERT(n > 0, std::invalid_argument);
    MLSGPU_ASSERT(n <= bufferSize, std::out_of_range);

    Timeplot::Action action("get", tworker, stat);

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

std::size_t CircularBufferBase::unallocated()
{
    if (allocPoints.empty())
        return bufferSize;
    else if (allocPoints.front() >= firstFree)
        return allocPoints.front() - firstFree;
    else
        return bufferSize - firstFree + allocPoints.front();
}

void *CircularBuffer::Allocation::get() const
{
    return ptr;
}

CircularBuffer::Allocation CircularBuffer::allocate(
    Timeplot::Worker &tworker, std::size_t bytes,
    Statistics::Variable *stat)
{
    Allocation ans;
    ans.base = CircularBufferBase::allocate(tworker, bytes, stat);
    ans.ptr = buffer + ans.base.get();
    return ans;
}

CircularBuffer::Allocation CircularBuffer::allocate(
    Timeplot::Worker &tworker,
    std::size_t elementSize, std::size_t elements,
    Statistics::Variable *stat)
{
    MLSGPU_ASSERT(elementSize > 0, std::invalid_argument);
    MLSGPU_ASSERT(elements <= size() / elementSize, std::out_of_range);
    return allocate(tworker, elementSize * elements, stat);
}

void CircularBuffer::free(const Allocation &alloc)
{
    CircularBufferBase::free(alloc.base);
}

CircularBuffer::CircularBuffer(const std::string &name, std::size_t size)
    :
    CircularBufferBase(name, size),
    allocator(Statistics::makeAllocator<Statistics::Allocator<std::allocator<char> > >(name)),
    buffer(NULL)
{
    buffer = allocator.allocate(size);
}

CircularBuffer::~CircularBuffer()
{
    allocator.deallocate(buffer, size());
}
