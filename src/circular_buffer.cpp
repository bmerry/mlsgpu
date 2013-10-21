/*
 * mlsgpu: surface reconstruction from point clouds
 * Copyright (C) 2013  University of Cape Town
 *
 * This file is part of mlsgpu.
 *
 * mlsgpu is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

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
    : point()
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
    action.setValue(n);

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
