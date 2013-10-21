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
 * Data structure for passing work between threads.
 */

#ifndef WORK_QUEUE_H
#define WORK_QUEUE_H

#if HAVE_CONFIG_H
# include <config.h>
#endif

#include <cstddef>
#include <stdexcept>
#include <map>
#include <queue>
#include <boost/thread/mutex.hpp>
#include <boost/thread/locks.hpp>
#include <boost/thread/condition_variable.hpp>
#include <boost/smart_ptr/scoped_array.hpp>
#include <boost/noncopyable.hpp>
#include "errors.h"

/**
 * Thread-safe queue, supporting multiple producers and multiple consumers. The
 * capacity is unbounded. It can additionally be "stopped", which will cause any
 * requests after the queue drains to return immediately with a
 * default-constructed value. It is the user's responsibility to use a type for
 * which this can be distinguished from real data.
 *
 * It is a requirement that the assignment operator for the value type does
 * not throw. In particular, containers should not be used, or should be
 * passed by smart pointer.
 *
 * @param ValueType   The type of data stored in the queue.
 */
template<typename ValueType>
class WorkQueue : public boost::noncopyable
{
public:
    typedef ValueType value_type;
    typedef std::size_t size_type;

    /**
     * Add an item to the queue. This will never block.
     *
     * @pre The queue is not stopped.
     */
    void push(const value_type &item);

    /**
     * Extract an item from the queue. This will block if the queue is empty.
     * Note that the item is returned by value. This is necessary since the
     * storage for the slot in the queue is immediately made available to be
     * overwritten.
     *
     * If the queue has been marked stopped and there is no more data in the
     * queue, it will return a default-constructed value.
     */
    value_type pop();

    /**
     * Determine whether calling @ref pop will block. In a multithreaded
     * environment the result should of course be considered immediately stale.
     * Note that if the queue has been stopped then this will return @c false.
     */
    bool empty();

    /**
     * Indicate that there will be no more data added. It is not safe to call
     * this simultaneously with @ref push.
     */
    void stop();

    /**
     * Indicate that there will be new data added. This should only be called
     * when there is only a single thread active. It is not necessary to call
     * it initially, as this is the initial state.
     */
    void start();

    /**
     * Constructor.
     */
    WorkQueue();

private:
    std::queue<value_type> queue;
    bool stopped;
    boost::mutex mutex;
    boost::condition_variable dataCondition;
    // TODO account for the memory
};


template<typename ValueType>
WorkQueue<ValueType>::WorkQueue()
    : stopped(false)
{
}

template<typename ValueType>
void WorkQueue<ValueType>::push(const ValueType &value)
{
    boost::lock_guard<boost::mutex> lock(mutex);
    MLSGPU_ASSERT(!stopped, state_error);
    queue.push(value);
    dataCondition.notify_one();
}

template<typename ValueType>
ValueType WorkQueue<ValueType>::pop()
{
    boost::unique_lock<boost::mutex> lock(mutex);
    while (!stopped && queue.empty())
        dataCondition.wait(lock);
    if (queue.empty())
        return value_type();
    else
    {
        value_type ans = queue.front();
        queue.pop();
        return ans;
    }
}

template<typename ValueType>
bool WorkQueue<ValueType>::empty()
{
    boost::unique_lock<boost::mutex> lock(mutex);
    return !stopped && queue.empty();
}

template<typename ValueType>
void WorkQueue<ValueType>::start()
{
    boost::lock_guard<boost::mutex> lock(mutex);
    stopped = false;
}

template<typename ValueType>
void WorkQueue<ValueType>::stop()
{
    boost::lock_guard<boost::mutex> lock(mutex);
    stopped = true;
    dataCondition.notify_all(); // wake up any consumers waiting on an empty queue
}

#endif /* !WORK_QUEUE_H */
