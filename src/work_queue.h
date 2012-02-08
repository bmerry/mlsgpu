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
#include <boost/thread/mutex.hpp>
#include <boost/thread/locks.hpp>
#include <boost/thread/condition_variable.hpp>
#include <boost/smart_ptr/scoped_array.hpp>
#include <boost/noncopyable.hpp>
#include "errors.h"

/**
 * Thread-safe bounded queue, supporting multiple producers and
 * multiple consumers.
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

    /// Returns the maximum capacity of the queue.
    size_type capacity() const;

    /**
     * Returns the number of elements in the queue. This should only be
     * used when debugging or when there are known to be no other threads
     * modifying the queue, as otherwise the result may be out-of-date
     * as soon as it returns.
     *
     * It is non-const because it requires the mutex to function.
     */
    size_type size();

    /**
     * Add an item to the queue. This will block if the queue is full.
     */
    void push(const value_type &item);

    /**
     * Extract an item from the queue. This will block if the queue is empty.
     * Note that the item is returned by value. This is necessary since the
     * storage for the slot in the queue is immediately made available to be
     * overwritten.
     */
    value_type pop();

    /**
     * Constructor.
     *
     * @param capacity Maximum capacity of the queue.
     *
     * @pre @a capacity > 0.
     */
    WorkQueue(size_type capacity);

private:
    size_type capacity_;
    size_type head_;
    size_type tail_;
    size_type size_;
    boost::condition_variable spaceCondition, dataCondition;
    boost::mutex mutex;
    boost::scoped_array<value_type> values;
};


template<typename ValueType>
WorkQueue<ValueType>::WorkQueue(size_type capacity)
    : capacity_(capacity), head_(0), tail_(0), size_(0)
{
    MLSGPU_ASSERT(capacity > 0, std::length_error);
    values.reset(new ValueType[capacity]);
}

template<typename ValueType>
typename WorkQueue<ValueType>::size_type WorkQueue<ValueType>::size()
{
    boost::unique_lock<boost::mutex> lock(mutex);
    return size;
}

template<typename ValueType>
typename WorkQueue<ValueType>::size_type WorkQueue<ValueType>::capacity() const
{
    return capacity_;
}

template<typename ValueType>
void WorkQueue<ValueType>::push(const ValueType &value)
{
    boost::unique_lock<boost::mutex> lock(mutex);
    while (size_ == capacity_)
    {
        spaceCondition.wait(lock);
    }
    values[tail_] = value;
    tail_++;
    if (tail_ == capacity_)
        tail_ = 0;
    size_++;
    dataCondition.notify_one();
}

template<typename ValueType>
ValueType WorkQueue<ValueType>::pop()
{
    boost::unique_lock<boost::mutex> lock(mutex);
    while (size_ == 0)
    {
        dataCondition.wait(lock);
    }
    ValueType ans = values[head_];
    head_++;
    if (head_ == capacity_)
        head_ = 0;
    size_--;
    spaceCondition.notify_one();
    return ans;
}

#endif /* !WORK_QUEUE_H */
