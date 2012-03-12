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
#include <tr1/cstdint>
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

    template<typename ValueType2, typename IdType2> friend class OrderedWorkQueue;
};

/**
 * A variation on a work queue in which workitems have sequential IDs and
 * can only be inserted in order. Attempting to push an out-of-order item
 * will block until the prior items have been pushed. This reduces efficiency
 * but makes certain use-cases more deterministic.
 */
template<typename ValueType, typename IdType = std::tr1::uint64_t>
class OrderedWorkQueue : private WorkQueue<ValueType>
{
private:
    typedef WorkQueue<ValueType> Base;
public:
    typedef typename Base::value_type value_type;
    typedef typename Base::size_type size_type;
    typedef IdType id_type;
    using Base::capacity;
    using Base::size;
    using Base::pop;

    /**
     * Constructor.
     *
     * @param capacity Maximum capacity of the queue.
     *
     * @pre @a capacity > 0.
     */
    OrderedWorkQueue(size_type capacity);

    /**
     * Add an item to the queue. This will block if the queue is full or
     * if @a id is not the next ID in sequence (starting from 0).
     */
    void push(const value_type &item, id_type id);

    /**
     * Add an item to the queue and advance a range of IDs. This will block
     * if the queue is full or if @a firstId is not the next ID in sequence
     * (starting from 0). Afterwards, @a lastId will be the new next-ID.
     */
    void push(const value_type &items, id_type firstId, id_type lastId);

    /**
     * Indicate that nothing should be pushed on the queue for a specific
     * ID. This will block until @a id is the next ID in the sequence, but
     * will then push nothing on the queue.
     */
    void skip(id_type id);

    /**
     * Indicate that nothing should be pushed on the queue for a specific
     * range of IDs. This will block until @a firstId is the next ID in the
     * sequence, but will then push nothing on the queue. Afterwards,
     * @a lastId will become the new next-ID.
     */
    void skip(id_type firstId, id_type lastId);

private:
    size_type next_;        ///< Next expected ID for @ref push or @ref skip
    boost::condition_variable idCondition;
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
    boost::lock_guard<boost::mutex> lock(mutex);
    return size_;
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


template<typename ValueType, typename IdType>
OrderedWorkQueue<ValueType, IdType>::OrderedWorkQueue(size_type capacity)
    : Base(capacity), next_(0)
{
}

template<typename ValueType, typename IdType>
void OrderedWorkQueue<ValueType, IdType>::push(const value_type &value, id_type id)
{
    push(value, id, id + 1);
}

template<typename ValueType, typename IdType>
void OrderedWorkQueue<ValueType, IdType>::push(const value_type &value, id_type firstId, id_type lastId)
{
    boost::unique_lock<boost::mutex> lock(this->mutex);
    while (next_ != firstId)
    {
        idCondition.wait(lock);
    }

    while (this->size_ == this->capacity_)
    {
        this->spaceCondition.wait(lock);
    }
    this->values[this->tail_] = value;
    this->tail_++;
    if (this->tail_ == this->capacity_)
        this->tail_ = 0;
    this->size_++;
    this->dataCondition.notify_one();
    next_ = lastId;
    idCondition.notify_all();
}

template<typename ValueType, typename IdType>
void OrderedWorkQueue<ValueType, IdType>::skip(id_type id)
{
    skip(id, id + 1);
}

template<typename ValueType, typename IdType>
void OrderedWorkQueue<ValueType, IdType>::skip(id_type firstId, id_type lastId)
{
    boost::unique_lock<boost::mutex> lock(this->mutex);
    while (next_ != firstId)
    {
        idCondition.wait(lock);
    }
    next_ = lastId;
    idCondition.notify_all();
}

#endif /* !WORK_QUEUE_H */
