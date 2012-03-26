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
    explicit WorkQueue(size_type capacity);

protected:
    boost::mutex mutex;

    /// Implementation of @ref push where the caller holds the mutex
    void pushUnlocked(boost::unique_lock<boost::mutex> &lock, const value_type &item);

    /// Implementation of @ref pop where the caller holds the mutex
    value_type popUnlocked(boost::unique_lock<boost::mutex> &lock);

private:
    size_type capacity_;
    size_type head_;
    size_type tail_;
    size_type size_;
    boost::condition_variable spaceCondition, dataCondition;
    boost::scoped_array<value_type> values;
};

/**
 * A pool of items which can be added to and removed from, with thread safety.
 * Currently this is just an alias to a WorkQueue, but in future it may be
 * extended to use something like a stack which is more cache-friendly.
 */
template<typename T>
class Pool : public WorkQueue<boost::shared_ptr<T> >
{
public:
    Pool(typename WorkQueue<boost::shared_ptr<T> >::size_type capacity)
        : WorkQueue<boost::shared_ptr<T> >(capacity) {}
};

/**
 * A work queue in which incoming work is batched into @em generations, with consumers
 * guaranteed to receive workitems ordered by generation. Unlike a standard
 * @ref WorkQueue, this class needs producers to interact quite closely with
 * it. The canonical order of operations is:
 *  -# All producers call @ref producerStart.
 *  -# The consumer threads are started.
 *  -# The producers enqueue work using @ref push, calling @ref producerNext whenever
 *     the producer moves on to its next generation.
 *  -# The producers call @ref producerStop and shut down.
 *  -# The consumers are shut down.
 */
template<typename ValueType, typename GenType = unsigned int, typename CompareGen = std::less<GenType> >
class GenerationalWorkQueue : protected WorkQueue<std::pair<ValueType, GenType> >
{
public:
    typedef ValueType value_type;
    typedef GenType gen_type;
    typedef typename WorkQueue<std::pair<ValueType, GenType> >::size_type size_type;

    using WorkQueue<std::pair<ValueType, GenType> >::capacity;
    using WorkQueue<std::pair<ValueType, GenType> >::size;

    /**
     * Indicate a change in the generation a producer is working on.
     *
     * @pre
     * - The producer was previously working on generation @a oldGen.
     * - @a oldGen is less than or equal to @a newGen.
     */
    void producerNext(const gen_type &oldGen, const gen_type &newGen);

    /**
     * Indicate that a producer has started and is working on initial
     * generation @a gen. If the actual generation is not known, a lower bound
     * should be used.
     *
     * @pre
     * - Must not be inside a nested @ref producerStart.
     */
    void producerStart(const gen_type &gen);

    /**
     * Indicate that a producer has completed work on generation @a gen. This
     * potentially unblocks other producers to enqueue work for following
     * generations.
     *
     * @pre Must be paired with a previous @ref producerStart.
     */
    void producerStop(const gen_type &gen);

    /**
     * Enqueue a work item. This may block if there are other producers
     * still working on a previous generation.
     *
     * @pre
     * - This must be called between @ref producerStart and @ref producerStop.
     * - The producer must have indicated (with @ref producerStart or @ref
     *   producerNext) that it is working on @a gen.
     */
    void push(const gen_type &gen, const value_type &item);

    /**
     * Enqueue a work item without a generation. This can be called
     * independently of whether producers are marked as active. The intended
     * use case is to enqueue special marker work items that request consumers
     * to shut down.
     *
     * When the corresponding item is popped, it will be returned with a
     * default-constructed generation.
     */
    void pushNoGen(const value_type &item);

    /**
     * Extract a work item from the queue.
     */
    value_type pop(gen_type &gen);

    /**
     * Constructor.
     *
     * @param capacity Maximum capacity of the queue.
     * @param compare  Less-than operator for comparing generations.
     *
     * @pre @a capacity > 0.
     */
    GenerationalWorkQueue(size_type capacity, const CompareGen &compare = CompareGen());

private:
    typedef std::map<GenType, unsigned int, CompareGen> active_type;

    /**
     * Number of active producers in each generation.
     */
    active_type active;

    /// Condition signaled when the generation changes
    boost::condition_variable nextGenCondition;
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
    pushUnlocked(lock, value);
}

template<typename ValueType>
void WorkQueue<ValueType>::pushUnlocked(
    boost::unique_lock<boost::mutex> &lock,
    const ValueType &value)
{
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
    return popUnlocked(lock);
}

template<typename ValueType>
ValueType WorkQueue<ValueType>::popUnlocked(boost::unique_lock<boost::mutex> &lock)
{
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


template<typename ValueType, typename GenType, typename CompareGen>
GenerationalWorkQueue<ValueType, GenType, CompareGen>::GenerationalWorkQueue(
    size_type capacity, const CompareGen &compare)
: WorkQueue<std::pair<ValueType, GenType> >(capacity), active(compare)
{
}

template<typename ValueType, typename GenType, typename CompareGen>
void GenerationalWorkQueue<ValueType, GenType, CompareGen>::producerStart(const gen_type &gen)
{
    boost::unique_lock<boost::mutex> lock(this->mutex);
    ++active[gen];
}

template<typename ValueType, typename GenType, typename CompareGen>
void GenerationalWorkQueue<ValueType, GenType, CompareGen>::producerNext(
    const gen_type &oldGen, const gen_type &newGen)
{
    boost::unique_lock<boost::mutex> lock(this->mutex);
    MLSGPU_ASSERT(!active.key_comp()(newGen, oldGen), std::invalid_argument);

    typename active_type::iterator pos = active.find(oldGen);
    MLSGPU_ASSERT(pos != active.end(), std::logic_error);

    // Must do this increment after we have done the assertion above, and before we
    // check whether the old generation has become inactive.
    ++active[newGen];
    if (--pos->second == 0)
    {
        // Old generation is no longer active
        if (pos == active.begin())
            nextGenCondition.notify_all();
        active.erase(pos);
    }
}

template<typename ValueType, typename GenType, typename CompareGen>
void GenerationalWorkQueue<ValueType, GenType, CompareGen>::producerStop(const gen_type &gen)
{
    boost::unique_lock<boost::mutex> lock(this->mutex);
    typename active_type::iterator pos = active.find(gen);
    MLSGPU_ASSERT(pos != active.end(), std::logic_error);
    if (--pos->second == 0)
    {
        // Generation is no longer active
        if (pos == active.begin())
            nextGenCondition.notify_all();
        active.erase(pos);
    }
}

template<typename ValueType, typename GenType, typename CompareGen>
void GenerationalWorkQueue<ValueType, GenType, CompareGen>::push(
    const gen_type &gen, const value_type &item)
{
    boost::unique_lock<boost::mutex> lock(this->mutex);
    MLSGPU_ASSERT(active.count(gen), std::logic_error);
    while (gen != active.begin()->first)
    {
        nextGenCondition.wait(lock);
        MLSGPU_ASSERT(active.count(gen), std::logic_error);
    }
    pushUnlocked(lock, std::make_pair(item, gen));
}

template<typename ValueType, typename GenType, typename CompareGen>
void GenerationalWorkQueue<ValueType, GenType, CompareGen>::pushNoGen(
    const value_type &item)
{
    boost::unique_lock<boost::mutex> lock(this->mutex);
    pushUnlocked(lock, std::make_pair(item, gen_type()));
}

template<typename ValueType, typename GenType, typename CompareGen>
ValueType GenerationalWorkQueue<ValueType, GenType, CompareGen>::pop(gen_type &gen)
{
    std::pair<value_type, gen_type> ans = WorkQueue<std::pair<ValueType, GenType> >::pop();
    gen = ans.second;
    return ans.first;
}

#endif /* !WORK_QUEUE_H */
