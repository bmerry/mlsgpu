/**
 * @file
 *
 * Thread-safe circular buffer for pipelining variable-sized data chunks.
 */

#ifndef CIRCULAR_BUFFER_H
#define CIRCULAR_BUFFER_H

#if HAVE_CONFIG_H
# include <config.h>
#endif

#include <cstddef>
#include <utility>
#include <string>
#include <list>
#include <boost/noncopyable.hpp>
#include <boost/thread/locks.hpp>
#include <boost/thread/condition_variable.hpp>
#include "tr1_cstdint.h"
#include "allocator.h"
#include "timeplot.h"

/**
 * Thread-safe circular buffer manager. It does not actually handle
 * storage, but manages allocations and deallocations from an abstract
 * contiguous pool. For a memory-backed buffer, use @ref CircularBuffer.
 *
 * Allocations are made with @ref allocate; these will block until sufficient
 * memory is available. The memory must later be returned with @ref free. It is
 * not required to free memory in the same order as allocations, but it should
 * be done roughly in this order for best performance. The intended use case is
 * multiple producers allocating memory to pass data to multiple consumers,
 * which free the memory.
 */
class CircularBufferBase : public boost::noncopyable
{
private:
    /**
     * Mutex taken by @ref allocate, to ensure that allocations make progress.
     * Without this, one thread could be continually making and releasing small
     * allocations, while starving a thread that needs one big allocation.
     *
     * This mutex must be taken @em before taking @ref mutex.
     */
    boost::mutex allocMutex;

    /// Mutex to protect internal data structures
    boost::mutex mutex;

    /// Condition signalled when it may be possible to allocate more memory
    boost::condition_variable spaceCondition;

    /// Total number of elements
    std::size_t bufferSize;

    /**
     * First free position. It is legal for this to be anything in the range
     * [0, @ref bufferSize]. The two end-points are equivalent.
     */
    std::size_t firstFree;

    /// Start positions of all live allocations.
    Statistics::Container::list<std::size_t> allocPoints;

public:
    /**
     * Metadata about an allocation. This contains both public information
     * about the location, and private information needed to free the
     * allocation. It can be freely copied.
     */
    class Allocation
    {
        friend class CircularBufferBase;
    private:
        /**
         * Iterator into @ref CircularBufferBase::allocPoints to be removed
         * on free.
         */
        Statistics::Container::list<std::size_t>::iterator point;

        /// Constructor used by @ref CircularBufferBase::allocate
        explicit Allocation(Statistics::Container::list<std::size_t>::iterator point);
    public:
        /// Creates an invalid allocation
        Allocation();

        /// Obtain the position of the allocation
        std::size_t get() const;
    };

    /**
     * Constructor.
     *
     * @param name       Name for allocator used for internal metadata.
     * @param size       Number of elements in the buffer.
     *
     * @pre @a size &gt; 0
     */
    explicit CircularBufferBase(const std::string &name, std::size_t size);

    /// Return number of elements in the buffer
    std::size_t size() const;

    /**
     * Return number of unallocated elements in the buffer. This should be
     * considered immediately stale in a multithreaded environment, but may
     * be useful for load-balancing heuristics.
     *
     * It is not declared @ref const because it needs to hold the mutex.
     */
    std::size_t unallocated();

    /**
     * Allocate items from the buffer.
     * @param tworker        Worker to which waiting time is accounted.
     * @param n              Number of items to allocate.
     *
     * @pre 0 &lt; @a n &lt; @ref size().
     */
    Allocation allocate(Timeplot::Worker &tworker, std::size_t n);

    /**
     * Free previously allocated items. Note that undefined behaviour
     * (including memory corruption or crash) will result if the
     * allocator provided was not returned from @ref allocate on this
     * circular buffer, or has already been freed.
     *
     * @param alloc          Allocation returned from @ref allocate.
     */
    void free(const Allocation &alloc);
};

/**
 * Thread-safe circular buffer for pipelining variable-sized data chunks.
 *
 * It is @em not safe for use with non-POD types, as memory will be
 * uninitialized.
 */
class CircularBuffer : protected CircularBufferBase
{
private:
    /// Allocator used to allocate and free @ref buffer
    Statistics::Allocator<std::allocator<char> > allocator;
    /// Memory backing the buffer
    char *buffer;
public:
    /**
     * Information about an allocation from @ref allocate
     */
    class Allocation
    {
        friend class CircularBuffer;
    private:
        CircularBufferBase::Allocation base;
        void *ptr;            ///< Pointer to the allocated memory

    public:
        void *get() const;    ///< Obtain the data pointer
    };

    using CircularBufferBase::size;
    using CircularBufferBase::unallocated;

    /**
     * Allocate some memory from the buffer. If the memory is not yet
     * available, this will block until it is.
     *
     * It is thread-safe to call this function at the same time as @a free.
     *
     * @warning The returned data is not necessarily aligned, and one should
     * not cast the pointer to a type that requires alignment. As an exception,
     * if @em all calls to @c allocate use the same @a elementSize then the
     * result is guaranteed to be an allocator-returned pointer plus a multiple
     * of @a elementSize.
     *
     * @param tworker         Worker to indicate waiting time.
     * @param elementSize     Size of a single element.
     * @param elements        Number of elements to allocate.
     * @return A pointer to the allocated data
     *
     * @pre
     * - 0 &lt; @a elementSize * @a maxElements &lt; @ref size()
     */
    Allocation allocate(Timeplot::Worker &tworker, std::size_t elementSize, std::size_t elements);

    /**
     * Variant of @ref allocate(Timeplot::Worker &, std::size_t, std::size_t)
     * that takes just a byte count.
     *
     * @param tworker         Worker to indicate waiting time.
     * @param bytes           Number of bytes to allocate.
     * @return Allocation information
     *
     * @pre
     * - 0 &lt; @a bytes &lt;= @ref size()
     */
    Allocation allocate(Timeplot::Worker &tworker, std::size_t bytes);

    /**
     * Free memory allocated by @ref allocate. Each call to @ref allocate must be matched with
     * one to this function.
     */
    void free(const Allocation &alloc);

    /**
     * Constructor.
     *
     * @param name      Buffer name used for memory statistic.
     * @param size      Bytes of storage to reserve.
     *
     * @pre @a size &gt; 0
     */
    CircularBuffer(const std::string &name, std::size_t size);

    /// Destructor
    ~CircularBuffer();
};

#endif /* !CIRCULAR_BUFFER_H */
