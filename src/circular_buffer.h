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

/**
 * @todo Track memory for allocPoints
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

    std::size_t bufferSize;
    std::list<std::size_t> allocPoints;
    std::size_t firstFree;

public:
    class Allocation
    {
        friend class CircularBufferBase;
    private:
        std::list<std::size_t>::iterator point;

        explicit Allocation(std::list<std::size_t>::iterator point);
    public:
        Allocation();

        std::size_t get() const;
    };

    explicit CircularBufferBase(std::size_t size);

    std::size_t size() const;

    Allocation allocate(std::size_t n);
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
    char *buffer;
public:
    class Allocation
    {
        friend class CircularBuffer;
    private:
        CircularBufferBase::Allocation base;
        char *ptr;

    public:
        void *get() const;
    };

    using CircularBufferBase::size;

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
     * @param elementSize     Size of a single element.
     * @param elements        Number of elements to allocate.
     * @return A pointer to the allocated data
     *
     * @pre
     * - 0 &lt; @a elementSize * @a maxElements &lt; @ref size()
     */
    Allocation allocate(std::size_t elementSize, std::size_t elements);

    /**
     * Variant of @ref allocate(std::size_t, std::size_t) that takes just a byte count.
     *
     * @param bytes           Number of bytes to allocate.
     * @return A pointer to the allocated data
     *
     * @pre
     * - 0 &lt; bytes &lt; @ref size()
     */
    Allocation allocate(std::size_t bytes);

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
     * @pre @a size &gt;= 2
     */
    CircularBuffer(const std::string &name, std::size_t size);

    /// Destructor
    ~CircularBuffer();
};

#endif /* !CIRCULAR_BUFFER_H */
