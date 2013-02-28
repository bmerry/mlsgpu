/**
 * Asynchronous writes through a @ref BinaryWriter.
 */

#ifndef ASYNC_IO
#define ASYNC_IO

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <cstddef>
#include <boost/smart_ptr/shared_ptr.hpp>
#include "binary_io.h"
#include "work_queue.h"
#include "worker_group.h"
#include "circular_buffer.h"
#include "timeplot.h"

class AsyncWriter;
namespace detail { class AsyncWriterWorker; }

/**
 * Wraps an allocation from @ref AsyncWriter::get.
 */
class AsyncWriterItem
{
    friend class AsyncWriter;
    friend class detail::AsyncWriterWorker;
private:
    CircularBuffer::Allocation alloc; ///< Memory allocation from the buffer
    /// Output file for writing (only defined after @ref AsyncWriter::push)
    boost::shared_ptr<BinaryWriter> out;
    /**
     * Number of bytes. Before calling @ref AsyncWriter::push, this contains
     * the allocated buffer size. After @ref AsyncWriter::push, it contains
     * the number of bytes that will be written, which may be less.
     */
    std::size_t count;
    /// Position in the file to write (only defined after @ref AsyncWriter::push)
    BinaryWriter::offset_type offset;
public:
    /**
     * Retrieve pointer to the raw data.
     */
    void *get() const { return alloc.get(); }
};

namespace detail
{

/// Thread worker for @ref AsyncWriter
class AsyncWriterWorker : public WorkerBase
{
private:
    ::AsyncWriter &owner;
public:
    explicit AsyncWriterWorker(AsyncWriter &owner);

    void operator()(AsyncWriterItem &item);
};

} // namespace detail

/**
 * Class for asynchronous writes to a @ref BinaryWriter. It manages a thread
 * pool that perform the actual writes, and manages its own buffer. Users call
 * @ref get to allocate from the buffer, followed by @ref push once the data
 * have been placed in the buffer.
 *
 * File handles are passed as shared pointers to facilitate automatic closing
 * of the file after it is no longer referenced.
 */
class AsyncWriter : public WorkerGroup<AsyncWriterItem, detail::AsyncWriterWorker, AsyncWriter>
{
public:
    /**
     * Obtain data from the buffer. The actual space for data is found by calling
     * @ref AsyncWriterItem::get.
     *
     * @param tworker      Worker to which waiting time is accounted.
     * @param bytes        Number of bytes to allocate.
     * @return An object in which to store the data.
     *
     * @pre 0 &lt; @a bytes &lt;= buffer size
     *
     * @warning Be careful not to confuse <code>.</code> with <code>-&gt;</code> when calling
     * @c get, or you will corrupt the @c shared_ptr.
     */
    boost::shared_ptr<AsyncWriterItem> get(Timeplot::Worker &tworker, std::size_t bytes);

    /**
     * Enqueue data for writing.
     * @param tworker      Unused.
     * @param item         Item retrieved by @ref get
     * @param out          Target file.
     * @param count        Number of bytes to write.
     * @param offset       Position in target file.
     *
     * @note It is legal to write fewer bytes than were initially allocated, although this
     * will waste space in the buffer.
     */
    void push(
        Timeplot::Worker &tworker,
        boost::shared_ptr<AsyncWriterItem> item,
        boost::shared_ptr<BinaryWriter> out,
        std::size_t count,
        BinaryWriter::offset_type offset);

    /// Return the data to the circular buffer
    void freeItem(boost::shared_ptr<AsyncWriterItem> item);

    /**
     * Constructor.
     *
     * @param numWorkers   Number of workers in the pool.
     * @param bufferSize   Bytes to allocate in the buffer.
     */
    explicit AsyncWriter(std::size_t numWorkers, std::size_t bufferSize);

private:
    typedef WorkerGroup<AsyncWriterItem, detail::AsyncWriterWorker, AsyncWriter> Base;

    CircularBuffer buffer;
};

#endif /* !ASYNC_IO */
