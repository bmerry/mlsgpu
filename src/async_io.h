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

class AsyncWriterItem
{
    friend class AsyncWriter;
    friend class detail::AsyncWriterWorker;
private:
    CircularBuffer::Allocation alloc;
    boost::shared_ptr<BinaryWriter> out;
    std::size_t count;
    BinaryWriter::offset_type offset;
public:
    void *get() const { return alloc.get(); }
};

namespace detail
{

class AsyncWriterWorker : public WorkerBase
{
private:
    ::AsyncWriter &owner;
public:
    explicit AsyncWriterWorker(AsyncWriter &owner);

    void operator()(AsyncWriterItem &item);
};

} // namespace detail

class AsyncWriter : public WorkerGroup<AsyncWriterItem, detail::AsyncWriterWorker, AsyncWriter>
{
public:
    boost::shared_ptr<AsyncWriterItem> get(Timeplot::Worker &tworker, std::size_t bytes);

    void push(
        Timeplot::Worker &tworker,
        boost::shared_ptr<AsyncWriterItem> item,
        boost::shared_ptr<BinaryWriter> out,
        std::size_t count,
        BinaryWriter::offset_type offset);

    void freeItem(boost::shared_ptr<AsyncWriterItem> item);

    explicit AsyncWriter(std::size_t numWorkers, std::size_t bufferSize);

private:
    typedef WorkerGroup<AsyncWriterItem, detail::AsyncWriterWorker, AsyncWriter> Base;

    CircularBuffer buffer;
};

#endif /* !ASYNC_IO */
