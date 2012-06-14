/**
 * @file
 *
 * Reader class that pulls data from memory, for ease of testing.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif

#include <cstddef>
#include <boost/smart_ptr/shared_array.hpp>
#include <boost/iostreams/device/array.hpp>
#include <boost/iostreams/stream.hpp>
#include "../src/fast_ply.h"
#include "memory_reader.h"

using FastPly::ReaderBase;

MemoryReader::MemoryHandle::MemoryHandle(const MemoryReader &owner, const char *data)
    : ReaderBase::Handle(owner, 0), vertexPtr(data + owner.getHeaderSize())
{
}

const char *MemoryReader::MemoryHandle::readRaw(size_type first, size_type last, char *buffer) const
{
    (void) buffer; // will be NULL anyway
    (void) last;
    return vertexPtr + first * owner.getVertexSize();
}

MemoryReader::MemoryReader(const char *data, std::size_t size, float smooth)
    : ReaderBase(smooth), data(data)
{
    boost::iostreams::array_source source(data, size);
    boost::iostreams::stream<boost::iostreams::array_source> in(source);
    readHeader(in);
    if ((size - getHeaderSize()) / getVertexSize() < this->size())
        throw boost::enable_error_info(FastPly::FormatError("Input source is too small to contain its vertices"));
}

ReaderBase::Handle *MemoryReader::createHandle() const
{
    return new MemoryHandle(*this, data);
}

ReaderBase::Handle *MemoryReader::createHandle(std::size_t bufferSize) const
{
    (void) bufferSize; // no buffer is used
    return new MemoryHandle(*this, data);
}

ReaderBase::Handle *MemoryReader::createHandle(boost::shared_array<char> buffer, std::size_t bufferSize) const
{
    (void) buffer; // no buffer is used
    (void) bufferSize;
    return new MemoryHandle(*this, data);
}
