/**
 * @file
 *
 * Reader class that pulls data from memory, for ease of testing.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif

#include <cstddef>
#include <cstring>
#include <boost/iostreams/device/array.hpp>
#include <boost/iostreams/stream.hpp>
#include "../src/fast_ply.h"
#include "memory_reader.h"

using FastPly::ReaderBase;

MemoryReader::MemoryHandle::MemoryHandle(const MemoryReader &owner, const char *data)
    : ReaderBase::Handle(owner), vertexPtr(data + owner.getHeaderSize())
{
}

void MemoryReader::MemoryHandle::readRaw(size_type first, size_type last, char *buffer) const
{
    MLSGPU_ASSERT(first <= last, std::invalid_argument);
    MLSGPU_ASSERT(buffer != NULL, std::invalid_argument);

    const std::size_t vertexSize = owner.getVertexSize();
    std::memcpy(buffer, vertexPtr + first * vertexSize, (last - first) * vertexSize);
}

MemoryReader::MemoryReader(const char *data, std::size_t size, float smooth, float maxRadius)
    : ReaderBase(smooth, maxRadius), data(data)
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
