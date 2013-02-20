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
#include "../src/binary_io.h"
#include "memory_reader.h"

void MemoryReader::openImpl(const boost::filesystem::path &path)
{
    (void) path;
    // No action required
}

void MemoryReader::closeImpl()
{
    // No action required
}

std::size_t MemoryReader::readImpl(
    void *buffer, std::size_t count, offset_type offset) const
{
    if (offset >= size_)
        return 0; // past the EOF
    if (count > size_ - offset)
        count = size_ - offset; // Clamp
    std::memcpy(buffer, data_ + offset, count);
    return count;
}

MemoryReader::offset_type MemoryReader::sizeImpl() const
{
    return size_;
}

MemoryReader::MemoryReader(const char *data, std::size_t size)
    : data_(data), size_(size)
{
}
