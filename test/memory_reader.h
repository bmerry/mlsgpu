/**
 * @file
 *
 * Reader class that pulls data from memory, for ease of testing.
 */

#ifndef MEMORY_READER_H
#define MEMORY_READER_H

#if HAVE_CONFIG_H
# include <config.h>
#endif

#include <cstddef>
#include <string>
#include <boost/filesystem/path.hpp>
#include "../src/binary_io.h"

/**
 * A reader that processes a range of existing memory.
 *
 * This is primarily intended for test code.
 */
class MemoryReader : public BinaryReader
{
private:
    const char *data_;
    std::size_t size_;

    virtual void openImpl(const boost::filesystem::path &path);
    virtual void closeImpl();
    virtual std::size_t readImpl(void *buffer, std::size_t count, offset_type offset) const;
    virtual offset_type sizeImpl() const;

public:
    /**
     * Construct from an existing memory range.
     * @param data             Start of memory region.
     * @param size             Bytes in memory region.
     * @note The memory range must not be deleted or modified until the object
     * is destroyed.
     */
    MemoryReader(const char *data, std::size_t size);
};

/**
 * Satisfies the requirements for a reader factory in @ref FastPly::Reader.
 */
class MemoryReaderFactory
{
public:
    typedef BinaryReader *result_type;

    BinaryReader *operator()() const
    {
        return new MemoryReader(content.data(), content.size());
    }

    MemoryReaderFactory(const std::string &content) : content(content) {}

private:
    std::string content;
};

#endif /* !MEMORY_READER_H */
