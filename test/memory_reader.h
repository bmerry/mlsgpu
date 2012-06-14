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
#include <boost/smart_ptr/shared_array.hpp>
#include "../src/fast_ply.h"

/**
 * A reader that processes a range of existing memory.
 *
 * This is primarily intended for test code.
 */
class MemoryReader : public FastPly::ReaderBase
{
private:
    class MemoryHandle : public FastPly::ReaderBase::Handle
    {
    private:
        /// Pointer to the first vertex
        const char *vertexPtr;

    protected:
        virtual const char *readRaw(size_type first, size_type last, char *buffer) const;

    public:
        /**
         * Constructor.
         * @param owner     The creating reader.
         * @param data      Pointer to the start of the file (the header, not the vertices).
         */
        explicit MemoryHandle(const MemoryReader &owner, const char *data);
    };

public:
    /**
     * Construct from an existing memory range.
     * @param data             Start of memory region.
     * @param size             Bytes in memory region.
     * @param smooth           Scale factor applied to radii as they're read.
     * @throw FormatError if the file was malformed
     * @note The memory range must not be deleted or modified until the object
     * is destroyed.
     */
    MemoryReader(const char *data, std::size_t size, float smooth);

    virtual Handle *createHandle() const;
    virtual Handle *createHandle(std::size_t bufferSize) const;
    virtual Handle *createHandle(boost::shared_array<char> buffer, std::size_t bufferSize) const;

private:
    const char *data;
};

#endif /* !MEMORY_READER_H */
