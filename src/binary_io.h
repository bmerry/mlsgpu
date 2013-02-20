/**
 * @file
 *
 * Binary file I/O classes with thread-safe absolute positioning.
 */

#ifndef BINARY_IO_H
#define BINARY_IO_H

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <cstddef>
#include <boost/filesystem/path.hpp>
#include "tr1_cstdint.h"

/// Enumeration of the types of binary reader
enum ReaderType
{
    MMAP_READER,
    STREAM_READER,
    SYSCALL_READER
};

/// Enumeration of the types of binary writer
enum WriterType
{
    STREAM_WRITER,
    SYSCALL_WRITER
};

/**
 * Base class that handles both reading and writing.
 */
class BinaryIO
{
public:
    /// Type used to represent positions in files.
    typedef std::tr1::uint64_t offset_type;

    /**
     * Open the file. This is implemented by calling @ref openImpl.
     *
     * @throw boost::exception if there was an error opening the file
     *
     * @pre The file is not already open.
     */
    void open(const boost::filesystem::path &path);

    /**
     * Close the file. This is implemented by calling @ref closeImpl.
     *
     * @pre The file is open.
     */
    void close();

    virtual ~BinaryIO();

    /**
     * Get the filename.
     *
     * @returns The filename if the file is open, or the empty string if it is closed.
     */
    const std::string &filename() const;

    /**
     * Returns whether the file is open.
     */
    bool isOpen() const;

private:
    bool isOpen_;
    std::string filename_;   /// Filename for error messages

    /**
     * Implements @ref open. It does not need to do any state checks, nor
     * put the filename into exceptions.
     */
    virtual void openImpl(const boost::filesystem::path &path) = 0;

    /**
     * Implements @ref close. It does not need to do any state checks, nor
     * put the filename into exceptions.
     */
    virtual void closeImpl() = 0;
};

/**
 * Abstract base class for binary file reads.
 */
class BinaryReader : public BinaryIO
{
public:
    /**
     * Reads up to @a count bytes from the file, starting at @a offset.
     *
     * @param buf      Buffer to receive the data
     * @param count    Number of bytes to read
     * @param offset   Position in file to start read
     * @return The number of bytes read.
     * @throw boost::exception if there was a low-level I/O error
     *
     * @pre The file is open.
     */
    std::size_t read(void *buf, std::size_t count, offset_type offset) const;

private:
    /**
     * Implementation of @ref read. It does not need to check whether the file is
     * open or put the filename into exceptions.
     */
    virtual std::size_t readImpl(void *buf, std::size_t count, offset_type offset) const = 0;
};

/**
 * Abstract base class for binary file writes.
 */
class BinaryWriter : public BinaryIO
{
public:
    /**
     * Writes up to @a count bytes from the file, starting at @a offset.
     *
     * @param buf      Buffer containing the data
     * @param count    Number of bytes to write
     * @param offset   Position in file to start write
     * @return The number of bytes written.
     * @throw boost::exception if there was a low-level I/O error
     *
     * @pre The file is open
     */
    std::size_t write(const void *buf, std::size_t count, offset_type offset) const;

private:
    /**
     * Implementation of @ref write. It does not need to check that the file is open or
     * put the filename into exceptions.
     */
    virtual std::size_t writeImpl(const void *buf, std::size_t count, offset_type offset) const = 0;
};

/**
 * Factory function to create a new reader of the specified type.
 */
BinaryReader *createReader(ReaderType type);

/**
 * Factory function to create a new writer of the specified type.
 */
BinaryWriter *createWriter(WriterType type);

#endif /* !BINARY_IO_H */
