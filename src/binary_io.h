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
#include <map>
#include <string>
#include <boost/noncopyable.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/iostreams/positioning.hpp>
#include <boost/iostreams/categories.hpp>
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

/// Wrapper around @ref ReaderType for use with @ref Choice.
class ReaderTypeWrapper
{
public:
    typedef ReaderType type;
    static std::map<std::string, ReaderType> getNameMap();
};

/// Wrapper around @ref WriterType for use with @ref Choice.
class WriterTypeWrapper
{
public:
    typedef WriterType type;
    static std::map<std::string, WriterType> getNameMap();
};

/**
 * Base class that handles both reading and writing.
 */
class BinaryIO : public boost::noncopyable
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

    /// Constructor
    BinaryIO();

    /**
     * Virtual destructor to allow for polymorphism. Note that all concrete
     * subclasses are responsible for closing the file themselves. It cannot
     * be done from here as the subclass will have already been destroyed.
     */
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
    bool isOpen_;            ///< Whether the file is open
    std::string filename_;   ///< Filename for error messages

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

    /**
     * Return the size of the file.
     *
     * @throw boost::exception if there was a low-level I/O error
     *
     * @pre The file is open.
     */
    offset_type size() const;

private:
    /**
     * Implements @ref read. It does not need to check whether the file is
     * open or put the filename into exceptions.
     */
    virtual std::size_t readImpl(void *buf, std::size_t count, offset_type offset) const = 0;

    /**
     * Implements @ref size. It does not need to check whether the file is
     * open or put the filename into exceptions.
     */
    virtual offset_type sizeImpl() const = 0;
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

    /**
     * Resize the file to the given size. It is not guaranteed to be possible to
     * shrink a file (this depends on the specific subclass). However, creating a new
     * file and then resizing it to the final desired size will work.
     *
     * This function is not guaranteed to be thread-safe.
     */
    void resize(offset_type size) const;

private:
    /**
     * Implements @ref write. It does not need to check that the file is open or
     * put the filename into exceptions.
     */
    virtual std::size_t writeImpl(const void *buf, std::size_t count, offset_type offset) const = 0;

    /**
     * Implements @ref resize. It does not need to check that the file is open or
     * put the filename into exceptions.
     */
    virtual void resizeImpl(offset_type size) const = 0;
};

/**
 * Wraps a @ref BinaryReader in an interface that makes it a source for
 * @c boost::iostreams.
 */
class BinaryReaderSource
{
public:
    typedef char char_type;
    typedef boost::iostreams::seekable_device_tag category;

    std::streamsize read(char *buffer, std::streamsize count);
    std::streamsize write(const char *buffer, std::streamsize count);
    std::streampos seek(boost::iostreams::stream_offset off, std::ios_base::seekdir way);

    explicit BinaryReaderSource(const BinaryReader &reader);

private:
    const BinaryReader &reader;
    BinaryReader::offset_type offset;
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
