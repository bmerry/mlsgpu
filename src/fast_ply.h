/**
 * @file
 *
 * PLY file format loading and saving routines optimized for just the
 * operations desired in mlsgpu.
 */

#ifndef FAST_PLY_H
#define FAST_PLY_H

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <string>
#include <cstddef>
#include <stdexcept>
#include <istream>
#include <fstream>
#include <ostream>
#include <map>
#include <string>
#include <vector>
#include <algorithm>
#include <utility>
#include "tr1_cstdint.h"
#include <boost/iostreams/device/mapped_file.hpp>
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/smart_ptr/scoped_ptr.hpp>
#include <boost/smart_ptr/shared_array.hpp>
#include <boost/type_traits/is_pointer.hpp>
#include <boost/ref.hpp>
#include <boost/noncopyable.hpp>
#include "splat.h"
#include "errors.h"

class TestFastPlyReader;

/**
 * Classes and functions for efficient access to PLY files in a narrow
 * range of possible formats.
 */
namespace FastPly
{

/// Enumeration of the subclasses of @ref FastPly::ReaderBase
enum ReaderType
{
    MMAP_READER,
    SYSCALL_READER
};

/// Enumeration of the subclasses of @ref FastPly::WriterBase
enum WriterType
{
    MMAP_WRITER,
    STREAM_WRITER
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
 * An exception that is thrown when an invalid PLY file is encountered.
 * This is used to signal all errors in a PLY file (including early end-of-file),
 * except for low-level I/O errors in parsing the header.
 */
class FormatError : public std::runtime_error
{
public:
    FormatError(const std::string &msg) : std::runtime_error(msg) {}
};

/**
 * Base class for quickly reading a subset of PLY files.
 * It only supports the following:
 * - Binary files, endianness matching the host.
 * - Only the "vertex" element is loaded.
 * - The "vertex" element must be the first element in the file.
 * - The x, y, z, nx, ny, nz, radius elements must all be present and FLOAT32.
 * - The vertex element must not contain any lists.
 *
 * This is a virtual base class that provides the interfaces and handles the
 * header, but the actual movement of data is down to the subclasses.
 *
 * An instance of this class just holds the metadata, but no OS resources or
 * buffers. To actually read the data, one calls @ref createHandle() to create
 * a handle, at which point the file is opened.
 */
class ReaderBase
{
    friend class ::TestFastPlyReader;
public:
    /// Size capable of holding maximum supported file size
    typedef std::tr1::uint64_t size_type;
    typedef Splat value_type;

    /**
     * A file handle to a PLY file, used for reading. This is a virtual base
     * class which is overloaded by each type of reader to provide the mechanisms
     * for accessing the file.
     *
     * A word about thread safety. Two threads must not simultaneously access the
     * same handle, because it is possible that handles encode an OS-level file
     * position. It is safe to simultaneously access two handles that reference
     * the same underlying file. However, if two handles share a buffer, it is
     * not safe to simultaneously access both.
     */
    class Handle : public boost::noncopyable
    {
    protected:
        /// The reader whose file we're reading
        const ReaderBase &owner;

    private:
        /**
         * A buffer used as a target for reads. This can be @c NULL if the subclass
         * doesn't require a buffer.
         * @see @ref readRaw
         */
        boost::shared_array<char> buffer;
        /**
         * Number of bytes allocated for @ref buffer, or 0 if it is @c NULL.
         */
        std::size_t bufferSize;

        /// Copy a single splat from raw data representation into an output iterator
        template<typename OutputIterator>
        void loadSplat(const char *buffer, OutputIterator out) const;

    protected:
        /**
         * Create with an internal buffer of specified size.
         * It is legal for the size to be 0, in which case the subclass does not require
         * a buffer (e.g. because of memory mapping).
         *
         * @throw std::runtime_error If @a bufferSize < <code>owner.getVertexSize()</code>.
         */
        Handle(const ReaderBase &owner, std::size_t bufferSize);

        /**
         * Create with a user-provided buffer. This buffer can be shared between handles.
         *
         * @throw std::runtime_error If @a bufferSize < <code>owner.getVertexSize()</code>.
         */
        Handle(const ReaderBase &owner, boost::shared_array<char> buffer, std::size_t bufferSize);

        /**
         * Method implemented in subclasses to back the read. It must copy the specified
         * vertices to the buffer, in exactly the form they exist in the file i.e. just
         * a byte-level copy of the relevant data, without selecting the required fields.
         *
         * If the handle was created with no buffer, then @a buffer will be @c NULL.
         *
         * @param first,last      %Range of vertices to read.
         * @param buffer          Output buffer, or @c NULL if no buffer created.
         * @return A pointer to the data.
         *
         * @pre @a first &lt;= @a last &lt;= @ref size().
         * @post If @a buffer is non-NULL then it is also the return value.
         */
        virtual const char *readRaw(size_type first, size_type last, char *buffer) const = 0;

    public:
        /**
         * Copy out a contiguous selection of the vertices.
         * @param first,last      %Range of vertices to copy.
         * @param out             Target of copy.
         * @return The output iterator after the copy.
         * @pre @a first &lt;= @a last &lt;= @ref size().
         */
        template<typename OutputIterator>
            OutputIterator read(size_type first, size_type last, OutputIterator out) const;

        virtual ~Handle() {}
    };

    /// Number of vertices in the file
    size_type size() const { return vertexCount; }

    /// Number of bytes per vertex
    size_type getVertexSize() const { return vertexSize; }

    /**
     * Open the file and return a @ref ReaderBase::Handle for reading it. The user
     * provides a buffer to be used. Handles can share buffers, provided that
     * multiple threads do not try to use them at the same time.
     *
     * @param buffer         Shared buffer to be used.
     * @param bufferSize     Number of bytes in @a buffer.
     *
     * @pre @a bufferSize >= @ref getVertexSize().
     */
    virtual Handle *createHandle(boost::shared_array<char> buffer, std::size_t bufferSize) const = 0;

    /**
     * Open the file and return a @ref ReaderBase::Handle for reading it. The default
     * implementation calls #createHandle(boost::shared_array<char>, std::size_t) const
     * with a freshly-allocated buffer.
     *
     * @param bufferSize     Size to be used for internally-allocated buffer.
     *
     * @pre @a bufferSize >= @ref getVertexSize().
     */
    virtual Handle *createHandle(std::size_t bufferSize) const;

    /**
     * Open the file and return a @ref ReaderBase::Handle for reading it.
     *
     * The default implementation calls #createHandle(std::size_t) const with
     * a default size.
     *
     * @see @ref ReaderBase::Handle.
     */
    virtual Handle *createHandle() const;

    virtual ~ReaderBase() {}

private:
    /// Scale factor for radii
    float smooth;

    /// The properties found in the file.
    enum Property
    {
        X, Y, Z,
        NX, NY, NZ,
        RADIUS
    };
    static const unsigned int numProperties = 7;
    size_type headerSize;              ///< Bytes before the first vertex
    size_type vertexSize;              ///< Bytes per vertex
    size_type vertexCount;             ///< Number of vertices
    size_type offsets[numProperties];  ///< Byte offsets of each property within a vertex

protected:
    /**
     * Construct from a file.
     *
     * This will open the file to parse the header and then close it again.
     * If an exception is thrown, it will have the filename stored in it
     * using @c boost::errinfo_file_name.
     *
     * @param filename         File to open.
     * @param smooth           Scale factor applied to radii as they're read.
     * @throw FormatError if the header is malformed.
     * @throw std::ios::failure if there was an I/O error.
     */
    ReaderBase(const std::string &filename, float smooth);

    /**
     * Construct from an arbitrary stream.
     *
     * This is a more generic constructor that does not require the header
     * to be stored in a file. The subclass @em must call @ref readHeader
     * to load the header.
     *
     * @see @ref readHeader
     */
    ReaderBase(float smooth);

    /**
     * Does the heavy lifting of parsing the header. This is called by
     * the constructor if it takes a file, otherwise by the subclass
     * constructor.
     */
    void readHeader(std::istream &in);

    /// Return the number of bytes from the beginning of the file to the first vertex
    size_type getHeaderSize() const { return headerSize; }
};

/**
 * Reader that uses a memory mapping to access the file.
 * It must be possible to mmap the entire file (thus, a 64-bit
 * address space is needed to handle very large files).
 */
class MmapReader : public ReaderBase
{
private:
    class MmapHandle : public ReaderBase::Handle
    {
    private:
        /// The memory mapping
        boost::iostreams::mapped_file_source mapping;

        /// Pointer to the first vertex.
        const char *vertexPtr;

    protected:
        virtual const char *readRaw(size_type first, size_type last, char *buffer) const;

    public:
        explicit MmapHandle(const MmapReader &owner, const std::string &filename);
    };

public:
    /**
     * @copydoc ReaderBase::ReaderBase(const std::string &, float)
     */
    explicit MmapReader(const std::string &filename, float smooth);

    virtual Handle *createHandle() const;
    virtual Handle *createHandle(std::size_t bufferSize) const;
    virtual Handle *createHandle(boost::shared_array<char> buffer, std::size_t bufferSize) const;

private:
    const std::string filename;
};

/**
 * A reader that processes a range of existing memory.
 *
 * This is primarily intended for test code.
 * @todo Move it into the test directory
 */
class MemoryReader : public ReaderBase
{
private:
    class MemoryHandle : public ReaderBase::Handle
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

/**
 * Reader that uses @c read or equivalent OS-level functionality.
 *
 * @todo Port to Windows
 */
class SyscallReader : public ReaderBase
{
private:
    class SyscallHandle : public ReaderBase::Handle
    {
    private:
        int fd;

    public:
        /**
         * @copydoc ReaderBase::Handle::Handle(const ReaderBase &, boost::shared_array<char>, std::size_t)
         */
        SyscallHandle(const SyscallReader &owner, boost::shared_array<char> buffer, std::size_t bufferSize);

        virtual const char * readRaw(size_type first, size_type last, char *buffer) const;

        virtual ~SyscallHandle();
    };

    std::string filename;

public:
    /**
     * @copydoc ReaderBase::ReaderBase(const std::string &, float)
     */
    SyscallReader(const std::string &filename, float smooth);

    virtual Handle *createHandle(boost::shared_array<char> buffer, std::size_t bufferSize) const;
};

/// Common code shared by @ref MmapWriter and @ref StreamWriter
class WriterBase
{
public:
    /// Size capable of holding maximum supported file size
    typedef std::tr1::uint64_t size_type;

    virtual ~WriterBase();

    /**
     * Determines whether @ref open has been successfully called.
     */
    bool isOpen() const;

    /**
     * Add a comment to be written by @ref open.
     * @pre @ref open has not yet been successfully called.
     */
    void addComment(const std::string &comment);

    /**
     * Set the number of vertices that will be in the file.
     * @pre @ref open has not yet been successfully called.
     */
    void setNumVertices(size_type numVertices);

    /**
     * Set the number of indices that will be in the file.
     * @pre @ref open has not yet been successfully called.
     */
    void setNumTriangles(size_type numTriangles);

    /**
     * Create the file and write the header.
     * @pre @ref open has not yet been successfully called.
     */
    virtual void open(const std::string &filename) = 0;

    /**
     * Allocate storage in memory and write the header to it.
     * This version is primarily aimed at testing, to avoid
     * writing to file and reading back in.
     *
     * The memory is allocated with <code>new[]</code>, and
     * the caller is responsible for freeing it with <code>delete[]</code>.
     */
    virtual std::pair<char *, size_type> open() = 0;

    /**
     * Flush all data to the file and close it.
     *
     * After doing this, it is possible to open a new file, although the
     * comments will not be reset.
     */
    virtual void close() = 0;

    /**
     * Write a range of vertices.
     * @param first          Index of first vertex to write.
     * @param count          Number of vertices to write.
     * @param data           Array of <code>float[3]</code> values.
     * @pre @a first + @a count <= @a numVertices.
     */
    virtual void writeVertices(size_type first, size_type count, const float *data) = 0;

    /**
     * Write a range of triangles.
     * @param first          Index of first triangle to write.
     * @param count          Number of triangles to write.
     * @param data           Array of <code>uint32_t[3]</code> values containing indices.
     * @pre @a first + @a count <= @a numTriangles.
     */
    virtual void writeTriangles(size_type first, size_type count, const std::tr1::uint32_t *data) = 0;

    /**
     * Whether the class supports writing the data out-of-order.
     */
    virtual bool supportsOutOfOrder() const = 0;

protected:
    /// Bytes per vertex
    static const size_type vertexSize = 3 * sizeof(float);
    /// Bytes per triangle
    static const size_type triangleSize = 1 + 3 * sizeof(std::tr1::uint32_t);

    WriterBase();

    size_type getNumVertices() const;
    size_type getNumTriangles() const;

    /// Returns the header based on stored values
    std::string makeHeader();

    /// Sets the flag indicating whether the file is open
    void setOpen(bool open);

private:
    /// Storage for comments until they can be written by @ref open.
    std::vector<std::string> comments;
    size_type numVertices;              ///< Number of vertices (defaults to zero)
    size_type numTriangles;             ///< Number of triangles (defaults to zero)
    bool isOpen_;                       ///< Whether the file has been opened
};

/**
 * PLY file writer that only supports one format.
 * The supported format has:
 *  - Binary format with host endianness;
 *  - Vertices with x, y, z as 32-bit floats (no normals);
 *  - Faces with 32-bit unsigned integer indices;
 *  - 3 indices per face;
 *  - Arbitrary user-provided comments.
 * At present, the entire file is memory-mapped, which may significantly
 * limit the file size when using a 32-bit address space.
 *
 * Writing a file is done in phases:
 *  -# Set comments with @ref addComment and indicate the number of
 *     vertices and indices with @ref setNumVertices and @ref setNumTriangles.
 *  -# Write the header using @ref open.
 *  -# Use @ref writeVertices and @ref writeTriangles to write the data.
 *
 * The requirement for knowing the number of vertices and indices up front is a
 * limitation of the PLY format. If it is not possible to know this up front, you
 * will need to dump the vertices and indices to raw temporary files and stitch
 * it all together later.
 *
 * The final phase (writing of vertices and indices) is thread-safe, provided
 * that each thread is writing to a disjoint section of the file.
 *
 * @bug Due to the way Boost creates the file, it will have the executable bit
 * set on POSIX systems.
 */
class MmapWriter : public WriterBase
{
public:
    /// Constructor
    MmapWriter();

    virtual void open(const std::string &filename);
    virtual std::pair<char *, size_type> open();
    virtual void close();
    virtual void writeVertices(size_type first, size_type count, const float *data);
    virtual void writeTriangles(size_type first, size_type count, const std::tr1::uint32_t *data);
    virtual bool supportsOutOfOrder() const;

private:
    char *vertexPtr;                    /// Pointer to storage for the first vertex
    char *trianglePtr;                  /// Pointer to storage for the first triangle

    /**
     * The memory mapping backed by the output file. When using the in-memory
     * version, the pointer is NULL.
     */
    boost::scoped_ptr<boost::iostreams::mapped_file_sink> mapping;
};

/**
 * PLY file writer that only supports one format.
 * This class has exactly the same interface as @ref MmapWriter, and allows for
 * out-of-order writing. The advantage over @ref MmapWriter is that it does not
 * require a large virtual address space. However, it is potentially less
 * efficient.
 */
class StreamWriter : public WriterBase
{
public:
    /// Constructor
    StreamWriter() {}

    virtual void open(const std::string &filename);
    virtual std::pair<char *, size_type> open();
    virtual void close();
    virtual void writeVertices(size_type first, size_type count, const float *data);
    virtual void writeTriangles(size_type first, size_type count, const std::tr1::uint32_t *data);
    virtual bool supportsOutOfOrder() const;

private:
    /// Code shared by both @c open methods.
    void openCommon(const std::string &header);

    /**
     * Output stream. It is wrapped in a smart pointer because the type depends
     * on which open function was used.
     */
    boost::scoped_ptr<std::ostream> file;

    /// Position in file where vertices start
    boost::iostreams::stream_offset vertexOffset;
    /// Position in file where triangles start
    boost::iostreams::stream_offset triangleOffset;
};


/**
 * Factory function to create a new reader of the specified type.
 */
ReaderBase *createReader(ReaderType type, const std::string &filename, float smooth);

/**
 * Factory function to create a new writer of the specified type.
 */
WriterBase *createWriter(WriterType type);


template<typename OutputIterator>
void ReaderBase::Handle::loadSplat(const char *buffer, OutputIterator out) const
{
    Splat tmp;
    loadSplat(buffer, &tmp);
    *out = tmp;
}

template<>
void ReaderBase::Handle::loadSplat<Splat *>(const char *buffer, Splat *out) const;

template<typename OutputIterator>
OutputIterator ReaderBase::Handle::read(size_type first, size_type last, OutputIterator out) const
{
    MLSGPU_ASSERT(first <= last && last <= owner.size(), std::out_of_range);

    const size_type vertexSize = owner.getVertexSize();
    size_type blockSize;
    if (bufferSize == 0)
        blockSize = last - first;
    else
        blockSize = bufferSize / vertexSize;
    MLSGPU_ASSERT(blockSize > 0, std::length_error);

    for (size_type i = first; i < last; i += blockSize)
    {
        size_type blockEnd = std::min(last, i + blockSize);
        const char *data = readRaw(i, blockEnd, buffer.get());
        for (size_type j = i; j < blockEnd; j++)
        {
            loadSplat(data, out);
            data += vertexSize;
            ++out;
        }
    }
    return out;
}

} // namespace FastPly

#endif /* FAST_PLY_H */
