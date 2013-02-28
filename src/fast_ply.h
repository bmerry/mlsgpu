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
#include <boost/filesystem/path.hpp>
#include <boost/function.hpp>
#include <boost/smart_ptr/scoped_ptr.hpp>
#include <boost/smart_ptr/shared_ptr.hpp>
#include <boost/smart_ptr/scoped_array.hpp>
#include <boost/type_traits/is_pointer.hpp>
#include <boost/ref.hpp>
#include <boost/noncopyable.hpp>
#include "splat.h"
#include "errors.h"
#include "allocator.h"
#include "binary_io.h"
#include "async_io.h"
#include "timeplot.h"

class TestFastPlyReader;

/**
 * Classes and functions for efficient access to PLY files in a narrow
 * range of possible formats.
 */
namespace FastPly
{

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
 * An instance of this class just holds the metadata, but no OS resources or
 * buffers. To actually read the data, one creates a @ref Handle,
 * at which point the file is opened.
 */
class Reader
{
    friend class ::TestFastPlyReader;
public:
    /// Size capable of holding maximum supported file size
    typedef BinaryReader::offset_type size_type;
    typedef Splat value_type;

    /**
     * A file handle to a PLY file, used for reading.
     *
     * It is safe for two threads to simultaneously read from the same handle.
     */
    class Handle : public boost::noncopyable
    {
    protected:
        /// The reader whose file we're reading
        const Reader &owner;
        /// Low-level handle
        boost::scoped_ptr<BinaryReader> reader;

    public:
        /// Constructor
        Handle(const Reader &owner);

        /**
         * Low-level read access. The vertices are copied to @a buffer, in
         * exactly the form they exist in the file i.e. just a byte-level copy
         * of the relevant data, without selecting the required fields.
         *
         * @param first,last      %Range of vertices to read.
         * @param buffer          Output buffer.
         * @return A pointer to the data.
         *
         * @pre @a first &lt;= @a last &lt;= @ref size().
         * @pre @a buffer is not @c NULL and has at least <code>(last - first) * owner.getVertexSize() bytes</code>.
         */
        void readRaw(size_type first, size_type last, char *buffer) const;

        /**
         * Convenience wrapper around @ref Reader::decode.
         *
         * @see @ref Reader::decode.
         */
        Splat decode(const char *buffer, std::size_t offset) const
        {
            return owner.decode(buffer, offset);
        }

        /**
         * Copy out a contiguous selection of the vertices.
         *
         * @warning This is a low-performance function intended only for testing.
         * High-performance code should directly use @ref readRaw and @ref decode
         * in order to manage the buffer and be able to control overlap of
         * I/O and decoding.
         *
         * @param first,last      %Range of vertices to copy.
         * @param out             Target of copy.
         * @return The output iterator after the copy.
         * @pre @a first &lt;= @a last &lt;= @ref size().
         */
        template<typename OutputIterator>
            OutputIterator read(size_type first, size_type last, OutputIterator out) const;
    };

    /**
     * Extract a single splat from the raw buffer representation.
     *
     * @param buffer     A buffer returned by @ref Handle::readRaw
     * @param offset     The number of the splat within the buffer
     * @return The splat at the specified offset, with a computed quality
     */
    Splat decode(const char *buffer, std::size_t offset) const;

    /// Number of vertices in the file
    size_type size() const { return vertexCount; }

    /// Number of bytes per vertex
    size_type getVertexSize() const { return vertexSize; }

    /**
     * Construct from a file.
     *
     * This will open the file to parse the header and then close it again.
     * If an exception is thrown, it will have the filename stored in it
     * using @c boost::errinfo_file_name.
     *
     * @param readerType       Type to use for binary file access
     * @param path             File to open.
     * @param smooth           Scale factor applied to radii as they're read.
     * @param maxRadius        Cap for radius (prior to scaling by @a smooth).
     * @throw FormatError if the header is malformed.
     * @throw std::ios::failure if there was an I/O error.
     */
    Reader(
        ReaderType readerType,
        const boost::filesystem::path &path,
        float smooth, float maxRadius);

    /**
     * Construct from a filename, using a custom factory to generate the
     * underlying @ref BinaryReader.
     */
    Reader(
        boost::function<BinaryReader *()> readerFactory,
        const boost::filesystem::path &path,
        float smooth, float maxRadius);

private:
    /// Factory to generate file handles for low-level file access
    boost::function<BinaryReader *()> readerFactory;

    /// Path to the file
    boost::filesystem::path path;

    /// Scale factor for radii
    float smooth;

    /// Radius limit
    float maxRadius;

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
 * PLY file writer that only supports one format.
 * The supported format has:
 *  - Binary format with host endianness;
 *  - Vertices with x, y, z as 32-bit floats (no normals);
 *  - Faces with 32-bit unsigned integer indices;
 *  - 3 indices per face;
 *  - Arbitrary user-provided comments.
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
 */
class Writer
{
public:
    /// Size capable of holding maximum supported file size
    typedef BinaryWriter::offset_type size_type;

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
    void open(const std::string &filename);

    /**
     * Prepare to write another file. This will usually cause the old file
     * to be closed, but if it has been used with the asynchronous write
     * functions it will remain open until there are no remaining references.
     */
    void close();

    /**
     * Write a range of vertices.
     * @param first          Index of first vertex to write.
     * @param count          Number of vertices to write.
     * @param data           Array of <code>float[3]</code> values.
     * @pre @a first + @a count <= @a numVertices.
     */
    void writeVertices(size_type first, size_type count, const float *data);

    /**
     * Write vertices asynchronously. The caller must have obtained the item
     * from the asynchronous writer and populated it.
     * @param tworker        Worker for accounting the time (possibly unused?)
     * @param first          Index of first vertex to write.
     * @param count          Number of vertices to write.
     * @param data           Array of <code>float[3]</code> values.
     * @param async          Asynchronous writer that will do the writing.
     * @pre @a first + @a count <= @a numVertices.
     */
    void writeVertices(
        Timeplot::Worker &tworker,
        size_type first, size_type count,
        const boost::shared_ptr<AsyncWriterItem> &data,
        AsyncWriter &async);

    /**
     * Write a range of triangles.
     * @param first          Index of first triangle to write.
     * @param count          Number of triangles to write.
     * @param data           Array of <code>uint32_t[3]</code> values containing indices.
     * @pre @a first + @a count <= @a numTriangles.
     */
    void writeTriangles(size_type first, size_type count, const std::tr1::uint32_t *data);

    /**
     * Write a range of triangles which have been pre-encoded. Each triangle must contain
     * 13 bytes, the first of which is 3, followed by the 3 32-bit indices. If this class
     * ever supports endian conversion, they must be in the file endianness. In other words,
     * the data must be ready to be written to the file with no further conversions.
     *
     * @param first          Index of first triangle to write.
     * @param count          Number of triangles to write.
     * @param data           Raw data, as described above.
     * @pre @a first + @a count <= @a numTriangles.
     */
    void writeTrianglesRaw(size_type first, size_type count, const std::tr1::uint8_t *data);

    /**
     * Write triangles asynchronously.
     *
     * @param tworker        Worker for accounting the time (possibly unused?)
     * @param first          Index of first triangle to write.
     * @param count          Number of triangles to write.
     * @param data           Raw data, as for @ref writeTrianglesRaw
     * @param async          Asynchronous writer that will do the writing.
     * @pre @a first + @a count <= @a numTriangles.
     */
    void writeTrianglesRaw(
        Timeplot::Worker &tworker,
        size_type first, size_type count,
        const boost::shared_ptr<AsyncWriterItem> &data,
        AsyncWriter &async);

    size_type getNumVertices() const;
    size_type getNumTriangles() const;

    /// Bytes per vertex
    static const size_type vertexSize = 3 * sizeof(float);
    /// Bytes per triangle
    static const size_type triangleSize = 1 + 3 * sizeof(std::tr1::uint32_t);

    /// Constructor
    explicit Writer(WriterType writerType);

    /**
     * Constructor with a custom low-level writer.
     */
    explicit Writer(boost::function<boost::shared_ptr<BinaryWriter>()> handleFactory);

private:
    class InternalFactory
    {
    private:
        const WriterType writerType;
    public:
        typedef boost::shared_ptr<BinaryWriter> result_type;

        result_type operator()() { return result_type(createWriter(writerType)); }
        explicit InternalFactory(WriterType writerType) : writerType(writerType) {}
    };

    Statistics::Variable &writeVerticesTime;
    Statistics::Variable &writeTrianglesTime;

    /// Handle factory, used when the file is closed to make a new handle
    boost::function<boost::shared_ptr<BinaryWriter>()> handleFactory;

    /// Storage for comments until they can be written by @ref open.
    std::vector<std::string> comments;
    size_type numVertices;              ///< Number of vertices (defaults to zero)
    size_type numTriangles;             ///< Number of triangles (defaults to zero)

protected:
    /// File handle (non-NULL if the file is open)
    boost::shared_ptr<BinaryWriter> handle;

    BinaryWriter::offset_type vertexStart;   ///< Offset in file to start of vertices
    BinaryWriter::offset_type triangleStart; ///< Offset in file to start of triangles

    /// Returns the header based on stored values
    std::string makeHeader();
};


template<typename OutputIterator>
OutputIterator Reader::Handle::read(size_type first, size_type last, OutputIterator out) const
{
    MLSGPU_ASSERT(first <= last && last <= owner.size(), std::out_of_range);

    const std::size_t vertexSize = owner.getVertexSize();
    const std::size_t bufferSize = std::max(vertexSize, std::size_t(4096));
    const std::size_t blockSize = bufferSize / vertexSize;
    boost::scoped_array<char> buffer(new char[bufferSize]);

    for (size_type i = first; i < last; i += blockSize)
    {
        size_type blockEnd = std::min(last, i + blockSize);
        readRaw(i, blockEnd, buffer.get());
        for (size_type j = i; j < blockEnd; j++)
        {
            *out++ = decode(buffer.get(), j - i);
        }
    }
    return out;
}

} // namespace FastPly

#endif /* FAST_PLY_H */
