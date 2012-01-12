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
#include <ostream>
#include <string>
#include <vector>
#include <tr1/cstdint>
#include <boost/iostreams/device/mapped_file.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/smart_ptr/scoped_ptr.hpp>

class Splat;
class TestFastPlyReader;

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
 * Class for quickly reading a subset of PLY files.
 * It only supports the following:
 * - Binary files, endianness matching the host.
 * - Only the "vertex" element is loaded.
 * - The "vertex" element must be the first element in the file.
 * - The x, y, z, nx, ny, nz, radius elements must all be present and FLOAT32.
 * - The vertex element must not contain any lists.
 * - It must be possible to mmap the entire file (thus, a 64-bit
 *   address space is needed to handle very large files).
 *
 * In addition to memory-mapping a file, it can also accept an existing
 * memory range (this is mainly provided to simplify testing).
 */
class Reader
{
    friend class ::TestFastPlyReader;
public:
    /// Size capable of holding maximum supported file size
    typedef boost::iostreams::mapped_file_source::size_type size_type;

    /// Construct from a file
    explicit Reader(const std::string &filename);

    /// Construct from an existing memory range
    Reader(const char *data, size_type size);

    /// Number of vertices in the file
    size_type numVertices() const { return vertexCount; }

    /**
     * Copy out a contiguous selection of the vertices.
     * @param first            First vertex to copy.
     * @param count            Number of vertices to copy.
     * @param out              Target of copy.
     * @pre @a first + @a count <= @a numVertices.
     */
    void readVertices(size_type first, size_type count, Splat *out);
private:
    /// The memory mapping, if constructed from a filename; otherwise @c NULL.
    boost::scoped_ptr<boost::iostreams::mapped_file_source> mapping;

    /// Pointer to the start of the whole file.
    const char *filePtr;
    /// Pointer to the first vertex.
    const char *vertexPtr;

    /// The properties found in the file.
    enum Property
    {
        X, Y, Z,
        NX, NY, NZ,
        RADIUS
    };
    static const unsigned int numProperties = 7;
    size_type vertexSize;              ///< Bytes per vertex
    size_type vertexCount;             ///< Number of vertices
    size_type offsets[numProperties];  ///< Byte offsets of each property within a vertex

    void readHeader(std::istream &in); ///< Does the heavy lifting of parsing the header
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
 */
class Writer
{
public:
    /// Size capable of holding maximum supported file size
    typedef boost::iostreams::mapped_file_source::size_type size_type;

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
     * Determines whether @ref open has been successfully called.
     */
    bool isOpen();

    /**
     * Write a range of vertices.
     * @param first          Index of first vertex to write.
     * @param count          Number of vertices to write.
     * @param data           Array of <code>float[3]</code> values.
     * @pre @a first + @a count <= @a numVertices.
     */
    void writeVertices(size_type first, size_type count, const float *data);

    /**
     * Write a range of triangles.
     * @param first          Index of first triangle to write.
     * @param count          Number of triangles to write.
     * @param data           Array of <code>uint32_t[3]</code> values containing indices.
     * @pre @a first + @a count <= @a numTriangles.
     */
    void writeTriangles(size_type first, size_type count, const std::tr1::uint32_t *data);

private:
    static const size_type vertexSize = 3 * sizeof(float);
    static const size_type triangleSize = 1 + 3 * sizeof(std::tr1::uint32_t);

    std::vector<std::string> comments;
    size_type numVertices, numTriangles;
    char *vertexPtr;
    char *trianglePtr;
    boost::scoped_ptr<boost::iostreams::mapped_file_sink> mapping;

    void writeHeader(std::ostream &o);
};

} // namespace FastPly

#endif /* FAST_PLY_H */
