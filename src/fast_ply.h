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
     * @throw std::out_of_range if @a first + @a count is greater than the number of vertices.
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

} // namespace FastPly

#endif /* FAST_PLY_H */
