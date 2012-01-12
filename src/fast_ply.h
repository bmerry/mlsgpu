/**
 * @file
 *
 * PLY file format loading and saving routines optimized for just the
 * operations desired in mlsgpu.
 *
 * It only supports the following for input:
 * - Binary files, endianness matching the host.
 * - Only the "vertex" element is loaded.
 * - The "vertex" element must be the first element in the file.
 * - The x, y, z, nx, ny, nz, radius elements must all be FLOAT32.
 * - The vertex element must not contain any lists.
 * - It must be possible to mmap the entire file (thus, a 64-bit
 *   address space is needed to handle very large files).
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

namespace FastPLY
{

/**
 * An exception that is thrown when an invalid PLY file is encountered.
 * This is used to signal all errors in a PLY file (including early end-of-file).
 */
class FormatError : public std::runtime_error
{
public:
    FormatError(const std::string &msg) : std::runtime_error(msg) {}
};

class Reader
{
public:
    typedef boost::iostreams::mapped_file_source::size_type size_type;

    explicit Reader(const std::string &filename);
    Reader(const char *data, size_type size);

    size_type numVertices() const { return vertexCount; }
    void readVertices(size_type first, size_type count, Splat *out);
private:
    boost::scoped_ptr<boost::iostreams::mapped_file_source> mapping;
    const char *filePtr;
    const char *vertexPtr;

    enum Field
    {
        X, Y, Z,
        NX, NY, NZ,
        RADIUS
    };
    static const unsigned int numProperties = 7;
    size_type vertexSize;
    size_type vertexCount;
    size_type offsets[numProperties];

    void readHeader(std::istream &in);
};

} // namespace FastPLY

#endif /* FAST_PLY_H */
