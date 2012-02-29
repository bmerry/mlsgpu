/**
 * @file
 *
 * Writer class that stores results in memory for easy testing.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <stdexcept>
#include <string>
#include <boost/array.hpp>
#include <tr1/cstdint>
#include <limits>
#include <cstring>
#include "../src/errors.h"
#include "../src/fast_ply.h"
#include "memory_writer.h"

MemoryWriter::MemoryWriter()
{
}

void MemoryWriter::open(const std::string &filename)
{
    MLSGPU_ASSERT(!isOpen(), std::runtime_error);

    // Ignore the filename
    (void) filename;

    // NaN is tempting, but violates the Strict Weak Ordering requirements
    const boost::array<float, 3> badVertex = {{ -1000.0f, -1000.0f, -1000.0f }};
    const boost::array<std::tr1::uint32_t, 3> badTriangle = {{ UINT32_MAX, UINT32_MAX, UINT32_MAX }};
    vertices.resize(getNumVertices(), badVertex);
    triangles.resize(getNumTriangles(), badTriangle);
    setOpen(true);
}

std::pair<char *, MemoryWriter::size_type> MemoryWriter::open()
{
    MLSGPU_ASSERT(!isOpen(), std::runtime_error);

    vertices.resize(getNumVertices());
    triangles.resize(getNumTriangles());
    setOpen(true);

    return std::make_pair((char *) NULL, size_type(0));
}

void MemoryWriter::close()
{
    setOpen(false);
}

void MemoryWriter::writeVertices(size_type first, size_type count, const float *data)
{
    MLSGPU_ASSERT(isOpen(), std::runtime_error);
    MLSGPU_ASSERT(first + count <= getNumVertices() && first <= std::numeric_limits<size_type>::max() - count, std::out_of_range);

    std::memcpy(&vertices[first][0], data, count * 3 * sizeof(float));
}

void MemoryWriter::writeTriangles(size_type first, size_type count, const std::tr1::uint32_t *data)
{
    MLSGPU_ASSERT(isOpen(), std::runtime_error);
    MLSGPU_ASSERT(first + count <= getNumTriangles() && first <= std::numeric_limits<size_type>::max() - count, std::out_of_range);

    std::memcpy(&triangles[first][0], data, count * 3 * sizeof(std::tr1::uint32_t));
}

bool MemoryWriter::supportsOutOfOrder() const
{
    return true;
}


