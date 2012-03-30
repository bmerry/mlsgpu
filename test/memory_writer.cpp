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

MemoryWriter::MemoryWriter() : curOutput(NULL)
{
}

void MemoryWriter::open(const std::string &filename)
{
    MLSGPU_ASSERT(!isOpen(), state_error);

    // NaN is tempting, but violates the Strict Weak Ordering requirements used to
    // check for isometry.
    const boost::array<float, 3> badVertex = {{ -1000.0f, -1000.0f, -1000.0f }};
    const boost::array<std::tr1::uint32_t, 3> badTriangle = {{ UINT32_MAX, UINT32_MAX, UINT32_MAX }};

    curOutput = &outputs[filename];
    // Clear any previous data that might have been written
    curOutput->vertices.clear();
    curOutput->triangles.clear();
    curOutput->vertices.resize(getNumVertices(), badVertex);
    curOutput->triangles.resize(getNumTriangles(), badTriangle);
    setOpen(true);
}

std::pair<char *, MemoryWriter::size_type> MemoryWriter::open()
{
    open("");
    return std::make_pair((char *) NULL, size_type(0));
}

void MemoryWriter::close()
{
    setOpen(false);
    curOutput = NULL;
}

void MemoryWriter::writeVertices(size_type first, size_type count, const float *data)
{
    MLSGPU_ASSERT(isOpen(), state_error);
    MLSGPU_ASSERT(first + count <= getNumVertices() && first <= std::numeric_limits<size_type>::max() - count, std::out_of_range);

    std::memcpy(&curOutput->vertices[first][0], data, count * 3 * sizeof(float));
}

void MemoryWriter::writeTriangles(size_type first, size_type count, const std::tr1::uint32_t *data)
{
    MLSGPU_ASSERT(isOpen(), state_error);
    MLSGPU_ASSERT(first + count <= getNumTriangles() && first <= std::numeric_limits<size_type>::max() - count, std::out_of_range);

    std::memcpy(&curOutput->triangles[first][0], data, count * 3 * sizeof(std::tr1::uint32_t));
}

bool MemoryWriter::supportsOutOfOrder() const
{
    return true;
}

const MemoryWriter::Output &MemoryWriter::getOutput(const std::string &filename) const
{
    std::tr1::unordered_map<std::string, Output>::const_iterator pos = outputs.find(filename);
    if (pos == outputs.end())
        throw std::invalid_argument("No such output file `" + filename + "'");
    return pos->second;
}
