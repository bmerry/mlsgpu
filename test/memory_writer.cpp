/**
 * @file
 *
 * Writer class that stores results in memory for easy testing.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <string>
#include <algorithm>
#include <locale>
#include <sstream>
#include <boost/lexical_cast.hpp>
#include "../src/binary_io.h"
#include "memory_reader.h"
#include "memory_writer.h"

MemoryWriter::MemoryWriter() : curOutput(NULL)
{
}

void MemoryWriter::openImpl(const boost::filesystem::path &filename)
{
    curOutput = &outputs[filename.string()];
    // Clear any previous data that might have been written
    curOutput->clear();
}

void MemoryWriter::closeImpl()
{
    curOutput = NULL;
}

std::size_t MemoryWriter::writeImpl(const void *buffer, std::size_t count, offset_type offset) const
{
    if (curOutput->size() < count + offset)
        curOutput->resize(count + offset);

    const char *p = (const char *) buffer;
    std::copy(p, p + count, curOutput->begin() + offset);
    return count;
}

void MemoryWriter::resizeImpl(offset_type size) const
{
    curOutput->resize(size);
}

const std::string &MemoryWriter::getOutput(const std::string &filename) const
{
    std::tr1::unordered_map<std::string, std::string>::const_iterator pos = outputs.find(filename);
    if (pos == outputs.end())
        throw std::invalid_argument("No such output file `" + filename + "'");
    return pos->second;
}

MemoryWriterPly::MemoryWriterPly()
    : FastPly::Writer(new MemoryWriter)
{
}

const std::string &MemoryWriterPly::getOutput(const std::string &filename)
{
    return static_cast<MemoryWriter *>(getHandle())->getOutput(filename);
}

void MemoryWriterPly::parse(
    const std::string &content,
    std::vector<boost::array<float, 3> > &vertices,
    std::vector<boost::array<std::tr1::uint32_t, 3> > &triangles)
{
    std::istringstream in(content);
    in.imbue(std::locale::classic());
    std::string line;
    const std::string vertexPrefix = "element vertex ";
    const std::string trianglePrefix = "element face ";

    std::size_t numVertices = 0, numTriangles = 0;
    std::size_t headerSize = 0;
    while (getline(in, line))
    {
        if (line.substr(0, vertexPrefix.size()) == vertexPrefix)
            numVertices = boost::lexical_cast<std::size_t>(line.substr(vertexPrefix.size()));
        else if (line.substr(0, trianglePrefix.size()) == trianglePrefix)
            numTriangles = boost::lexical_cast<std::size_t>(line.substr(trianglePrefix.size()));
        else if (line == "end_header")
        {
            headerSize = in.tellg();
            break;
        }
    }

    MemoryReader handle(content.data(), content.size());
    handle.open("memory"); // filename is irrelevant
    vertices.resize(numVertices);
    handle.read(&vertices[0][0], numVertices * sizeof(vertices[0]), headerSize);

    triangles.resize(numTriangles);
    BinaryReader::offset_type pos = headerSize + numVertices * sizeof(vertices[0]);
    for (std::size_t i = 0; i < numTriangles; i++)
    {
        handle.read(&triangles[i][0], sizeof(triangles[i]), pos + 1);
        pos += 1 + sizeof(triangles[i]);
    }
}
