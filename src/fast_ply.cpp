/**
 * @file
 *
 * Implementation of the FastPly namespace.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif

#if HAVE_PREAD && !defined(_POSIX_C_SOURCE)
# define _POSIX_C_SOURCE 200809L
#endif

#include <string>
#include <cstddef>
#include <string>
#include <iterator>
#include <sstream>
#include <istream>
#include <cstdlib>
#include "tr1_cstdint.h"
#include <cstring>
#include <algorithm>
#include <limits>
#include <cstring>
#include <cerrno>
#include <memory>
#include <locale>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/foreach.hpp>
#include <boost/exception/all.hpp>
#include <boost/bind.hpp>
#include <boost/iostreams/stream.hpp>
#include "fast_ply.h"
#include "splat.h"
#include "errors.h"
#include "binary_io.h"

namespace FastPly
{

/**
 * The type of a field in a PLY file.
 */
enum FieldType
{
    INT8,
    UINT8,
    INT16,
    UINT16,
    INT32,
    UINT32,
    FLOAT32,
    FLOAT64
};

/**
 * Splits a string on whitespace, using operator>>.
 *
 * @param line The string to split.
 * @return A vector of tokens, not containing whitespace.
 */
static std::vector<std::string> splitLine(const std::string &line)
{
    std::istringstream splitter(line);
    return std::vector<std::string>(std::istream_iterator<std::string>(splitter), std::istream_iterator<std::string>());
}

/**
 * Maps the label for a type in the PLY header to a type token.
 * The types int, uint and float are mapped to INT32, UINT32 and FLOAT32
 * respectively.
 *
 * @param t The name of the type from the PLY header.
 * @return The #FieldType value corresponding to @a t.
 * @throw #FastPly::FormatError if @a t is not recognized.
 */
static FieldType parseType(const std::string &t) throw(FormatError)
{
    if (t == "int8" || t == "char") return INT8;
    else if (t == "uint8" || t == "uchar") return UINT8;
    else if (t == "int16") return INT16;
    else if (t == "uint16") return UINT16;
    else if (t == "int32" || t == "int") return INT32;
    else if (t == "uint32" || t == "uint") return UINT32;
    else if (t == "float32" || t == "float") return FLOAT32;
    else if (t == "float64") return FLOAT64;
    else throw boost::enable_error_info(FormatError("Unknown type `" + t + "'"));
}

static Reader::size_type fieldSize(const FieldType f)
{
    switch (f)
    {
    case INT8:
    case UINT8:
        return 1;
    case INT16:
    case UINT16:
        return 2;
    case INT32:
    case UINT32:
    case FLOAT32:
        return 4;
    case FLOAT64:
        return 8;
    }
    std::abort();
    return 0;
}

/**
 * Retrieve a line from the header, throwing a suitable exception on failure.
 *
 * @param in The input stream containing the header
 * @return A line from @a in
 * @throw FormatError on EOF
 * @throw std::ios::failure on other I/O error
 */
static std::string getHeaderLine(std::istream &in) throw(boost::exception)
{
    std::string line;
    if (!getline(in, line))
    {
        if (in.eof())
            throw boost::enable_error_info(FormatError("End of file in PLY header"));
        else if (in.bad())
        {
            throw boost::enable_error_info(std::ios::failure("Failed to read line from header"))
                << boost::errinfo_errno(errno);
        }
        else
        {
            throw boost::enable_error_info(std::ios::failure("Failed to read line from header"));
        }
    }
    return line;
}

static bool cpuLittleEndian()
{
    std::tr1::uint32_t x = 0x12345678;
    std::tr1::uint8_t y[4];

    std::memcpy(y, &x, 4);
    return y[0] == 0x78 && y[1] == 0x56 && y[2] == 0x34 && y[3] == 0x12;
}

static bool cpuBigEndian()
{
    std::tr1::uint32_t x = 0x12345678;
    std::tr1::uint8_t y[4];

    std::memcpy(y, &x, 4);
    return y[0] == 0x12 && y[1] == 0x34 && y[2] == 0x56 && y[3] == 0x78;
}

void Reader::readHeader(std::istream &in)
{
    try
    {
        static const char * const propertyNames[numProperties] =
        {
            "x", "y", "z", "nx", "ny", "nz", "radius"
        };

        vertexSize = 0;
        size_type elements = 0;
        bool haveProperty[numProperties] = {};

        std::string line = getHeaderLine(in);
        if (line != "ply")
            throw boost::enable_error_info(FormatError("PLY signature missing"));

        // read the header
        bool haveFormat = false;
        while (true)
        {
            std::vector<std::string> tokens;

            line = getHeaderLine(in);
            tokens = splitLine(line);
            if (tokens.empty())
                continue; // ignore blank lines
            if (tokens[0] == "end_header")
                break;
            else if (tokens[0] == "format")
            {
                if (tokens.size() != 3)
                    throw boost::enable_error_info(FormatError("Malformed format line"));

                if (tokens[1] == "ascii")
                    throw boost::enable_error_info(FormatError("PLY ASCII format not supported"));
                else if (tokens[1] == "binary_big_endian")
                {
                    if (!cpuBigEndian())
                        throw boost::enable_error_info(FormatError("PLY big endian format not supported on this CPU"));
                }
                else if (tokens[1] == "binary_little_endian")
                {
                    if (!cpuLittleEndian())
                        throw boost::enable_error_info(FormatError("PLY little endian format not supported on this CPU"));
                }
                else
                {
                    throw boost::enable_error_info(FormatError("Unknown PLY format " + tokens[1]));
                }

                if (tokens[2] != "1.0")
                    throw boost::enable_error_info(FormatError("Unknown PLY version " + tokens[2]));

                haveFormat = true;
            }
            else if (tokens[0] == "element")
            {
                if (tokens.size() != 3)
                    throw boost::enable_error_info(FormatError("Malformed element line"));
                std::string elementName = tokens[1];
                size_type elementCount;
                try
                {
                    elementCount = boost::lexical_cast<size_type>(tokens[2]);
                }
                catch (boost::bad_lexical_cast &e)
                {
                    throw boost::enable_error_info(FormatError("Malformed element line or too many elements"));
                }

                if (elements == 0)
                {
                    /* Expect the vertex element */
                    if (elementName != "vertex")
                        throw boost::enable_error_info(FormatError("First element is not vertex"));
                    vertexCount = elementCount;
                }
                elements++;
            }
            else if (tokens[0] == "property")
            {
                if (tokens.size() < 3)
                    throw boost::enable_error_info(FormatError("Malformed property line"));
                bool isList;
                FieldType lengthType, valueType;
                std::string name;

                if (tokens[1] == "list")
                {
                    if (tokens.size() != 5)
                        throw boost::enable_error_info(FormatError("Malformed property line"));
                    isList = true;
                    lengthType = parseType(tokens[2]);
                    valueType = parseType(tokens[3]);
                    if (lengthType == FLOAT32 || lengthType == FLOAT64)
                        throw boost::enable_error_info(FormatError("List cannot have floating-point count"));
                    name = tokens[4];
                }
                else
                {
                    if (tokens.size() != 3)
                        throw boost::enable_error_info(FormatError("Malformed property line"));
                    isList = false;
                    lengthType = INT32; // unused, just to avoid undefined values
                    valueType = parseType(tokens[1]);
                    name = tokens[2];
                }

                if (elements == 0)
                    throw boost::enable_error_info(FormatError("Property `" + name + "' appears before any element declaration"));
                if (elements == 1)
                {
                    /* Vertex element - match it up to the expected fields */
                    if (isList)
                        throw boost::enable_error_info(FormatError("Lists in a vertex are not supported"));
                    for (unsigned int i = 0; i < numProperties; i++)
                    {
                        if (name == propertyNames[i])
                        {
                            if (haveProperty[i])
                                throw boost::enable_error_info(FormatError("Duplicate property " + name));
                            if (valueType != FLOAT32)
                                throw boost::enable_error_info(FormatError("Property " + name + " must be FLOAT32"));
                            haveProperty[i] = true;
                            offsets[i] = vertexSize;
                            break;
                        }
                    }
                    vertexSize += fieldSize(valueType);
                }
            }
        }

        if (!haveFormat)
            throw boost::enable_error_info(FormatError("No format line found"));

        if (elements < 1)
            throw boost::enable_error_info(FormatError("No elements found"));

        for (unsigned int i = 0; i < numProperties; i++)
            if (!haveProperty[i])
                throw boost::enable_error_info(FormatError(std::string("Property ") + propertyNames[i] + " not found"));

        headerSize = in.tellg();
    }
    catch (boost::exception &e)
    {
        e << boost::errinfo_file_name(path.string());
        throw;
    }
}

Splat Reader::decode(const char *buffer, std::size_t offset) const
{
    buffer += offset * getVertexSize();

    Splat ans;
    std::memcpy(&ans.position[0], buffer + offsets[X], sizeof(float));
    std::memcpy(&ans.position[1], buffer + offsets[Y], sizeof(float));
    std::memcpy(&ans.position[2], buffer + offsets[Z], sizeof(float));
    std::memcpy(&ans.radius,      buffer + offsets[RADIUS], sizeof(float));
    std::memcpy(&ans.normal[0],   buffer + offsets[NX], sizeof(float));
    std::memcpy(&ans.normal[1],   buffer + offsets[NY], sizeof(float));
    std::memcpy(&ans.normal[2],   buffer + offsets[NZ], sizeof(float));
    ans.radius = std::min(ans.radius, maxRadius);
    ans.radius *= smooth;
    ans.quality = 1.0 / (ans.radius * ans.radius);
    return ans;
}

Reader::Reader(
    ReaderType readerType,
    const boost::filesystem::path &path,
    float smooth, float maxRadius)
    : readerFactory(boost::bind(createReader, readerType)), path(path), smooth(smooth), maxRadius(maxRadius)
{
    boost::scoped_ptr<BinaryReader> reader(readerFactory());
    reader->open(path);
    boost::iostreams::stream<BinaryReaderSource> in(*reader);
    readHeader(in);
}

Reader::Reader(
    boost::function<BinaryReader *()> readerFactory,
    const boost::filesystem::path &path,
    float smooth, float maxRadius)
    : readerFactory(readerFactory), path(path), smooth(smooth), maxRadius(maxRadius)
{
    boost::scoped_ptr<BinaryReader> reader(readerFactory());
    reader->open(path);
    boost::iostreams::stream<BinaryReaderSource> in(*reader);
    readHeader(in);
}

Reader::Handle::Handle(const Reader &owner)
    : owner(owner), reader(owner.readerFactory())
{
    reader->open(owner.path);
    if ((reader->size() - owner.getHeaderSize()) / owner.getVertexSize() < owner.size())
        throw boost::enable_error_info(std::ios::failure("File is too small to contain all its vertices"))
            << boost::errinfo_file_name(owner.path.string());
}

void Reader::Handle::readRaw(size_type first, size_type last, char *buffer) const
{
    MLSGPU_ASSERT(first <= last, std::invalid_argument);
    MLSGPU_ASSERT(buffer != NULL, std::invalid_argument);
    const std::size_t vertexSize = owner.getVertexSize();
    reader->read(buffer, (last - first) * vertexSize, owner.getHeaderSize() + first * vertexSize);
}


bool Writer::isOpen() const
{
    return handle->isOpen();
}

void Writer::addComment(const std::string &comment)
{
    MLSGPU_ASSERT(!isOpen(), state_error);
    comments.push_back(comment);
}

void Writer::setNumVertices(size_type numVertices)
{
    MLSGPU_ASSERT(!isOpen(), state_error);
    this->numVertices = numVertices;
}

void Writer::setNumTriangles(size_type numTriangles)
{
    MLSGPU_ASSERT(!isOpen(), state_error);
    this->numTriangles = numTriangles;
}

Writer::Writer(WriterType writerType) : 
    writeVerticesTime(Statistics::getStatistic<Statistics::Variable>("writer.writeVertices.time")),
    writeTrianglesTime(Statistics::getStatistic<Statistics::Variable>("writer.writeTriangles.time")),
    handleFactory(InternalFactory(writerType)),
    comments(), numVertices(0), numTriangles(0)
{
    handle = handleFactory();
}

Writer::Writer(boost::function<boost::shared_ptr<BinaryWriter>()> handleFactory) : 
    writeVerticesTime(Statistics::getStatistic<Statistics::Variable>("writer.writeVertices.time")),
    writeTrianglesTime(Statistics::getStatistic<Statistics::Variable>("writer.writeTriangles.time")),
    handleFactory(handleFactory),
    comments(), numVertices(0), numTriangles(0),
    handle(handleFactory())
{
    handle = this->handleFactory();
}

Writer::size_type Writer::getNumVertices() const
{
    return numVertices;
}

Writer::size_type Writer::getNumTriangles() const
{
    return numTriangles;
}

std::string Writer::makeHeader()
{
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out.exceptions(std::ios::failbit | std::ios::badbit);
    out << "ply\n";
    if (cpuLittleEndian())
        out << "format binary_little_endian 1.0\n";
    else if (cpuBigEndian())
        out << "format binary_big_endian 1.0\n";
    else
        throw std::runtime_error("CPU is neither big- nor little-endian");

    BOOST_FOREACH(const std::string &s, comments)
    {
        out << "comment " << s << '\n';
    }

    out << "element vertex " << numVertices << '\n'
        << "property float32 x\n"
        << "property float32 y\n"
        << "property float32 z\n"
        << "element face " << numTriangles << '\n'
        << "property list uint8 uint32 vertex_indices\n"
        << "comment padding:";
    /* Use a comment to pad the header to a multiple of 4 bytes, so that the
     * vertex data will be nicely aligned.
     */

    std::size_t size = (int) out.tellp() + 12; /* 12 for \nend_header\n */
    while (size % 4 != 0)
    {
        out << 'X';
        size++;
    }
    out << "\nend_header\n";
    assert(out.str().size() % 4 == 0);
    return out.str();
}

void Writer::open(const std::string &filename)
{
    handle->open(filename);

    std::string header = makeHeader();
    handle->resize(header.size() + getNumVertices() * vertexSize + getNumTriangles() * triangleSize);
    handle->write(header.data(), header.size(), 0);
    vertexStart = header.size();
    triangleStart = vertexStart + getNumVertices() * vertexSize;
}

void Writer::close()
{
    // Note: the handle is not closed, because it may still be accessed by an AsyncWriter
    handle = handleFactory();
}

void Writer::writeVertices(size_type first, size_type count, const float *data)
{
    MLSGPU_ASSERT(first + count <= getNumVertices() && first <= std::numeric_limits<size_type>::max() - count, std::out_of_range);
    Statistics::Timer timer(writeVerticesTime);
    handle->write(data, count * vertexSize, vertexStart + first * vertexSize);
}

void Writer::writeVertices(
    Timeplot::Worker &tworker,
    size_type first, size_type count,
    const boost::shared_ptr<AsyncWriterItem> &data,
    AsyncWriter &async)
{
    MLSGPU_ASSERT(first + count <= getNumVertices() && first <= std::numeric_limits<size_type>::max() - count, std::out_of_range);
    async.push(tworker, data, handle, count * vertexSize, vertexStart + first * vertexSize);
}

void Writer::writeTrianglesRaw(size_type first, size_type count, const std::tr1::uint8_t *data)
{
    MLSGPU_ASSERT(first + count <= getNumTriangles() && first <= std::numeric_limits<size_type>::max() - count, std::out_of_range);
    Statistics::Timer timer(writeTrianglesTime);
    handle->write(data, count * triangleSize, triangleStart + first * triangleSize);
}

void Writer::writeTrianglesRaw(
    Timeplot::Worker &tworker,
    size_type first, size_type count,
    const boost::shared_ptr<AsyncWriterItem> &data,
    AsyncWriter &async)
{
    MLSGPU_ASSERT(first + count <= getNumTriangles() && first <= std::numeric_limits<size_type>::max() - count, std::out_of_range);
    async.push(tworker, data, handle, count * triangleSize, triangleStart + first * triangleSize);
}

void Writer::writeTriangles(size_type first, size_type count, const std::tr1::uint32_t *data)
{
    MLSGPU_ASSERT(first + count <= getNumTriangles() && first <= std::numeric_limits<size_type>::max() - count, std::out_of_range);

    Statistics::Timer timer(writeTrianglesTime);
    BinaryWriter::offset_type pos = triangleStart + first * triangleSize;
    while (count > 0)
    {
        const unsigned int bufferTriangles = 8192;
        char buffer[bufferTriangles * triangleSize];
        char *ptr = buffer;
        unsigned int triangles = std::min(size_type(bufferTriangles), count);
        for (unsigned int i = 0; i < triangles; i++, ptr += triangleSize, data += 3)
        {
            ptr[0] = 3;
            std::memcpy(ptr + 1, data, 3 * sizeof(data[0]));
        }
        pos += handle->write(buffer, ptr - buffer, pos);
        count -= triangles;
    }
}

} // namespace FastPly
