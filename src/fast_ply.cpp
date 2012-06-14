/**
 * @file
 *
 * Implementation of the FastPly namespace.
 */

#if HAVE_CONFIG_H
# include <config.h>
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
#include <boost/iostreams/device/mapped_file.hpp>
#include <boost/iostreams/device/array.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/foreach.hpp>
#include <boost/exception/all.hpp>
#include "fast_ply.h"
#include "splat.h"
#include "errors.h"

#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>

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

static ReaderBase::size_type fieldSize(const FieldType f)
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
static std::string getHeaderLine(std::istream &in) throw(FormatError)
{
    try
    {
        std::string line;
        getline(in, line);
        return line;
    }
    catch (std::ios::failure &e)
    {
        if (in.eof())
            throw boost::enable_error_info(FormatError("End of file in PLY header"));
        else
            throw;
    }
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

void ReaderBase::readHeader(std::istream &in)
{
    static const char * const propertyNames[numProperties] =
    {
        "x", "y", "z", "nx", "ny", "nz", "radius"
    };

    vertexSize = 0;
    size_type elements = 0;
    bool haveProperty[numProperties] = {};

    in.exceptions(std::ios::failbit);
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

ReaderBase::ReaderBase(const std::string &filename, float smooth)
    : smooth(smooth)
{
    try
    {
        std::ifstream in(filename.c_str());
        readHeader(in);
    }
    catch (boost::exception &e)
    {
        e << boost::errinfo_file_name(filename);
        throw;
    }
}

ReaderBase::ReaderBase(float smooth) : smooth(smooth)
{
}

ReaderBase::Handle::Handle(const ReaderBase &owner, std::size_t bufferSize)
    : owner(owner), bufferSize(bufferSize)
{
    if (bufferSize != 0)
    {
        if (owner.getVertexSize() > bufferSize)
            throw std::runtime_error("The vertices each take too many bytes");
        buffer.reset(new char[bufferSize]);
    }
}

ReaderBase::Handle::Handle(const ReaderBase &owner, boost::shared_array<char> buffer, std::size_t bufferSize)
    : owner(owner), buffer(buffer), bufferSize(bufferSize)
{
    if (owner.getVertexSize() > bufferSize)
        throw std::runtime_error("The vertices each take too many bytes");
}

template<>
void ReaderBase::Handle::loadSplat<Splat *>(const char *buffer, Splat *out) const
{
    float radius;
    std::memcpy(&out->position[0], buffer + owner.offsets[X], sizeof(float));
    std::memcpy(&out->position[1], buffer + owner.offsets[Y], sizeof(float));
    std::memcpy(&out->position[2], buffer + owner.offsets[Z], sizeof(float));
    std::memcpy(&radius,           buffer + owner.offsets[RADIUS], sizeof(float));
    std::memcpy(&out->normal[0],   buffer + owner.offsets[NX], sizeof(float));
    std::memcpy(&out->normal[1],   buffer + owner.offsets[NY], sizeof(float));
    std::memcpy(&out->normal[2],   buffer + owner.offsets[NZ], sizeof(float));
    radius *= owner.smooth;
    out->radius = radius;
    out->quality = 1.0 / (radius * radius);
}

ReaderBase::Handle *ReaderBase::createHandle() const
{
    return createHandle(128 * 1024 * 1024);
}

ReaderBase::Handle *ReaderBase::createHandle(std::size_t bufferSize) const
{
    boost::shared_array<char> buffer(new char[bufferSize]);
    return createHandle(buffer, bufferSize);
}

MmapReader::MmapHandle::MmapHandle(const MmapReader &owner, const std::string &filename)
    : ReaderBase::Handle(owner, 0)
{
    try
    {
        mapping.open(filename);
        vertexPtr = mapping.data() + owner.getHeaderSize();
    }
    catch (boost::exception &e)
    {
        e << boost::errinfo_file_name(filename);
        throw;
    }
}

const char *MmapReader::MmapHandle::readRaw(size_type first, size_type last, char *buffer) const
{
    (void) buffer; // will be NULL anyway
    (void) last;
    return vertexPtr + first * owner.getVertexSize();
}

MmapReader::MmapReader(const std::string &filename, float smooth = 1.0f)
    : ReaderBase(filename, smooth), filename(filename)
{
}

ReaderBase::Handle *MmapReader::createHandle() const
{
    return new MmapHandle(*this, filename);
}

ReaderBase::Handle *MmapReader::createHandle(std::size_t bufferSize) const
{
    (void) bufferSize; // no buffer is used
    return new MmapHandle(*this, filename);
}

ReaderBase::Handle *MmapReader::createHandle(boost::shared_array<char> buffer, std::size_t bufferSize) const
{
    (void) buffer; // no buffer is used
    (void) bufferSize;
    return new MmapHandle(*this, filename);
}


MemoryReader::MemoryHandle::MemoryHandle(const MemoryReader &owner, const char *data)
    : ReaderBase::Handle(owner, 0), vertexPtr(data + owner.getHeaderSize())
{
}

const char *MemoryReader::MemoryHandle::readRaw(size_type first, size_type last, char *buffer) const
{
    (void) buffer; // will be NULL anyway
    (void) last;
    return vertexPtr + first * owner.getVertexSize();
}

MemoryReader::MemoryReader(const char *data, std::size_t size, float smooth)
    : ReaderBase(smooth), data(data)
{
    boost::iostreams::array_source source(data, size);
    boost::iostreams::stream<boost::iostreams::array_source> in(source);
    readHeader(in);
    if ((size - getHeaderSize()) / getVertexSize() < this->size())
        throw boost::enable_error_info(FormatError("Input source is too small to contain its vertices"));
}

ReaderBase::Handle *MemoryReader::createHandle() const
{
    return new MemoryHandle(*this, data);
}

ReaderBase::Handle *MemoryReader::createHandle(std::size_t bufferSize) const
{
    (void) bufferSize; // no buffer is used
    return new MemoryHandle(*this, data);
}

ReaderBase::Handle *MemoryReader::createHandle(boost::shared_array<char> buffer, std::size_t bufferSize) const
{
    (void) buffer; // no buffer is used
    (void) bufferSize;
    return new MemoryHandle(*this, data);
}


SyscallReader::SyscallHandle::SyscallHandle(
    const SyscallReader &owner, boost::shared_array<char> buffer, std::size_t bufferSize)
: Handle(owner, buffer, bufferSize)
{
    fd = open(owner.filename.c_str(), O_RDONLY);
    if (fd < 0)
    {
        throw boost::enable_error_info(std::ios::failure("Could not open file"))
            << boost::errinfo_errno(errno)
            << boost::errinfo_file_name(owner.filename);
    }
}

const char * SyscallReader::SyscallHandle::readRaw(size_type first, size_type last, char *buffer) const
{
    std::size_t vertexSize = owner.getVertexSize();

    ssize_t remain = vertexSize * (last - first);
    char *ptr = buffer;
    lseek(fd, static_cast<const SyscallReader &>(owner).getHeaderSize() + first * vertexSize, SEEK_SET);

    while (remain > 0)
    {
        ssize_t bytes = ::read(fd, (void *) ptr, remain);
        if (bytes < 0)
        {
            if (errno == EAGAIN || errno == EINTR)
                continue;
            throw boost::enable_error_info(std::ios::failure("read failed"))
                << boost::errinfo_errno(errno);
        }
        else if (bytes == 0)
        {
            throw boost::enable_error_info(std::ios::failure("unexpected end of file"));
        }
        else
        {
            ptr += bytes;
            remain -= bytes;
        }
    }
    return buffer;
}

SyscallReader::SyscallHandle::~SyscallHandle()
{
    close(fd);
}

SyscallReader::SyscallReader(const std::string &filename, float smooth)
    : ReaderBase(filename, smooth), filename(filename)
{
}

ReaderBase::Handle *SyscallReader::createHandle(
    boost::shared_array<char> buffer, std::size_t bufferSize) const
{
    return new SyscallHandle(*this, buffer, bufferSize);
}


bool WriterBase::isOpen() const
{
    return isOpen_;
}

void WriterBase::addComment(const std::string &comment)
{
    MLSGPU_ASSERT(!isOpen(), state_error);
    comments.push_back(comment);
}

void WriterBase::setNumVertices(size_type numVertices)
{
    MLSGPU_ASSERT(!isOpen(), state_error);
    this->numVertices = numVertices;
}

void WriterBase::setNumTriangles(size_type numTriangles)
{
    MLSGPU_ASSERT(!isOpen(), state_error);
    this->numTriangles = numTriangles;
}

WriterBase::WriterBase() : comments(), numVertices(0), numTriangles(0), isOpen_(false) {}
WriterBase::~WriterBase() {}

WriterBase::size_type WriterBase::getNumVertices() const
{
    return numVertices;
}

WriterBase::size_type WriterBase::getNumTriangles() const
{
    return numTriangles;
}

void WriterBase::setOpen(bool open)
{
    isOpen_ = open;
}

std::string WriterBase::makeHeader()
{
    std::ostringstream out;
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


MmapWriter::MmapWriter() : vertexPtr(NULL), trianglePtr(NULL) {}

void MmapWriter::open(const std::string &filename)
{
    MLSGPU_ASSERT(!isOpen(), state_error);

    std::string header = makeHeader();

    boost::iostreams::mapped_file_params params(filename);
    params.mode = std::ios_base::out;
    // TODO: check for overflow here
    params.new_file_size = header.size() + getNumVertices() * vertexSize + getNumTriangles() * triangleSize;
    mapping.reset(new boost::iostreams::mapped_file_sink(params));

    std::memcpy(mapping->data(), header.data(), header.size());
    vertexPtr = mapping->data() + header.size();
    trianglePtr = vertexPtr + getNumVertices() * vertexSize;

    setOpen(true);
}

std::pair<char *, MmapWriter::size_type> MmapWriter::open()
{
    MLSGPU_ASSERT(!isOpen(), state_error);

    const std::string header = makeHeader();

    size_type size = header.size() + getNumVertices() * vertexSize + getNumTriangles() * triangleSize;
    char *data = new char[size];

    /* Code below here must not throw */
    std::memcpy(data, header.data(), header.size());
    vertexPtr = data + header.size();
    trianglePtr = vertexPtr + getNumVertices() * vertexSize;

    setOpen(true);
    return std::make_pair(data, size);
}

void MmapWriter::close()
{
    mapping.reset();
    setOpen(false);
}

void MmapWriter::writeVertices(size_type first, size_type count, const float *data)
{
    MLSGPU_ASSERT(isOpen(), state_error);
    MLSGPU_ASSERT(first + count <= getNumVertices() && first <= std::numeric_limits<size_type>::max() - count, std::out_of_range);
    memcpy(vertexPtr + first * vertexSize, data, count * vertexSize);
}

void MmapWriter::writeTriangles(size_type first, size_type count, const std::tr1::uint32_t *data)
{
    MLSGPU_ASSERT(isOpen(), state_error);
    MLSGPU_ASSERT(first + count <= getNumTriangles() && first <= std::numeric_limits<size_type>::max() - count, std::out_of_range);

    char *ptr = trianglePtr + first * triangleSize;
    for (size_type i = count; i > 0; i--, ptr += triangleSize, data += 3)
    {
        *ptr = 3;
        memcpy(ptr + 1, data, 3 * sizeof(data[0]));
    }
}

bool MmapWriter::supportsOutOfOrder() const
{
    return true;
}

void StreamWriter::openCommon(const std::string &header)
{
    if (!*file)
    {
        throw boost::enable_error_info(std::ios::failure("Could not open file"));
    }
    file->exceptions(std::ios::failbit | std::ios::badbit);
    *file << header;

    vertexOffset = header.size();
    triangleOffset = vertexOffset + getNumVertices() * vertexSize;
    setOpen(true);
}

void StreamWriter::open(const std::string &filename)
{
    MLSGPU_ASSERT(!isOpen(), state_error);

    try
    {
        std::string header = makeHeader();
        file.reset(new std::ofstream(filename.c_str(), std::ios::out | std::ios::binary));
        openCommon(header);
    }
    catch (boost::exception &e)
    {
        e << boost::errinfo_file_name(filename);
        throw;
    }
}

std::pair<char *, StreamWriter::size_type> StreamWriter::open()
{
    MLSGPU_ASSERT(!isOpen(), state_error);

    std::string header = makeHeader();
    size_type size = header.size() + getNumVertices() * vertexSize + getNumTriangles() * triangleSize;
    char *data = new char[size];

    try
    {
        file.reset(new boost::iostreams::stream<boost::iostreams::array_sink>(data, size));
        openCommon(header);
        return std::make_pair(data, size);
    }
    catch (std::exception &e)
    {
        delete[] data;
        throw;
    }
}

void StreamWriter::close()
{
    file.reset();
    setOpen(false);
}

void StreamWriter::writeVertices(size_type first, size_type count, const float *data)
{
    MLSGPU_ASSERT(isOpen(), state_error);
    MLSGPU_ASSERT(first + count <= getNumVertices() && first <= std::numeric_limits<size_type>::max() - count, std::out_of_range);

    file->seekp(boost::iostreams::offset_to_position(vertexOffset + first * vertexSize));
    /* Note: we're assuming that streamsize is big enough to hold anything we can hold in RAM */
    file->write(reinterpret_cast<const char *>(data), count * vertexSize);
}

void StreamWriter::writeTriangles(size_type first, size_type count, const std::tr1::uint32_t *data)
{
    MLSGPU_ASSERT(isOpen(), state_error);
    MLSGPU_ASSERT(first + count <= getNumTriangles() && first <= std::numeric_limits<size_type>::max() - count, std::out_of_range);

    file->seekp(boost::iostreams::offset_to_position(triangleOffset + first * triangleSize));
    while (count > 0)
    {
        const unsigned int bufferTriangles = 256;
        char buffer[bufferTriangles * triangleSize];
        char *ptr = buffer;
        unsigned int triangles = std::min(size_type(bufferTriangles), count);
        for (unsigned int i = 0; i < triangles; i++, ptr += triangleSize, data += 3)
        {
            ptr[0] = 3;
            memcpy(ptr + 1, data, 3 * sizeof(data[0]));
        }
        file->write(buffer, ptr - buffer);
        count -= triangles;
    }
}

bool StreamWriter::supportsOutOfOrder() const
{
    return true;
}


std::map<std::string, ReaderType> ReaderTypeWrapper::getNameMap()
{
    std::map<std::string, ReaderType> ans;
    ans["mmap"] = MMAP_READER;
    ans["syscall"] = SYSCALL_READER;
    return ans;
}

std::map<std::string, WriterType> WriterTypeWrapper::getNameMap()
{
    std::map<std::string, WriterType> ans;
    ans["mmap"] = MMAP_WRITER;
    ans["stream"] = STREAM_WRITER;
    return ans;
}

ReaderBase *createReader(ReaderType type, const std::string &filename, float smooth)
{
    switch (type)
    {
    case MMAP_READER: return new MmapReader(filename, smooth);
    case SYSCALL_READER: return new SyscallReader(filename, smooth);
    }
    return NULL; // should never be reached
}

WriterBase *createWriter(WriterType type)
{
    switch (type)
    {
    case MMAP_WRITER: return new MmapWriter();
    case STREAM_WRITER: return new StreamWriter();
    }
    return NULL; // should never be reached
}

} // namespace FastPly
