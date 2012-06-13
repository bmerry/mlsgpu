/**
 * @file
 *
 * Test code for @ref fast_ply.cpp.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif

#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/extensions/HelperMacros.h>
#include <string>
#include <cstddef>
#include <algorithm>
#include <vector>
#include <iterator>
#include <utility>
#include <boost/smart_ptr/scoped_array.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/bind.hpp>
#include <boost/ref.hpp>
#include "../src/fast_ply.h"
#include "../src/splat.h"
#include "testmain.h"

using namespace std;
using namespace FastPly;

class TestFastPlyReader : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestFastPlyReader);
    CPPUNIT_TEST_EXCEPTION(testEmpty, FormatError);
    CPPUNIT_TEST_EXCEPTION(testBadSignature, FormatError);
    CPPUNIT_TEST_EXCEPTION(testBadFormatFormat, FormatError);
    CPPUNIT_TEST_EXCEPTION(testBadFormatVersion, FormatError);
    CPPUNIT_TEST_EXCEPTION(testBadFormatLength, FormatError);
    CPPUNIT_TEST_EXCEPTION(testBadElementCount, FormatError);
    CPPUNIT_TEST_EXCEPTION(testBadElementOverflow, FormatError);
    CPPUNIT_TEST_EXCEPTION(testBadElementHex, FormatError);
    CPPUNIT_TEST_EXCEPTION(testBadElementLength, FormatError);
    CPPUNIT_TEST_EXCEPTION(testBadPropertyLength, FormatError);
    CPPUNIT_TEST_EXCEPTION(testBadPropertyListLength, FormatError);
    CPPUNIT_TEST_EXCEPTION(testBadPropertyListType, FormatError);
    CPPUNIT_TEST_EXCEPTION(testBadPropertyType, FormatError);
    CPPUNIT_TEST_EXCEPTION(testBadHeaderToken, FormatError);
    CPPUNIT_TEST_EXCEPTION(testEarlyProperty, FormatError);
    CPPUNIT_TEST_EXCEPTION(testDuplicateProperty, FormatError);
    CPPUNIT_TEST_EXCEPTION(testMissingEnd, FormatError);
    CPPUNIT_TEST_EXCEPTION(testShortFile, FormatError);
    CPPUNIT_TEST_EXCEPTION(testList, FormatError);
    CPPUNIT_TEST_EXCEPTION(testNotFloat, FormatError);
    CPPUNIT_TEST_EXCEPTION(testFormatAscii, FormatError);
    CPPUNIT_TEST_EXCEPTION(testFileNotFound, std::ios_base::failure);

    CPPUNIT_TEST(testReadHeader);
    CPPUNIT_TEST(testRead);
    CPPUNIT_TEST(testReadIterator);
    CPPUNIT_TEST_SUITE_END();

private:
    string content;                    ///< Convenience data store for the file content

    /// Populates content with some useful data for a read test
    void setupRead(int numVertices);

public:
    /**
     * @name Negative tests
     * @{
     * These tests are all expected to throw an exception,
     */

    void testEmpty();                  ///< Empty file
    void testBadSignature();           ///< Signature is not PLY
    void testBadFormatFormat();        ///< Format is not @c ascii etc.
    void testBadFormatVersion();       ///< Version is not 1.0
    void testBadFormatLength();        ///< Wrong number of tokens on format line
    void testBadElementCount();        ///< Negative number of elements
    void testBadElementOverflow();     ///< Element count too large to hold
    void testBadElementHex();          ///< Element count encoded in hex
    void testBadElementLength();       ///< Wrong number of tokens on element line
    void testBadPropertyLength();      ///< Wrong number of tokens on simple property line
    void testBadPropertyListLength();  ///< Wrong number of tokens on list property line
    void testBadPropertyListType();    ///< List length type is not valid
    void testBadPropertyType();        ///< Property type is not valid
    void testBadHeaderToken();         ///< Unrecognized header line
    void testEarlyProperty();          ///< Property line before any element line
    void testDuplicateProperty();      ///< Two properties with the same name
    void testMissingEnd();             ///< Header ends without @c end_header
    void testShortFile();              ///< File too small to hold all the vertex data
    void testList();                   ///< Vertex element contains a list
    void testNotFloat();               ///< Vertex property is not a float
    void testFormatAscii();            ///< Ascii format file
    void testFileNotFound();           ///< PLY file does not exist
    /** @} */

    /**
     * @name Positive tests
     * @{
     */
    void testReadHeader();             ///< Checks that header-related fields are set properly
    void testRead();                   ///< Tests @ref FastPly::Reader::read with a pointer
    void testReadIterator();           ///< Tests @ref FastPly::Reader::read with an output iterator
    /** @} */

    /**
     * Sets @ref content to @a header plus @a payloadBytes bytes of arbitrary data.
     * @a payloadBytes defaults to a medium-sized value so that the negative tests can be
     * sure that a format error is not due to a short file.
     */
    void setContent(const string &header, size_t payloadBytes = 256);
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestFastPlyReader, TestSet::perBuild());

void TestFastPlyReader::setContent(const string &header, size_t payloadBytes)
{
    content = header + string(payloadBytes, '\0');
}

void TestFastPlyReader::testEmpty()
{
    setContent("");
    MemoryReader(content.data(), content.size(), 1.0f);
}

void TestFastPlyReader::testBadSignature()
{
    setContent("ply no not really");
    MemoryReader(content.data(), content.size(), 1.0f);
}

void TestFastPlyReader::testBadFormatFormat()
{
    setContent("ply\nformat binary_little_endiannotreally 1.0\nelement vertex 1\nend_header\n");
    MemoryReader(content.data(), content.size(), 1.0f);
}

void TestFastPlyReader::testBadFormatVersion()
{
    setContent("ply\nformat binary_little_endian 1.01\nelement vertex 1\nend_header\n");
    MemoryReader(content.data(), content.size(), 1.0f);
}

void TestFastPlyReader::testBadFormatLength()
{
    setContent("ply\nformat\nelement vertex 1\nend_header\n");
    MemoryReader(content.data(), content.size(), 1.0f);
}

void TestFastPlyReader::testBadElementCount()
{
    setContent("ply\nformat binary_little_endian 1.0\nelement vertex -1\nend_header\n");
    MemoryReader(content.data(), content.size(), 1.0f);
}

void TestFastPlyReader::testBadElementOverflow()
{
    setContent("ply\nformat binary_little_endian 1.0\nelement vertex 123456789012345678901234567890\nend_header\n");
    MemoryReader(content.data(), content.size(), 1.0f);
}

void TestFastPlyReader::testBadElementHex()
{
    setContent("ply\nformat binary_little_endian 1.0\nelement vertex 0xDEADBEEF\nend_header\n");
    MemoryReader(content.data(), content.size(), 1.0f);
}

void TestFastPlyReader::testBadElementLength()
{
    setContent("ply\nformat binary_little_endian 1.0\nelement\nend_header\n");
    MemoryReader(content.data(), content.size(), 1.0f);
}

void TestFastPlyReader::testBadPropertyLength()
{
    setContent("ply\nformat binary_little_endian 1.0\nelement vertex 0\nproperty int int int x\nend_header\n");
    MemoryReader(content.data(), content.size(), 1.0f);
}

void TestFastPlyReader::testBadPropertyListLength()
{
    setContent("ply\nformat binary_little_endian 1.0\nelement vertex 0\nproperty list int x\nend_header\n");
    MemoryReader(content.data(), content.size(), 1.0f);
}

void TestFastPlyReader::testBadPropertyListType()
{
    setContent("ply\nformat binary_little_endian 1.0\nelement vertex 0\nproperty list float int x\nend_header\n");
    MemoryReader(content.data(), content.size(), 1.0f);
}

void TestFastPlyReader::testBadPropertyType()
{
    setContent("ply\nformat binary_little_endian 1.0\nelement vertex 0\nproperty int1 x\nend_header\n");
    MemoryReader(content.data(), content.size(), 1.0f);
}

void TestFastPlyReader::testBadHeaderToken()
{
    setContent("ply\nformat binary_little_endian 1.0\nelement vertex 0\nfoo\nend_header\n");
    MemoryReader(content.data(), content.size(), 1.0f);
}

void TestFastPlyReader::testEarlyProperty()
{
    setContent("ply\nformat binary_little_endian 1.0\nproperty int x\nelement vertex 0\nend_header\n");
    MemoryReader(content.data(), content.size(), 1.0f);
}

void TestFastPlyReader::testDuplicateProperty()
{
    setContent("ply\nformat binary_little_endian 1.0\nelement vertex 0\nproperty int x\nproperty float x\nend_header\n");
    MemoryReader(content.data(), content.size(), 1.0f);
}

void TestFastPlyReader::testMissingEnd()
{
    setContent("ply\nformat binary_little_endian 1.0\nelement vertex 0\nproperty int x\n");
    MemoryReader(content.data(), content.size(), 1.0f);
}

void TestFastPlyReader::testShortFile()
{
    setContent(
        "ply\n"
        "format binary_little_endian 1.0\n"
        "element vertex 5\n"
        "property float32 x\n"
        "property float32 y\n"
        "property float32 z\n"
        "property float32 nx\n"
        "property float32 ny\n"
        "property float32 nz\n"
        "property float32 radius\n"
        "property uint8 foo\n"
        "end_header\n", 29 * 5 - 1);
    MemoryReader(content.data(), content.size(), 1.0f);
}

void TestFastPlyReader::testList()
{
    setContent(
        "ply\n"
        "format binary_little_endian 1.0\n"
        "element vertex 5\n"
        "property float32 x\n"
        "property float32 y\n"
        "property list uchar float32 z\n"
        "property float32 nx\n"
        "property float32 ny\n"
        "property float32 nz\n"
        "property float32 radius\n"
        "property uint8 foo\n"
        "end_header\n");
    MemoryReader(content.data(), content.size(), 1.0f);
}

void TestFastPlyReader::testNotFloat()
{
    setContent(
        "ply\n"
        "format binary_little_endian 1.0\n"
        "element vertex 5\n"
        "property float32 x\n"
        "property float32 y\n"
        "property float32 z\n"
        "property uint32 nx\n"
        "property float32 ny\n"
        "property float32 nz\n"
        "property float32 radius\n"
        "property uint8 foo\n"
        "end_header\n");
    MemoryReader(content.data(), content.size(), 1.0f);
}

void TestFastPlyReader::testFormatAscii()
{
    setContent(
        "ply\n"
        "format ascii 1.0\n"
        "element vertex 5\n"
        "property float32 x\n"
        "property float32 y\n"
        "property float32 z\n"
        "property float32 nx\n"
        "property float32 ny\n"
        "property float32 nz\n"
        "property float32 radius\n"
        "property uint8 foo\n"
        "end_header\n");
    MemoryReader(content.data(), content.size(), 1.0f);
}

void TestFastPlyReader::testFileNotFound()
{
    MmapReader r("not_a_real_file.in", 1.0f);
}

void TestFastPlyReader::testReadHeader()
{
    const string header = 
        "ply\n"
        "format binary_little_endian 1.0\n"
        "element vertex 5\n"
        "property float32 z\n"
        "property float32 y\n"
        "property float32 x\n"
        "property int16 bar\n"
        "property float32 nx\n"
        "property float32 ny\n"
        "property float32 nz\n"
        "property float32 radius\n"
        "property uint8 foo\n"
        "end_header\n";
    setContent(header);
    MemoryReader r(content.data(), content.size(), 1.0f);
    CPPUNIT_ASSERT_EQUAL(31, int(r.vertexSize));
    CPPUNIT_ASSERT_EQUAL(5, int(r.vertexCount));

    CPPUNIT_ASSERT_EQUAL(8, int(r.offsets[ReaderBase::X]));
    CPPUNIT_ASSERT_EQUAL(4, int(r.offsets[ReaderBase::Y]));
    CPPUNIT_ASSERT_EQUAL(0, int(r.offsets[ReaderBase::Z]));
    CPPUNIT_ASSERT_EQUAL(14, int(r.offsets[ReaderBase::NX]));
    CPPUNIT_ASSERT_EQUAL(18, int(r.offsets[ReaderBase::NY]));
    CPPUNIT_ASSERT_EQUAL(22, int(r.offsets[ReaderBase::NZ]));
    CPPUNIT_ASSERT_EQUAL(26, int(r.offsets[ReaderBase::RADIUS]));

    CPPUNIT_ASSERT_EQUAL(int(header.size()), int(r.getHeaderSize()));
}

void TestFastPlyReader::setupRead(int numVertices)
{
    // TODO: memory leak on exception
    boost::scoped_array<float> vertices(new float[numVertices * 7]);
    for (int i = 0; i < numVertices; i++)
        for (int j = 0; j < 7; j++)
            vertices[i * 7 + j] = i * 100.0f + j;
    string header =
        "ply\n"
        "format binary_little_endian 1.0\n"
        "element vertex ";
    header += boost::lexical_cast<string>(numVertices);
    header += "\n"
        "property float32 y\n"
        "property float32 z\n"
        "property float32 x\n"
        "property float32 nx\n"
        "property float32 ny\n"
        "property float32 nz\n"
        "property float32 radius\n"
        "end_header\n";
    setContent(header, numVertices * 7 * sizeof(float));
    copy((const char *) vertices.get(),
         (const char *) (vertices.get() + numVertices * 7),
         content.begin() + header.size());
}

void TestFastPlyReader::testRead()
{
    setupRead(5);

    MemoryReader r(content.data(), content.size(), 2.0f);
    boost::scoped_ptr<ReaderBase::Handle> h(r.createHandle());

    Splat out[4] = {};
    h->read(1, 4, out);
    CPPUNIT_ASSERT_EQUAL(0.0f, out[3].position[0]); // check for overwriting
    for (int i = 0; i < 3; i++)
    {
        CPPUNIT_ASSERT_EQUAL(i * 100.0f + 102.0f, out[i].position[0]);
        CPPUNIT_ASSERT_EQUAL(i * 100.0f + 100.0f, out[i].position[1]);
        CPPUNIT_ASSERT_EQUAL(i * 100.0f + 101.0f, out[i].position[2]);
        CPPUNIT_ASSERT_EQUAL(i * 100.0f + 103.0f, out[i].normal[0]);
        CPPUNIT_ASSERT_EQUAL(i * 100.0f + 104.0f, out[i].normal[1]);
        CPPUNIT_ASSERT_EQUAL(i * 100.0f + 105.0f, out[i].normal[2]);
        CPPUNIT_ASSERT_EQUAL(2.0f * (i * 100.0f + 106.0f), out[i].radius);
    }

    CPPUNIT_ASSERT_THROW(h->read(1, 6, out), std::out_of_range);
}

void TestFastPlyReader::testReadIterator()
{
    setupRead(10000);

    MemoryReader r(content.data(), content.size(), 2.0f);
    boost::scoped_ptr<ReaderBase::Handle> h(r.createHandle());

    vector<Splat> out;
    h->read(2, 9500, back_inserter(out));
    CPPUNIT_ASSERT_EQUAL(9500 - 2, int(out.size()));
    for (int i = 2; i < 9500; i++)
    {
        CPPUNIT_ASSERT_EQUAL(i * 100.0f + 2.0f, out[i - 2].position[0]);
        CPPUNIT_ASSERT_EQUAL(i * 100.0f + 0.0f, out[i - 2].position[1]);
        CPPUNIT_ASSERT_EQUAL(i * 100.0f + 1.0f, out[i - 2].position[2]);
        CPPUNIT_ASSERT_EQUAL(i * 100.0f + 3.0f, out[i - 2].normal[0]);
        CPPUNIT_ASSERT_EQUAL(i * 100.0f + 4.0f, out[i - 2].normal[1]);
        CPPUNIT_ASSERT_EQUAL(i * 100.0f + 5.0f, out[i - 2].normal[2]);
        CPPUNIT_ASSERT_EQUAL(2.0f * (i * 100.0f + 6.0f), out[i - 2].radius);
    }

    CPPUNIT_ASSERT_THROW(h->read(1, 10001, back_inserter(out)), std::out_of_range);
}

/**
 * Abstract base class for testing the writers in the FastPly namespace.
 */
template<typename Writer>
class TestFastPlyWriterBase : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestFastPlyWriterBase<Writer>);
    CPPUNIT_TEST_EXCEPTION(testBadFilename, std::ios_base::failure);
    CPPUNIT_TEST(testSimple);
    CPPUNIT_TEST(testState);
    CPPUNIT_TEST(testOverrun);
    CPPUNIT_TEST_SUITE_END_ABSTRACT();
public:
    void testBadFilename();   ///< Try to write to an invalid filename, check for error
    void testSimple();        ///< Test normal operation
    void testState();         ///< Test assertions that the file is/is not open
    void testOverrun();       ///< Test writing beyond the end of the file
};

template<typename Writer>
void TestFastPlyWriterBase<Writer>::testBadFilename()
{
    Writer w;
    w.open("/not_a_valid_filename/");
}

template<typename Writer>
void TestFastPlyWriterBase<Writer>::testSimple()
{
    const float vertices[4 * 3] =
    {
        1.0f, 2.0f, 4.0f,
        -1.0f, -2.0f, -4.0f,
        5.5f, 6.25f, 7.75f,
        8.0f, 9.0f, 10.5f
    };
    /* Note: not all these indices index the available vertices.
     * That doesn't get checked by Writer, and it allows us to check
     * that the full range gets written correctly.
     */
    const std::tr1::uint32_t indices[9] =
    {
        0, 1, 2,
        100, 2000, 300000,
        4294967295U, 4294967294U, 4294967293U
    };
    /* Note: I've hard-coded for little endian. The test will fail on a big-endian
     * machine.
     */
    const std::string expectedHeader =
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment my comment 1\n"
        "comment my comment 23\n"
        "element vertex 4\n"
        "property float32 x\n"
        "property float32 y\n"
        "property float32 z\n"
        "element face 3\n"
        "property list uint8 uint32 vertex_indices\n"
        "comment padding:XX\n"
        "end_header\n";
    const typename Writer::size_type headerSize = expectedHeader.size();

    Writer w;
    w.addComment("my comment 1");
    w.addComment("my comment 23");
    w.setNumVertices(4);
    w.setNumTriangles(3);

    pair<char *, typename Writer::size_type> range = w.open();
    boost::scoped_array<char> data(range.first);
    typename Writer::size_type size = range.second;
    CPPUNIT_ASSERT(data.get() != NULL);
    CPPUNIT_ASSERT_EQUAL(headerSize + 87, size);

    if (w.supportsOutOfOrder())
    {
        w.writeVertices(1, 2, vertices + 1 * 3);
        w.writeTriangles(1, 2, indices + 1 * 3);
        w.writeVertices(0, 1, vertices);
        w.writeTriangles(0, 1, indices);
        w.writeVertices(3, 1, vertices + 3 * 3);
    }
    else
    {
        w.writeVertices(0, 1, vertices);
        w.writeVertices(1, 2, vertices + 1 * 3);
        w.writeVertices(3, 1, vertices + 3 * 3);
        w.writeTriangles(0, 1, indices);
        w.writeTriangles(1, 2, indices + 1 * 3);
    }
    w.close();

    CPPUNIT_ASSERT_EQUAL(expectedHeader, std::string(data.get(), headerSize));
    CPPUNIT_ASSERT(0 == memcmp(data.get() + headerSize, vertices, sizeof(vertices)));
    CPPUNIT_ASSERT_EQUAL(3, int(data[headerSize + 48]));
    CPPUNIT_ASSERT(0 == memcmp(data.get() + headerSize + 49, indices + 0, 12));
    CPPUNIT_ASSERT_EQUAL(3, int(data[headerSize + 61]));
    CPPUNIT_ASSERT(0 == memcmp(data.get() + headerSize + 62, indices + 3, 12));
    CPPUNIT_ASSERT_EQUAL(3, int(data[headerSize + 74]));
    CPPUNIT_ASSERT(0 == memcmp(data.get() + headerSize + 75, indices + 6, 12));
}

template<typename Writer>
void TestFastPlyWriterBase<Writer>::testState()
{
    Writer w;

    w.setNumVertices(2);
    w.setNumTriangles(2);

    CPPUNIT_ASSERT_THROW(w.writeVertices(0, 1, NULL), state_error);
    CPPUNIT_ASSERT_THROW(w.writeTriangles(0, 1, NULL), state_error);

    pair<char *, typename Writer::size_type> range = w.open();
    boost::scoped_array<char> data(range.first);

    CPPUNIT_ASSERT_THROW(w.open(), state_error);
    CPPUNIT_ASSERT_THROW(w.setNumVertices(3), state_error);
    CPPUNIT_ASSERT_THROW(w.setNumTriangles(3), state_error);
}

template<typename Writer>
void TestFastPlyWriterBase<Writer>::testOverrun()
{
    const float vertices[4 * 3] = {}; // content does not matter
    const std::tr1::uint32_t indices[9] = {};

    Writer w;
    w.setNumVertices(4);
    w.setNumTriangles(3);

    pair<char *, typename Writer::size_type> range = w.open();
    boost::scoped_array<char> data(range.first);

    /* Just to check that normal writes work, and to
     * to position a streaming writer.
     */
    w.writeVertices(0, 1, vertices);

    /* The real test */
    CPPUNIT_ASSERT_THROW(w.writeVertices(1, 4, vertices), std::out_of_range);
    CPPUNIT_ASSERT_THROW(w.writeVertices(2, MmapWriter::size_type(-1), vertices), std::out_of_range);
    CPPUNIT_ASSERT_THROW(w.writeVertices(MmapWriter::size_type(-1), 2, vertices), std::out_of_range);
    w.writeVertices(1, 3, vertices);

    w.writeTriangles(0, 2, indices);
    CPPUNIT_ASSERT_THROW(w.writeTriangles(1, 3, indices), std::out_of_range);
    CPPUNIT_ASSERT_THROW(w.writeTriangles(2, MmapWriter::size_type(-1), indices), std::out_of_range);
    CPPUNIT_ASSERT_THROW(w.writeTriangles(MmapWriter::size_type(-1), 2, indices), std::out_of_range);
}

class TestMmapWriter : public TestFastPlyWriterBase<MmapWriter>
{
    CPPUNIT_TEST_SUB_SUITE(TestMmapWriter, TestFastPlyWriterBase<MmapWriter>);
    CPPUNIT_TEST_SUITE_END();
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestMmapWriter, TestSet::perBuild());

class TestStreamWriter : public TestFastPlyWriterBase<StreamWriter>
{
    CPPUNIT_TEST_SUB_SUITE(TestStreamWriter, TestFastPlyWriterBase<StreamWriter>);
    CPPUNIT_TEST_SUITE_END();
public:
    void testSequence();      ///< Test writing things out of order
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestStreamWriter, TestSet::perBuild());
