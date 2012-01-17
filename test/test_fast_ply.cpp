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
#include <boost/smart_ptr/scoped_array.hpp>
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
    CPPUNIT_TEST(testReadVertices);
    CPPUNIT_TEST_SUITE_END();

private:
    string content;                    ///< Convenience data store for the file content

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
    void testReadVertices();           ///< Tests @ref FastPly::Reader::readVertices
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
    Reader(content.data(), content.size());
}

void TestFastPlyReader::testBadSignature()
{
    setContent("ply no not really");
    Reader(content.data(), content.size());
}

void TestFastPlyReader::testBadFormatFormat()
{
    setContent("ply\nformat binary_little_endiannotreally 1.0\nelement vertex 1\nend_header\n");
    Reader(content.data(), content.size());
}

void TestFastPlyReader::testBadFormatVersion()
{
    setContent("ply\nformat binary_little_endian 1.01\nelement vertex 1\nend_header\n");
    Reader(content.data(), content.size());
}

void TestFastPlyReader::testBadFormatLength()
{
    setContent("ply\nformat\nelement vertex 1\nend_header\n");
    Reader(content.data(), content.size());
}

void TestFastPlyReader::testBadElementCount()
{
    setContent("ply\nformat binary_little_endian 1.0\nelement vertex -1\nend_header\n");
    Reader(content.data(), content.size());
}

void TestFastPlyReader::testBadElementOverflow()
{
    setContent("ply\nformat binary_little_endian 1.0\nelement vertex 123456789012345678901234567890\nend_header\n");
    Reader(content.data(), content.size());
}

void TestFastPlyReader::testBadElementHex()
{
    setContent("ply\nformat binary_little_endian 1.0\nelement vertex 0xDEADBEEF\nend_header\n");
    Reader(content.data(), content.size());
}

void TestFastPlyReader::testBadElementLength()
{
    setContent("ply\nformat binary_little_endian 1.0\nelement\nend_header\n");
    Reader(content.data(), content.size());
}

void TestFastPlyReader::testBadPropertyLength()
{
    setContent("ply\nformat binary_little_endian 1.0\nelement vertex 0\nproperty int int int x\nend_header\n");
    Reader(content.data(), content.size());
}

void TestFastPlyReader::testBadPropertyListLength()
{
    setContent("ply\nformat binary_little_endian 1.0\nelement vertex 0\nproperty list int x\nend_header\n");
    Reader(content.data(), content.size());
}

void TestFastPlyReader::testBadPropertyListType()
{
    setContent("ply\nformat binary_little_endian 1.0\nelement vertex 0\nproperty list float int x\nend_header\n");
    Reader(content.data(), content.size());
}

void TestFastPlyReader::testBadPropertyType()
{
    setContent("ply\nformat binary_little_endian 1.0\nelement vertex 0\nproperty int1 x\nend_header\n");
    Reader(content.data(), content.size());
}

void TestFastPlyReader::testBadHeaderToken()
{
    setContent("ply\nformat binary_little_endian 1.0\nelement vertex 0\nfoo\nend_header\n");
    Reader(content.data(), content.size());
}

void TestFastPlyReader::testEarlyProperty()
{
    setContent("ply\nformat binary_little_endian 1.0\nproperty int x\nelement vertex 0\nend_header\n");
    Reader(content.data(), content.size());
}

void TestFastPlyReader::testDuplicateProperty()
{
    setContent("ply\nformat binary_little_endian 1.0\nelement vertex 0\nproperty int x\nproperty float x\nend_header\n");
    Reader(content.data(), content.size());
}

void TestFastPlyReader::testMissingEnd()
{
    setContent("ply\nformat binary_little_endian 1.0\nelement vertex 0\nproperty int x\n");
    Reader(content.data(), content.size());
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
    Reader(content.data(), content.size());
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
    Reader(content.data(), content.size());
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
    Reader(content.data(), content.size());
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
    Reader(content.data(), content.size());
}

void TestFastPlyReader::testFileNotFound()
{
    Reader r("not_a_real_file.in");
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
    Reader r(content.data(), content.size());
    CPPUNIT_ASSERT_EQUAL(31, int(r.vertexSize));
    CPPUNIT_ASSERT_EQUAL(5, int(r.vertexCount));

    CPPUNIT_ASSERT_EQUAL(8, int(r.offsets[Reader::X]));
    CPPUNIT_ASSERT_EQUAL(4, int(r.offsets[Reader::Y]));
    CPPUNIT_ASSERT_EQUAL(0, int(r.offsets[Reader::Z]));
    CPPUNIT_ASSERT_EQUAL(14, int(r.offsets[Reader::NX]));
    CPPUNIT_ASSERT_EQUAL(18, int(r.offsets[Reader::NY]));
    CPPUNIT_ASSERT_EQUAL(22, int(r.offsets[Reader::NZ]));
    CPPUNIT_ASSERT_EQUAL(26, int(r.offsets[Reader::RADIUS]));

    CPPUNIT_ASSERT_EQUAL(int(header.size()), int(r.vertexPtr - r.filePtr));
}

void TestFastPlyReader::testReadVertices()
{
    float vertices[5][7];
    for (int i = 0; i < 5; i++)
        for (int j = 0; j < 7; j++)
            vertices[i][j] = i * 100.0f + j;
    const string header =
        "ply\n"
        "format binary_little_endian 1.0\n"
        "element vertex 5\n"
        "property float32 y\n"
        "property float32 z\n"
        "property float32 x\n"
        "property float32 nx\n"
        "property float32 ny\n"
        "property float32 nz\n"
        "property float32 radius\n"
        "end_header\n";
    setContent(header, sizeof(vertices));
    copy((const char *) vertices, (const char *) vertices + sizeof(vertices), content.begin() + header.size());

    Reader r(content.data(), content.size());
    Splat out[4] = {};
    r.readVertices(1, 3, out);
    CPPUNIT_ASSERT_EQUAL(0.0f, out[3].position[0]); // check for overwriting
    for (int i = 0; i < 3; i++)
    {
        CPPUNIT_ASSERT_EQUAL(i * 100.0f + 102.0f, out[i].position[0]);
        CPPUNIT_ASSERT_EQUAL(i * 100.0f + 100.0f, out[i].position[1]);
        CPPUNIT_ASSERT_EQUAL(i * 100.0f + 101.0f, out[i].position[2]);
        CPPUNIT_ASSERT_EQUAL(i * 100.0f + 103.0f, out[i].normal[0]);
        CPPUNIT_ASSERT_EQUAL(i * 100.0f + 104.0f, out[i].normal[1]);
        CPPUNIT_ASSERT_EQUAL(i * 100.0f + 105.0f, out[i].normal[2]);
        CPPUNIT_ASSERT_EQUAL(i * 100.0f + 106.0f, out[i].radius);
    }

    CPPUNIT_ASSERT_THROW(r.readVertices(1, 5, out), std::out_of_range);
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
        "end_header   \n";
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

    if (Writer::outOfOrder)
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

    CPPUNIT_ASSERT_THROW(w.writeVertices(0, 1, NULL), std::runtime_error);
    CPPUNIT_ASSERT_THROW(w.writeTriangles(0, 1, NULL), std::runtime_error);

    pair<char *, typename Writer::size_type> range = w.open();
    boost::scoped_array<char> data(range.first);

    CPPUNIT_ASSERT_THROW(w.open(), std::runtime_error);
    CPPUNIT_ASSERT_THROW(w.setNumVertices(3), std::runtime_error);
    CPPUNIT_ASSERT_THROW(w.setNumTriangles(3), std::runtime_error);
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
    CPPUNIT_TEST(testSequence);
    CPPUNIT_TEST_SUITE_END();
public:
    void testSequence();      ///< Test writing things out of order
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestStreamWriter, TestSet::perBuild());

void TestStreamWriter::testSequence()
{
    const float vertices[4 * 3] = {}; // content does not matter
    const std::tr1::uint32_t indices[9] = {};

    StreamWriter w;
    w.setNumVertices(4);
    w.setNumTriangles(3);

    pair<char *, StreamWriter::size_type> range = w.open();
    boost::scoped_array<char> data(range.first);

    // Triangles before vertices
    CPPUNIT_ASSERT_THROW(w.writeTriangles(0, 1, indices), std::runtime_error);
    // Starting from non-zero vertex
    CPPUNIT_ASSERT_THROW(w.writeVertices(1, 2, vertices), std::runtime_error);

    w.writeVertices(0, 2, vertices);

    // Triangles before finishing vertices
    CPPUNIT_ASSERT_THROW(w.writeTriangles(0, 1, indices), std::runtime_error);
    // Overwrite already-written
    CPPUNIT_ASSERT_THROW(w.writeVertices(1, 2, vertices), std::runtime_error);
    // Skip a vertex
    CPPUNIT_ASSERT_THROW(w.writeVertices(3, 1, vertices), std::runtime_error);

    w.writeVertices(2, 2, vertices);

    // Starting from non-zero triangle
    CPPUNIT_ASSERT_THROW(w.writeTriangles(1, 1, indices), std::runtime_error);

    w.writeTriangles(0, 1, indices);

    // Overwrite
    CPPUNIT_ASSERT_THROW(w.writeTriangles(0, 1, indices), std::runtime_error);
    // Going back to vertices
    CPPUNIT_ASSERT_THROW(w.writeVertices(3, 1, vertices), std::runtime_error);
    // Skipping
    CPPUNIT_ASSERT_THROW(w.writeTriangles(2, 1, indices), std::runtime_error);

    w.writeTriangles(1, 2, indices);
}
