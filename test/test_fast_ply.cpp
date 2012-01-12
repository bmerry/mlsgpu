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

    CPPUNIT_TEST(testReadHeader);
    CPPUNIT_TEST(testReadVertices);
    CPPUNIT_TEST_SUITE_END();

private:
    string content;

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
