/**
 * @file
 *
 * Test code for @ref fast_ply.cpp.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif

#ifndef BOOST_FILESYSTEM_VERSION
# define BOOST_FILESYSTEM_VERSION 3
#endif

#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/extensions/ExceptionTestCaseDecorator.h>
#include <string>
#include <cstddef>
#include <algorithm>
#include <vector>
#include <iterator>
#include <utility>
#include <fstream>
#include <sstream>
#include <limits>
#include <boost/smart_ptr/scoped_array.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/bind.hpp>
#include <boost/ref.hpp>
#include <boost/exception/all.hpp>
#include <boost/filesystem.hpp>
#include "../src/fast_ply.h"
#include "../src/splat.h"
#include "memory_reader.h"
#include "memory_writer.h"
#include "testutil.h"

using namespace FastPly;

static const std::string testFilename = "test_fast_ply.ply";

/**
 * Tests for @ref FastPly::Reader.
 */
class TestFastPlyReader : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestFastPlyReader);
    TEST_EXCEPTION_FILENAME(testEmpty, FormatError, testFilename);
    TEST_EXCEPTION_FILENAME(testBadSignature, FormatError, testFilename);
    TEST_EXCEPTION_FILENAME(testBadFormatFormat, FormatError, testFilename);
    TEST_EXCEPTION_FILENAME(testBadFormatVersion, FormatError, testFilename);
    TEST_EXCEPTION_FILENAME(testBadFormatLength, FormatError, testFilename);
    TEST_EXCEPTION_FILENAME(testBadElementCount, FormatError, testFilename);
    TEST_EXCEPTION_FILENAME(testBadElementOverflow, FormatError, testFilename);
    TEST_EXCEPTION_FILENAME(testBadElementHex, FormatError, testFilename);
    TEST_EXCEPTION_FILENAME(testBadElementLength, FormatError, testFilename);
    TEST_EXCEPTION_FILENAME(testBadPropertyLength, FormatError, testFilename);
    TEST_EXCEPTION_FILENAME(testBadPropertyListLength, FormatError, testFilename);
    TEST_EXCEPTION_FILENAME(testBadPropertyListType, FormatError, testFilename);
    TEST_EXCEPTION_FILENAME(testBadPropertyType, FormatError, testFilename);
    TEST_EXCEPTION_FILENAME(testBadPropertyLine, FormatError, testFilename);
    TEST_EXCEPTION_FILENAME(testBadHeaderToken, FormatError, testFilename);
    TEST_EXCEPTION_FILENAME(testEarlyProperty, FormatError, testFilename);
    TEST_EXCEPTION_FILENAME(testDuplicateProperty, FormatError, testFilename);
    TEST_EXCEPTION_FILENAME(testMissingEnd, FormatError, testFilename);
    TEST_EXCEPTION_FILENAME(testShortFile, boost::exception, testFilename);
    TEST_EXCEPTION_FILENAME(testList, FormatError, testFilename);
    TEST_EXCEPTION_FILENAME(testNotFloat, FormatError, testFilename);
    TEST_EXCEPTION_FILENAME(testFormatAscii, FormatError, testFilename);
    TEST_EXCEPTION_FILENAME(testFormatMissing, FormatError, testFilename);

    CPPUNIT_TEST(testReadHeader);
    CPPUNIT_TEST(testRead);
    CPPUNIT_TEST(testReadZero);
    CPPUNIT_TEST(testReadIterator);
    CPPUNIT_TEST_SUITE_END();

private:
    std::string content;                    ///< Convenience data store for the file content

protected:
    /// Populates content with some useful data for a read test
    void setupRead(int numVertices);

    /**
     * Check that data read from the output of @ref setupRead is correct.
     *
     * @param offset          Index of first vertex in the given range
     * @param first,last      Contiguous range of splats read from file
     */
    template<typename ForwardIterator>
    void verify(int offset, ForwardIterator first, ForwardIterator last);

    /**
     * Create an instance of a reader. The reader uses @ref MemoryReader as the
     * backend.
     *
     * @param fileContent     Data to place in the file
     * @param filename        Filename to be used for the file
     * @param smooth          Smoothing factor to pass to the constructor
     * @param maxRadius       Max radius to pass to the constructor
     *
     * @return A reader that will read @a fileContent.
     * @throw boost::exception if the header is invalid.
     */
    Reader *factory(const std::string &fileContent,
                    const std::string &filename = testFilename,
                    float smooth = 1.0f,
                    float maxRadius = std::numeric_limits<float>::infinity()) const;

public:
    /**
     * @name Negative tests
     * @{
     * These tests are all expected to throw an exception.
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
    void testBadPropertyLine();        ///< Property line has insufficient tokens
    void testBadHeaderToken();         ///< Unrecognized header line
    void testEarlyProperty();          ///< Property line before any element line
    void testDuplicateProperty();      ///< Two properties with the same name
    void testMissingEnd();             ///< Header ends without @c end_header
    void testShortFile();              ///< File too small to hold all the vertex data
    void testList();                   ///< Vertex element contains a list
    void testNotFloat();               ///< Vertex property is not a float
    void testFormatAscii();            ///< Ascii format file
    void testFormatMissing();          ///< No format line
    /** @} */

    /**
     * @name Positive tests
     * @{
     */
    void testReadHeader();             ///< Checks that header-related fields are set properly
    void testRead();                   ///< Tests @ref FastPly::Reader::Handle::read with a pointer
    void testReadZero();               ///< Tests a zero-splat read
    void testReadIterator();           ///< Tests @ref FastPly::Reader::Handle::read with an output iterator
    /** @} */

    /**
     * Sets @ref content to @a header plus @a payloadBytes bytes of arbitrary data.
     * @a payloadBytes defaults to a medium-sized value so that the negative tests can be
     * sure that a format error is not due to a short file.
     */
    void setContent(const std::string &header, size_t payloadBytes = 256);

    /**
     * Retrieves the content stored with @ref setContent.
     */
    const std::string &getContent() { return content; }
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestFastPlyReader, TestSet::perBuild());

void TestFastPlyReader::setContent(const std::string &header, size_t payloadBytes)
{
    content = header + std::string(payloadBytes, '\0');
}

Reader *TestFastPlyReader::factory(
    const std::string &content, const std::string &filename,
    float smooth, float maxRadius) const
{
    return new Reader(MemoryReaderFactory(content), filename, smooth, maxRadius);
}

void TestFastPlyReader::testEmpty()
{
    setContent("");
    boost::scoped_ptr<Reader> r(factory(content));
}

void TestFastPlyReader::testBadSignature()
{
    setContent("ply no not really");
    boost::scoped_ptr<Reader> r(factory(content));
}

void TestFastPlyReader::testBadFormatFormat()
{
    setContent("ply\nformat binary_little_endiannotreally 1.0\nelement vertex 1\nend_header\n");
    boost::scoped_ptr<Reader> r(factory(content));
}

void TestFastPlyReader::testBadFormatVersion()
{
    setContent("ply\nformat binary_little_endian 1.01\nelement vertex 1\nend_header\n");
    boost::scoped_ptr<Reader> r(factory(content));
}

void TestFastPlyReader::testBadFormatLength()
{
    setContent("ply\nformat\nelement vertex 1\nend_header\n");
    boost::scoped_ptr<Reader> r(factory(content));
}

void TestFastPlyReader::testBadElementCount()
{
    setContent("ply\nformat binary_little_endian 1.0\nelement vertex -1\nend_header\n");
    boost::scoped_ptr<Reader> r(factory(content));
}

void TestFastPlyReader::testBadElementOverflow()
{
    setContent("ply\nformat binary_little_endian 1.0\nelement vertex 123456789012345678901234567890\nend_header\n");
    boost::scoped_ptr<Reader> r(factory(content));
}

void TestFastPlyReader::testBadElementHex()
{
    setContent("ply\nformat binary_little_endian 1.0\nelement vertex 0xDEADBEEF\nend_header\n");
    boost::scoped_ptr<Reader> r(factory(content));
}

void TestFastPlyReader::testBadElementLength()
{
    setContent("ply\nformat binary_little_endian 1.0\nelement\nend_header\n");
    boost::scoped_ptr<Reader> r(factory(content));
}

void TestFastPlyReader::testBadPropertyLength()
{
    setContent("ply\nformat binary_little_endian 1.0\nelement vertex 0\nproperty int int int x\nend_header\n");
    boost::scoped_ptr<Reader> r(factory(content));
}

void TestFastPlyReader::testBadPropertyListLength()
{
    setContent("ply\nformat binary_little_endian 1.0\nelement vertex 0\nproperty list int x\nend_header\n");
    boost::scoped_ptr<Reader> r(factory(content));
}

void TestFastPlyReader::testBadPropertyListType()
{
    setContent("ply\nformat binary_little_endian 1.0\nelement vertex 0\nproperty list float int x\nend_header\n");
    boost::scoped_ptr<Reader> r(factory(content));
}

void TestFastPlyReader::testBadPropertyType()
{
    setContent("ply\nformat binary_little_endian 1.0\nelement vertex 0\nproperty int1 x\nend_header\n");
    boost::scoped_ptr<Reader> r(factory(content));
}

void TestFastPlyReader::testBadPropertyLine()
{
    setContent("ply\nformat binary_little_endian 1.0\nelement vertex 0\nproperty int\nend_header\n");
    boost::scoped_ptr<Reader> r(factory(content));
}

void TestFastPlyReader::testBadHeaderToken()
{
    setContent("ply\nformat binary_little_endian 1.0\nelement vertex 0\nfoo\nend_header\n");
    boost::scoped_ptr<Reader> r(factory(content));
}

void TestFastPlyReader::testEarlyProperty()
{
    setContent("ply\nformat binary_little_endian 1.0\nproperty int x\nelement vertex 0\nend_header\n");
    boost::scoped_ptr<Reader> r(factory(content));
}

void TestFastPlyReader::testDuplicateProperty()
{
    setContent("ply\nformat binary_little_endian 1.0\nelement vertex 0\nproperty float x\nproperty float x\nend_header\n");
    boost::scoped_ptr<Reader> r(factory(content));
}

void TestFastPlyReader::testMissingEnd()
{
    setContent("ply\nformat binary_little_endian 1.0\nelement vertex 0\nproperty int x\n");
    boost::scoped_ptr<Reader> r(factory(content));
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
    boost::scoped_ptr<Reader> r(factory(content));
    Reader::Handle handle(*r);

    Splat splats[5];
    handle.read(0, 5, &splats[0]);
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
    boost::scoped_ptr<Reader> r(factory(content));
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
    boost::scoped_ptr<Reader> r(factory(content));
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
    boost::scoped_ptr<Reader> r(factory(content));
}

void TestFastPlyReader::testFormatMissing()
{
    setContent(
        "ply\n"
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
    boost::scoped_ptr<Reader> r(factory(content));
}

void TestFastPlyReader::testReadHeader()
{
    const std::string header = 
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
    boost::scoped_ptr<Reader> r(factory(content));
    CPPUNIT_ASSERT_EQUAL(31, int(r->vertexSize));
    CPPUNIT_ASSERT_EQUAL(5, int(r->vertexCount));

    CPPUNIT_ASSERT_EQUAL(8, int(r->offsets[Reader::X]));
    CPPUNIT_ASSERT_EQUAL(4, int(r->offsets[Reader::Y]));
    CPPUNIT_ASSERT_EQUAL(0, int(r->offsets[Reader::Z]));
    CPPUNIT_ASSERT_EQUAL(14, int(r->offsets[Reader::NX]));
    CPPUNIT_ASSERT_EQUAL(18, int(r->offsets[Reader::NY]));
    CPPUNIT_ASSERT_EQUAL(22, int(r->offsets[Reader::NZ]));
    CPPUNIT_ASSERT_EQUAL(26, int(r->offsets[Reader::RADIUS]));

    CPPUNIT_ASSERT_EQUAL(int(header.size()), int(r->getHeaderSize()));
}

void TestFastPlyReader::setupRead(int numVertices)
{
    boost::scoped_array<float> vertices(new float[numVertices * 7]);
    for (int i = 0; i < numVertices; i++)
        for (int j = 0; j < 7; j++)
            vertices[i * 7 + j] = i * 100.0f + j;
    std::string header =
        "ply\n"
        "format binary_little_endian 1.0\n"
        "element vertex ";
    header += boost::lexical_cast<std::string>(numVertices);
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

template<typename ForwardIterator>
void TestFastPlyReader::verify(int offset, ForwardIterator first, ForwardIterator last)
{
    int pos = offset;
    for (ForwardIterator i = first; i != last; ++i, ++pos)
    {
        CPPUNIT_ASSERT_EQUAL(pos * 100.0f + 2.0f, i->position[0]);
        CPPUNIT_ASSERT_EQUAL(pos * 100.0f + 0.0f, i->position[1]);
        CPPUNIT_ASSERT_EQUAL(pos * 100.0f + 1.0f, i->position[2]);
        CPPUNIT_ASSERT_EQUAL(pos * 100.0f + 3.0f, i->normal[0]);
        CPPUNIT_ASSERT_EQUAL(pos * 100.0f + 4.0f, i->normal[1]);
        CPPUNIT_ASSERT_EQUAL(pos * 100.0f + 5.0f, i->normal[2]);
        CPPUNIT_ASSERT_EQUAL(2.0f * std::min(250.0f, pos * 100.0f + 6.0f), i->radius);
    }
}

void TestFastPlyReader::testRead()
{
    setupRead(5);

    boost::scoped_ptr<Reader> r(factory(content, testFilename, 2.0f, 250.0f));
    Reader::Handle h(*r);

    Splat out[4] = {};
    h.read(1, 4, out);
    CPPUNIT_ASSERT_EQUAL(0.0f, out[3].position[0]); // check for overwriting
    verify(1, out, out + 3);

    CPPUNIT_ASSERT_THROW(h.read(1, 6, out), std::out_of_range);
}

void TestFastPlyReader::testReadZero()
{
    setupRead(5);

    boost::scoped_ptr<Reader> r(factory(content, testFilename, 2.0f));
    Reader::Handle h(*r);

    Splat out[4] = {};
    h.read(1, 1, out);
    CPPUNIT_ASSERT_EQUAL(0.0f, out[0].position[0]); // check for overwriting
}

void TestFastPlyReader::testReadIterator()
{
    setupRead(10000);

    boost::scoped_ptr<Reader> r(factory(content, testFilename, 2.0f, 250.0f));
    Reader::Handle h(*r);

    std::vector<Splat> out;
    h.read(2, 9500, back_inserter(out));
    CPPUNIT_ASSERT_EQUAL(9500 - 2, int(out.size()));
    verify(2, out.begin(), out.end());

    CPPUNIT_ASSERT_THROW(h.read(1, 10001, back_inserter(out)), std::out_of_range);
}

/**
 * Tests error handling for @ref FastPly::Reader when file errors occur
 */
class TestFastPlyReaderFile : public TestFastPlyReader
{
    CPPUNIT_TEST_SUITE(TestFastPlyReaderFile);
    TEST_EXCEPTION_FILENAME(testFileNotFound, std::ios_base::failure, "not_a_real_file.ply");
    TEST_EXCEPTION_FILENAME(testFileRemoved, std::ios_base::failure, testFilename);
    CPPUNIT_TEST_SUITE_END();

public:
    void testFileNotFound();           ///< PLY file does not exist
    void testFileRemoved();            ///< File removed after the header is read
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestFastPlyReaderFile, TestSet::perBuild());

void TestFastPlyReaderFile::testFileNotFound()
{
    // Make sure it didn't get created by accident
    boost::filesystem::remove("not_a_real_file.ply");
    Reader r(SYSCALL_READER, "not_a_real_file.ply", 1.0f, 1000.0f);
}

void TestFastPlyReaderFile::testFileRemoved()
{
    setupRead(5);

    Reader r(SYSCALL_READER, testFilename, 1.0f, 1000.0f);
    boost::filesystem::remove(testFilename);
    Reader::Handle handle(r);
}

/**
 * Tests for @ref FastPly::Writer.
 */
class TestFastPlyWriter : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestFastPlyWriter);
    TEST_EXCEPTION_FILENAME(testBadFilename, std::ios_base::failure, "/not_a_valid_filename/");
    CPPUNIT_TEST(testSimple);
    CPPUNIT_TEST(testState);
    CPPUNIT_TEST(testOverrun);
    CPPUNIT_TEST_SUITE_END();
public:
    void testBadFilename();   ///< Try to write to an invalid filename, check for error
    void testSimple();        ///< Test normal operation
    void testState();         ///< Test assertions that the file is/is not open
    void testOverrun();       ///< Test writing beyond the end of the file
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestFastPlyWriter, TestSet::perBuild());

void TestFastPlyWriter::testBadFilename()
{
    Writer w(SYSCALL_WRITER);
    w.open("/not_a_valid_filename/");
}

void TestFastPlyWriter::testSimple()
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
    const typename std::size_t headerSize = expectedHeader.size();

    MemoryWriterPly w;

    w.addComment("my comment 1");
    w.addComment("my comment 23");
    w.setNumVertices(4);
    w.setNumTriangles(3);

    w.open("file");
    MLSGPU_ASSERT_EQUAL(headerSize + 87, w.getOutput("file").size());

    w.writeVertices(1, 2, vertices + 1 * 3);
    w.writeTriangles(1, 2, indices + 1 * 3);
    w.writeVertices(0, 1, vertices);
    w.writeTriangles(0, 1, indices);
    w.writeVertices(3, 1, vertices + 3 * 3);
    w.close();

    MLSGPU_ASSERT_EQUAL(headerSize + 87, w.getOutput("file").size());
    const char *data = w.getOutput("file").data();
    MLSGPU_ASSERT_EQUAL(expectedHeader, std::string(data, headerSize));
    CPPUNIT_ASSERT(0 == memcmp(data + headerSize, vertices, sizeof(vertices)));
    MLSGPU_ASSERT_EQUAL(3, data[headerSize + 48]);
    CPPUNIT_ASSERT(0 == memcmp(data + headerSize + 49, indices + 0, 12));
    MLSGPU_ASSERT_EQUAL(3, data[headerSize + 61]);
    CPPUNIT_ASSERT(0 == memcmp(data + headerSize + 62, indices + 3, 12));
    MLSGPU_ASSERT_EQUAL(3, data[headerSize + 74]);
    CPPUNIT_ASSERT(0 == memcmp(data + headerSize + 75, indices + 6, 12));
}

void TestFastPlyWriter::testState()
{
    MemoryWriterPly w;

    w.setNumVertices(2);
    w.setNumTriangles(2);

    CPPUNIT_ASSERT_THROW(w.writeVertices(0, 1, NULL), state_error);
    CPPUNIT_ASSERT_THROW(w.writeTrianglesRaw(0, 1, NULL), state_error);

    w.open("file");

    CPPUNIT_ASSERT_THROW(w.open("foo"), state_error);
    CPPUNIT_ASSERT_THROW(w.setNumVertices(3), state_error);
    CPPUNIT_ASSERT_THROW(w.setNumTriangles(3), state_error);
}

void TestFastPlyWriter::testOverrun()
{
    const float vertices[4 * 3] = {}; // content does not matter
    const std::tr1::uint32_t indices[9] = {};

    MemoryWriterPly w;
    w.setNumVertices(4);
    w.setNumTriangles(3);

    w.open("file");

    /* Just to check that normal writes work */
    w.writeVertices(0, 1, vertices);

    /* The real test */
    CPPUNIT_ASSERT_THROW(w.writeVertices(1, 4, vertices), std::out_of_range);
    CPPUNIT_ASSERT_THROW(w.writeVertices(2, Writer::size_type(-1), vertices), std::out_of_range);
    CPPUNIT_ASSERT_THROW(w.writeVertices(Writer::size_type(-1), 2, vertices), std::out_of_range);
    w.writeVertices(1, 3, vertices);

    w.writeTriangles(0, 2, indices);
    CPPUNIT_ASSERT_THROW(w.writeTriangles(1, 3, indices), std::out_of_range);
    CPPUNIT_ASSERT_THROW(w.writeTriangles(2, Writer::size_type(-1), indices), std::out_of_range);
    CPPUNIT_ASSERT_THROW(w.writeTriangles(Writer::size_type(-1), 2, indices), std::out_of_range);
}
