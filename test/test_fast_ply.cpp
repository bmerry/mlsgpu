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
#include <boost/smart_ptr/scoped_array.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/bind.hpp>
#include <boost/ref.hpp>
#include <boost/exception/all.hpp>
#include <boost/filesystem.hpp>
#include "../src/fast_ply.h"
#include "../src/splat.h"
#include "memory_reader.h"
#include "testutil.h"

using namespace std;
using namespace FastPly;

/**
 * Decorator that checks that an exception of a specific type is thrown that also
 * contains a specific filename encoded using @c boost::errinfo_file_name.
 */
template<class ExpectedException>
class FilenameExceptionTestCaseDecorator : public CppUnit::ExceptionTestCaseDecorator<ExpectedException>
{
public:
    FilenameExceptionTestCaseDecorator(CppUnit::TestCase *test, const std::string &filename)
        : CppUnit::ExceptionTestCaseDecorator<ExpectedException>(test), filename(filename) {}

private:
    const string filename;

    virtual void checkException(ExpectedException &e)
    {
        std::string *exceptionFilename = boost::get_error_info<boost::errinfo_file_name>(e);
        CPPUNIT_ASSERT(exceptionFilename != NULL);
        CPPUNIT_ASSERT_EQUAL(filename, *exceptionFilename);
    }
};

#define TEST_EXCEPTION_FILENAME(testMethod, ExceptionType, filename) \
    CPPUNIT_TEST_SUITE_ADD_TEST(                                     \
        (new FilenameExceptionTestCaseDecorator<ExceptionType>(      \
            new CppUnit::TestCaller<TestFixtureType>(                \
                context.getTestNameFor(#testMethod),                 \
                &TestFixtureType::testMethod,                        \
                context.makeFixture()), filename)))

static const string testFilename = "test_fast_ply.ply";

/**
 * Abstract base class for testing subclasses of @ref FastPly::ReaderBase.
 */
class TestFastPlyReaderBase : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestFastPlyReaderBase);
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
    CPPUNIT_TEST_SUITE_END_ABSTRACT();

private:
    string content;                    ///< Convenience data store for the file content

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
     * Create an instance of a reader for the appropriate subclass. This
     * simply creates the file and then calls
     * #factory(const string &, float) const.
     *
     * @param fileContent     Data to place in the file
     * @param filename        Filename to be used for the file, if a file is written
     * @param smooth          Smoothing factor to pass to the constructor
     *
     * @return A reader that will read @a fileContent.
     * @throw boost::exception if the header is invalid.
     */
    ReaderBase *factory(const string &fileContent,
                        const string &filename = testFilename,
                        float smooth = 1.0f) const;

    /**
     * Create an instance of a reader for the appropriate subclass. If an
     * exception is thrown, it must contain the filename as a @c
     * boost::errinfo_file_name attached to the exception. If the subclass is
     * not file-based, the factory function itself must ensure this.
     *
     * @param filename        Filename to be used for the file, if a file is written
     * @param smooth          Smoothing factor to pass to the constructor
     *
     * @return A reader that will read fileContext.
     * @throw boost::exception if the header is invalid.
     */
    virtual ReaderBase *factory(const string &filename,
                                float smooth) const = 0;

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
    void testRead();                   ///< Tests @ref FastPly::ReaderBase::Handle::read with a pointer
    void testReadZero();               ///< Tests a zero-splat read
    void testReadIterator();           ///< Tests @ref FastPly::ReaderBase::Handle::read with an output iterator
    /** @} */

    /**
     * Sets @ref content to @a header plus @a payloadBytes bytes of arbitrary data.
     * @a payloadBytes defaults to a medium-sized value so that the negative tests can be
     * sure that a format error is not due to a short file.
     */
    void setContent(const string &header, size_t payloadBytes = 256);

    /**
     * Retrieves the content stored with @ref setContent.
     */
    const std::string &getContent() { return content; }
};

void TestFastPlyReaderBase::setContent(const string &header, size_t payloadBytes)
{
    content = header + string(payloadBytes, '\0');
}

ReaderBase *TestFastPlyReaderBase::factory(const string &content, const string &filename, float smooth) const
{
    ofstream out(filename.c_str(), ios::out | ios::binary);
    out.write(content.data(), content.size());
    out.close();
    CPPUNIT_ASSERT(out);

    return factory(filename, smooth);
}

void TestFastPlyReaderBase::testEmpty()
{
    setContent("");
    boost::scoped_ptr<ReaderBase> r(factory(content));
}

void TestFastPlyReaderBase::testBadSignature()
{
    setContent("ply no not really");
    boost::scoped_ptr<ReaderBase> r(factory(content));
}

void TestFastPlyReaderBase::testBadFormatFormat()
{
    setContent("ply\nformat binary_little_endiannotreally 1.0\nelement vertex 1\nend_header\n");
    boost::scoped_ptr<ReaderBase> r(factory(content));
}

void TestFastPlyReaderBase::testBadFormatVersion()
{
    setContent("ply\nformat binary_little_endian 1.01\nelement vertex 1\nend_header\n");
    boost::scoped_ptr<ReaderBase> r(factory(content));
}

void TestFastPlyReaderBase::testBadFormatLength()
{
    setContent("ply\nformat\nelement vertex 1\nend_header\n");
    boost::scoped_ptr<ReaderBase> r(factory(content));
}

void TestFastPlyReaderBase::testBadElementCount()
{
    setContent("ply\nformat binary_little_endian 1.0\nelement vertex -1\nend_header\n");
    boost::scoped_ptr<ReaderBase> r(factory(content));
}

void TestFastPlyReaderBase::testBadElementOverflow()
{
    setContent("ply\nformat binary_little_endian 1.0\nelement vertex 123456789012345678901234567890\nend_header\n");
    boost::scoped_ptr<ReaderBase> r(factory(content));
}

void TestFastPlyReaderBase::testBadElementHex()
{
    setContent("ply\nformat binary_little_endian 1.0\nelement vertex 0xDEADBEEF\nend_header\n");
    boost::scoped_ptr<ReaderBase> r(factory(content));
}

void TestFastPlyReaderBase::testBadElementLength()
{
    setContent("ply\nformat binary_little_endian 1.0\nelement\nend_header\n");
    boost::scoped_ptr<ReaderBase> r(factory(content));
}

void TestFastPlyReaderBase::testBadPropertyLength()
{
    setContent("ply\nformat binary_little_endian 1.0\nelement vertex 0\nproperty int int int x\nend_header\n");
    boost::scoped_ptr<ReaderBase> r(factory(content));
}

void TestFastPlyReaderBase::testBadPropertyListLength()
{
    setContent("ply\nformat binary_little_endian 1.0\nelement vertex 0\nproperty list int x\nend_header\n");
    boost::scoped_ptr<ReaderBase> r(factory(content));
}

void TestFastPlyReaderBase::testBadPropertyListType()
{
    setContent("ply\nformat binary_little_endian 1.0\nelement vertex 0\nproperty list float int x\nend_header\n");
    boost::scoped_ptr<ReaderBase> r(factory(content));
}

void TestFastPlyReaderBase::testBadPropertyType()
{
    setContent("ply\nformat binary_little_endian 1.0\nelement vertex 0\nproperty int1 x\nend_header\n");
    boost::scoped_ptr<ReaderBase> r(factory(content));
}

void TestFastPlyReaderBase::testBadPropertyLine()
{
    setContent("ply\nformat binary_little_endian 1.0\nelement vertex 0\nproperty int\nend_header\n");
    boost::scoped_ptr<ReaderBase> r(factory(content));
}

void TestFastPlyReaderBase::testBadHeaderToken()
{
    setContent("ply\nformat binary_little_endian 1.0\nelement vertex 0\nfoo\nend_header\n");
    boost::scoped_ptr<ReaderBase> r(factory(content));
}

void TestFastPlyReaderBase::testEarlyProperty()
{
    setContent("ply\nformat binary_little_endian 1.0\nproperty int x\nelement vertex 0\nend_header\n");
    boost::scoped_ptr<ReaderBase> r(factory(content));
}

void TestFastPlyReaderBase::testDuplicateProperty()
{
    setContent("ply\nformat binary_little_endian 1.0\nelement vertex 0\nproperty float x\nproperty float x\nend_header\n");
    boost::scoped_ptr<ReaderBase> r(factory(content));
}

void TestFastPlyReaderBase::testMissingEnd()
{
    setContent("ply\nformat binary_little_endian 1.0\nelement vertex 0\nproperty int x\n");
    boost::scoped_ptr<ReaderBase> r(factory(content));
}

void TestFastPlyReaderBase::testShortFile()
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
    boost::scoped_ptr<ReaderBase> r(factory(content));
    boost::scoped_ptr<ReaderBase::Handle> handle(r->createHandle());

    Splat splats[5];
    handle->read(0, 5, &splats[0]);
}

void TestFastPlyReaderBase::testList()
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
    boost::scoped_ptr<ReaderBase> r(factory(content));
}

void TestFastPlyReaderBase::testNotFloat()
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
    boost::scoped_ptr<ReaderBase> r(factory(content));
}

void TestFastPlyReaderBase::testFormatAscii()
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
    boost::scoped_ptr<ReaderBase> r(factory(content));
}

void TestFastPlyReaderBase::testFormatMissing()
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
    boost::scoped_ptr<ReaderBase> r(factory(content));
}

void TestFastPlyReaderBase::testReadHeader()
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
    boost::scoped_ptr<ReaderBase> r(factory(content));
    CPPUNIT_ASSERT_EQUAL(31, int(r->vertexSize));
    CPPUNIT_ASSERT_EQUAL(5, int(r->vertexCount));

    CPPUNIT_ASSERT_EQUAL(8, int(r->offsets[ReaderBase::X]));
    CPPUNIT_ASSERT_EQUAL(4, int(r->offsets[ReaderBase::Y]));
    CPPUNIT_ASSERT_EQUAL(0, int(r->offsets[ReaderBase::Z]));
    CPPUNIT_ASSERT_EQUAL(14, int(r->offsets[ReaderBase::NX]));
    CPPUNIT_ASSERT_EQUAL(18, int(r->offsets[ReaderBase::NY]));
    CPPUNIT_ASSERT_EQUAL(22, int(r->offsets[ReaderBase::NZ]));
    CPPUNIT_ASSERT_EQUAL(26, int(r->offsets[ReaderBase::RADIUS]));

    CPPUNIT_ASSERT_EQUAL(int(header.size()), int(r->getHeaderSize()));
}

void TestFastPlyReaderBase::setupRead(int numVertices)
{
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

template<typename ForwardIterator>
void TestFastPlyReaderBase::verify(int offset, ForwardIterator first, ForwardIterator last)
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
        CPPUNIT_ASSERT_EQUAL(2.0f * (pos * 100.0f + 6.0f), i->radius);
    }
}

void TestFastPlyReaderBase::testRead()
{
    setupRead(5);

    boost::scoped_ptr<ReaderBase> r(factory(content, testFilename, 2.0f));
    boost::scoped_ptr<ReaderBase::Handle> h(r->createHandle());

    Splat out[4] = {};
    h->read(1, 4, out);
    CPPUNIT_ASSERT_EQUAL(0.0f, out[3].position[0]); // check for overwriting
    verify(1, out, out + 3);

    CPPUNIT_ASSERT_THROW(h->read(1, 6, out), std::out_of_range);
}

void TestFastPlyReaderBase::testReadZero()
{
    setupRead(5);

    boost::scoped_ptr<ReaderBase> r(factory(content, testFilename, 2.0f));
    boost::scoped_ptr<ReaderBase::Handle> h(r->createHandle());

    Splat out[4] = {};
    h->read(1, 1, out);
    CPPUNIT_ASSERT_EQUAL(0.0f, out[0].position[0]); // check for overwriting
}

void TestFastPlyReaderBase::testReadIterator()
{
    setupRead(10000);

    boost::scoped_ptr<ReaderBase> r(factory(content, testFilename, 2.0f));
    boost::scoped_ptr<ReaderBase::Handle> h(r->createHandle());

    vector<Splat> out;
    h->read(2, 9500, back_inserter(out));
    CPPUNIT_ASSERT_EQUAL(9500 - 2, int(out.size()));
    verify(2, out.begin(), out.end());

    CPPUNIT_ASSERT_THROW(h->read(1, 10001, back_inserter(out)), std::out_of_range);
}

/**
 * Tests for @ref FastPly::ReaderBase for subclasses that really use files
 */
class TestFastPlyReaderBaseFile : public TestFastPlyReaderBase
{
    CPPUNIT_TEST_SUB_SUITE(TestFastPlyReaderBaseFile, TestFastPlyReaderBase);
    TEST_EXCEPTION_FILENAME(testFileNotFound, std::ios_base::failure, "not_a_real_file.ply");
    TEST_EXCEPTION_FILENAME(testFileRemoved, std::ios_base::failure, testFilename);
    CPPUNIT_TEST_SUITE_END_ABSTRACT();

public:
    void testFileNotFound();           ///< PLY file does not exist
    void testFileRemoved();            ///< File removed after the header is read
};

void TestFastPlyReaderBaseFile::testFileNotFound()
{
    // Make sure it didn't get created by accident
    boost::filesystem::remove("not_a_real_file.ply");
    boost::scoped_ptr<ReaderBase> r(factory("not_a_real_file.ply", 1.0f));
}

void TestFastPlyReaderBaseFile::testFileRemoved()
{
    setupRead(5);

    boost::scoped_ptr<ReaderBase> r(factory(getContent(), testFilename, 2.0f));
    boost::filesystem::remove(testFilename);
    boost::scoped_ptr<ReaderBase::Handle> h(r->createHandle());
}

/**
 * Tests for @ref FastPly::MmapReader
 */
class TestFastPlyMmapReader : public TestFastPlyReaderBaseFile
{
    CPPUNIT_TEST_SUB_SUITE(TestFastPlyMmapReader, TestFastPlyReaderBaseFile);
    CPPUNIT_TEST_SUITE_END();
protected:
    virtual ReaderBase *factory(const string &filename, float smooth) const;
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestFastPlyMmapReader, TestSet::perBuild());

ReaderBase *TestFastPlyMmapReader::factory(const string &filename,
                                           float smooth) const
{
    return new MmapReader(filename, smooth);
}

/**
 * Tests for @ref FastPly::SyscallReader
 */
class TestFastPlySyscallReader : public TestFastPlyReaderBaseFile
{
    CPPUNIT_TEST_SUB_SUITE(TestFastPlySyscallReader, TestFastPlyReaderBaseFile);
    CPPUNIT_TEST_SUITE_END();
protected:
    using TestFastPlyReaderBase::factory;

    virtual ReaderBase *factory(const string &filename, float smooth) const;

public:
    void testBufferTooSmall();         ///< Handle created with too small a buffer for one splat
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestFastPlySyscallReader, TestSet::perBuild());

ReaderBase *TestFastPlySyscallReader::factory(const string &filename, float smooth) const
{
    return new SyscallReader(filename, smooth);
}


/**
 * Tests for @ref MemoryReader
 */
class TestFastPlyMemoryReader : public TestFastPlyReaderBase
{
    CPPUNIT_TEST_SUB_SUITE(TestFastPlyMemoryReader, TestFastPlyReaderBase);
    CPPUNIT_TEST_SUITE_END();
protected:
    virtual ReaderBase *factory(const string &filename, float smooth) const;

private:
    mutable string fileData; ///< Backing store for the data read from @ref factory.
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestFastPlyMemoryReader, TestSet::perBuild());

ReaderBase *TestFastPlyMemoryReader::factory(const string &filename, float smooth) const
{
    // Suck the data back into memory
    ifstream in(filename.c_str(), ios::in | ios::binary);
    in.exceptions(ios::failbit);
    ostringstream buf;
    buf << in.rdbuf();
    fileData = buf.str();

    try
    {
        return new MemoryReader(fileData.data(), fileData.size(), smooth);
    }
    catch (boost::exception &e)
    {
        e << boost::errinfo_file_name(filename);
        throw;
    }
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
