/*
 * mlsgpu: surface reconstruction from point clouds
 * Copyright (C) 2013  University of Cape Town
 *
 * This file is part of mlsgpu.
 *
 * mlsgpu is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/**
 * @file
 *
 * Tests for @ref binary_io.h.
 */

#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/extensions/HelperMacros.h>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/system/error_code.hpp>
#include <boost/scoped_ptr.hpp>
#include <fstream>
#include <sstream>
#include <cctype>
#include <locale>
#include <iomanip>
#include "testutil.h"
#include "../src/binary_io.h"
#include "../src/errors.h"
#include "../src/misc.h"

static const boost::filesystem::path badPath("/not_a_real_file/");
static const BinaryIO::offset_type seekPos = 9876543210LL;

/**
 * Base class for testing all sub-classes of @ref BinaryIO.
 */
class TestBinaryIO : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestBinaryIO);
    CPPUNIT_TEST(testOpenClose);
#if DEBUG
    TEST_EXCEPTION_FILENAME(testBadFilename, std::ios::failure, badPath.string());
    CPPUNIT_TEST(testAlreadyOpen);
    CPPUNIT_TEST(testAlreadyClosed);
#endif
    CPPUNIT_TEST_SUITE_END_ABSTRACT();

public:
    /**
     * Creates a binary file with some data, so that it can be opened by readers and
     * writers alike. The file is made >4GiB to ensure 64-bit cleanliness.
     */
    virtual void setUp();

    /**
     * Erase the temporary file created by setUp.
     */
    virtual void tearDown();

protected:
    boost::filesystem::path testPath;

    /// Create a instance of the class type
    virtual BinaryIO *factory() = 0;

private:
    void testOpenClose();     ///< Test open and close functions
    void testBadFilename();   ///< Test that opening a bogus filename fails
    void testAlreadyOpen();   ///< Test that attempting to re-open an open file fails
    void testAlreadyClosed(); ///< Test that attempting to close a closed file fails
};

/**
 * Tests for a subclass of @ref BinaryReader
 */
class TestBinaryReader : public TestBinaryIO
{
    CPPUNIT_TEST_SUB_SUITE(TestBinaryReader, TestBinaryIO);
    CPPUNIT_TEST(testReadMiddle);
    CPPUNIT_TEST(testReadEnd);
    CPPUNIT_TEST(testReadPastEnd);
    CPPUNIT_TEST(testReadZero);
    CPPUNIT_TEST(testSize);
    CPPUNIT_TEST_SUITE_END_ABSTRACT();

protected:
    BinaryReader *factoryReader();  ///< Cast the factory output to @ref BinaryReader

private:
    void testReadMiddle();    ///< Test a read entirely internal to the file
    void testReadEnd();       ///< Test a read that crosses the end of file
    void testReadPastEnd();   ///< Test a read that does not intersect the file
    void testReadZero();      ///< Test reading zero bytes
    void testSize();          ///< Test @ref BinaryReader::size
};

/**
 * Tests for a subclass of @ref BinaryWriter
 */
class TestBinaryWriter : public TestBinaryIO
{
    CPPUNIT_TEST_SUB_SUITE(TestBinaryWriter, TestBinaryIO);
    CPPUNIT_TEST(testWriteExtend);
    CPPUNIT_TEST(testWriteInside);
    CPPUNIT_TEST(testWriteZero);
    CPPUNIT_TEST(testResize);
    CPPUNIT_TEST_SUITE_END_ABSTRACT();

protected:
    BinaryWriter *factoryWriter();  ///< Cast the factory output to @ref BinaryWriter

    /// Checks
    void assertContent(const std::string &expected, const boost::filesystem::path &path,
                       const CppUnit::SourceLine &sourceLine);
#define ASSERT_CONTENT(expected, path) \
    assertContent( (expected), (path), CPPUNIT_SOURCELINE())

private:
    void testWriteExtend();      ///< Write past the current end
    void testWriteInside();      ///< Write within the file
    void testWriteZero();        ///< Test a zero-byte write
    void testResize();           ///< Test @ref BinaryWriter::resize
};

#define BINARY_READER_CLASS(name, readerType) \
    class name : public TestBinaryReader \
    { \
        CPPUNIT_TEST_SUB_SUITE(name, TestBinaryReader); \
        CPPUNIT_TEST_SUITE_END(); \
    protected: \
        virtual BinaryIO *factory() { return createReader(readerType); } \
    }; \
    CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(name, TestSet::perBuild())

BINARY_READER_CLASS(TestSyscallReader, SYSCALL_READER);
BINARY_READER_CLASS(TestMmapReader, MMAP_READER);
BINARY_READER_CLASS(TestStreamReader, STREAM_READER);

#define BINARY_WRITER_CLASS(name, writerType) \
    class name : public TestBinaryReader \
    { \
        CPPUNIT_TEST_SUB_SUITE(name, TestBinaryWriter); \
        CPPUNIT_TEST_SUITE_END(); \
    protected: \
        virtual BinaryIO *factory() { return createWriter(writerType); } \
    }; \
    CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(name, TestSet::perBuild())

BINARY_WRITER_CLASS(TestSyscallWriter, SYSCALL_WRITER);
BINARY_WRITER_CLASS(TestStreamWriter, STREAM_WRITER);

void TestBinaryIO::setUp()
{
    boost::filesystem::ofstream f;
    createTmpFile(testPath, f);
    f.exceptions(std::ios::failbit | std::ios::badbit);
    f << "hello world";
    f.seekp(seekPos);
    f << "big offset";
    f.close();
}

void TestBinaryIO::tearDown()
{
    boost::filesystem::remove(testPath);
}

void TestBinaryIO::testOpenClose()
{
    boost::scoped_ptr<BinaryIO> b(factory());
    CPPUNIT_ASSERT_EQUAL(false, b->isOpen());
    CPPUNIT_ASSERT_EQUAL(std::string(), b->filename());

    b->open(testPath);
    CPPUNIT_ASSERT_EQUAL(true, b->isOpen());
    CPPUNIT_ASSERT_EQUAL(testPath.string(), b->filename());

    b->close();
    CPPUNIT_ASSERT_EQUAL(false, b->isOpen());
    CPPUNIT_ASSERT_EQUAL(std::string(), b->filename());
}

void TestBinaryIO::testBadFilename()
{
    boost::scoped_ptr<BinaryIO> b(factory());
    b->open(badPath);
}

void TestBinaryIO::testAlreadyOpen()
{
    boost::scoped_ptr<BinaryIO> b(factory());
    b->open(testPath);
    CPPUNIT_ASSERT_THROW(b->open(testPath), state_error);
}

void TestBinaryIO::testAlreadyClosed()
{
    boost::scoped_ptr<BinaryIO> b(factory());
    CPPUNIT_ASSERT_THROW(b->close(), state_error);
}


BinaryReader *TestBinaryReader::factoryReader()
{
    // Cast with reference is to ensure a bad_cast exception on failure
    return &dynamic_cast<BinaryReader &>(*factory());
}

void TestBinaryReader::testReadMiddle()
{
    char buffer[4096];
    boost::scoped_ptr<BinaryReader> b(factoryReader());

    b->open(testPath);
    buffer[8] = '?'; // sentinel value
    std::size_t bytes = b->read(buffer, 8, 1);
    MLSGPU_ASSERT_EQUAL(8, bytes);
    CPPUNIT_ASSERT_EQUAL('?', buffer[bytes]);
    CPPUNIT_ASSERT_EQUAL(std::string("ello wor"), std::string(buffer, bytes));
}

void TestBinaryReader::testReadEnd()
{
    char buffer[4096];
    boost::scoped_ptr<BinaryReader> b(factoryReader());

    b->open(testPath);
    buffer[10] = '?'; // sentinel value
    std::size_t bytes = b->read(buffer, 32, seekPos);
    MLSGPU_ASSERT_EQUAL(10, bytes);
    CPPUNIT_ASSERT_EQUAL('?', buffer[bytes]);
    CPPUNIT_ASSERT_EQUAL(std::string("big offset"), std::string(buffer, bytes));
}

void TestBinaryReader::testReadPastEnd()
{
    char buffer[4096];
    boost::scoped_ptr<BinaryReader> b(factoryReader());

    b->open(testPath);
    buffer[0] = '?'; // sentinel value
    std::size_t bytes = b->read(buffer, 32, seekPos + 1000);
    MLSGPU_ASSERT_EQUAL(0, bytes);
    CPPUNIT_ASSERT_EQUAL('?', buffer[0]);

    // Make sure that a possible attempted seek past the end did not resize the file
    CPPUNIT_ASSERT_EQUAL(seekPos + strlen("big offset"), b->size());
}

void TestBinaryReader::testReadZero()
{
    char buffer[4096];
    boost::scoped_ptr<BinaryReader> b(factoryReader());

    b->open(testPath);
    buffer[0] = '?'; // sentinel value
    std::size_t bytes = b->read(buffer, 0, 5);
    MLSGPU_ASSERT_EQUAL(0, bytes);
    CPPUNIT_ASSERT_EQUAL('?', buffer[0]);
}

void TestBinaryReader::testSize()
{
    boost::scoped_ptr<BinaryReader> b(factoryReader());
    b->open(testPath);
    MLSGPU_ASSERT_EQUAL(seekPos + strlen("big offset"), b->size());
}


BinaryWriter *TestBinaryWriter::factoryWriter()
{
    return &dynamic_cast<BinaryWriter &>(*factory());
}

/// Formats a string so that control characters are readable
static std::string prettyPrint(const std::string &in)
{
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::oct << std::setprecision(3);
    for (std::size_t i = 0; i < in.size(); i++)
    {
        if (!std::isgraph(in[i]) && in[i] != ' ')
            out << '\\' << (unsigned int) in[i];
        else
            out << in[i];
    }
    return out.str();
}

void TestBinaryWriter::assertContent(
    const std::string &expected,
    const boost::filesystem::path &path,
    const CppUnit::SourceLine &sourceLine)
{
    std::ifstream in(path.c_str(), std::ios::in | std::ios::binary);
    in.exceptions(std::ios::failbit | std::ios::badbit);
    std::ostringstream out;
    out << in.rdbuf();
    CppUnit::assertEquals(prettyPrint(expected), prettyPrint(out.str()), sourceLine, "");
}

void TestBinaryWriter::testWriteExtend()
{
    const std::string msg = "goodbye world";
    const std::string expected = std::string(10, '\0') + msg;

    boost::scoped_ptr<BinaryWriter> b(factoryWriter());
    b->open(testPath);
    std::size_t bytes = b->write(msg.data(), msg.size(), 10);
    MLSGPU_ASSERT_EQUAL(msg.size(), bytes);
    b->close();

    ASSERT_CONTENT(expected, testPath);
}

void TestBinaryWriter::testWriteInside()
{
    const std::string msg = "goodbye world";
    const std::string expected = std::string(10, '\0') + msg + std::string(20, '\0');

    boost::scoped_ptr<BinaryWriter> b(factoryWriter());
    b->open(testPath);
    b->resize(expected.size());
    std::size_t bytes = b->write(msg.data(), msg.size(), 10);
    MLSGPU_ASSERT_EQUAL(msg.size(), bytes);
    b->close();

    ASSERT_CONTENT(expected, testPath);
}

void TestBinaryWriter::testWriteZero()
{
    boost::scoped_ptr<BinaryWriter> b(factoryWriter());
    b->open(testPath);
    std::size_t bytes = b->write(NULL, 0, 20);
    MLSGPU_ASSERT_EQUAL(0, bytes);
    b->close();

    MLSGPU_ASSERT_EQUAL(0, file_size(testPath));
}

void TestBinaryWriter::testResize()
{
    boost::scoped_ptr<BinaryWriter> b(factoryWriter());
    b->open(testPath);
    b->resize(seekPos);
    b->close();

    MLSGPU_ASSERT_EQUAL(seekPos, file_size(testPath));
}
