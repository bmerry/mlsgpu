/**
 * @file
 *
 * Test code for PLY file format support.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif

#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/extensions/HelperMacros.h>
#include <sstream>
#include <string>
#include "../extras/ply.h"
#include "../test/testutil.h"

using namespace std;

/**
 * Test fixture for PLY::Reader.
 */
class TestPlyReader : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestPlyReader);
    CPPUNIT_TEST_EXCEPTION(testEmpty, PLY::FormatError);
    CPPUNIT_TEST_EXCEPTION(testBadSignature, PLY::FormatError);
    CPPUNIT_TEST_EXCEPTION(testBadFormatFormat, PLY::FormatError);
    CPPUNIT_TEST_EXCEPTION(testBadFormatVersion, PLY::FormatError);
    CPPUNIT_TEST_EXCEPTION(testBadFormatLength, PLY::FormatError);
    CPPUNIT_TEST_EXCEPTION(testBadElementCount, PLY::FormatError);
    CPPUNIT_TEST_EXCEPTION(testBadElementOverflow, PLY::FormatError);
    CPPUNIT_TEST_EXCEPTION(testBadElementHex, PLY::FormatError);
    CPPUNIT_TEST_EXCEPTION(testBadElementLength, PLY::FormatError);
    CPPUNIT_TEST_EXCEPTION(testBadPropertyLength, PLY::FormatError);
    CPPUNIT_TEST_EXCEPTION(testBadPropertyListLength, PLY::FormatError);
    CPPUNIT_TEST_EXCEPTION(testBadPropertyListType, PLY::FormatError);
    CPPUNIT_TEST_EXCEPTION(testBadPropertyType, PLY::FormatError);
    CPPUNIT_TEST_EXCEPTION(testBadHeaderToken, PLY::FormatError);
    CPPUNIT_TEST_EXCEPTION(testDuplicateElement, PLY::FormatError);
    CPPUNIT_TEST_EXCEPTION(testEarlyProperty, PLY::FormatError);
    CPPUNIT_TEST_EXCEPTION(testDuplicateProperty, PLY::FormatError);
    CPPUNIT_TEST_EXCEPTION(testMissingEnd, PLY::FormatError);
    CPPUNIT_TEST_EXCEPTION(testWrongPosition, std::invalid_argument);
    CPPUNIT_TEST_EXCEPTION(testWrongRange, std::invalid_argument);
    CPPUNIT_TEST_EXCEPTION(testDereferenceEnd, std::invalid_argument);
    CPPUNIT_TEST_EXCEPTION(testSkipToStarted, PLY::FormatError);
    // TODO rewrite CPPUNIT_TEST_EXCEPTION(testSkipToWrongType, std::bad_cast);
    CPPUNIT_TEST_EXCEPTION(testSkipToMissed, PLY::FormatError);
    CPPUNIT_TEST_EXCEPTION(testSkipToMissing, PLY::FormatError);
    CPPUNIT_TEST(testFormatAscii);
    CPPUNIT_TEST(testFormatLittleEndian);
    CPPUNIT_TEST(testFormatBigEndian);
    CPPUNIT_TEST(testListType);
    CPPUNIT_TEST(testSimpleType);
    CPPUNIT_TEST(testComment);
    CPPUNIT_TEST(testReadFieldAscii);
    CPPUNIT_TEST(testReadFieldLittleEndian);
    CPPUNIT_TEST(testReadFieldBigEndian);
    CPPUNIT_TEST(testSkip);
    CPPUNIT_TEST(testSkipTo);
    CPPUNIT_TEST(testSkipToEmpty);
    CPPUNIT_TEST_SUITE_END();

private:
    /**
     * Convenient place to store a PLY header for tests to access.
     */
    stringbuf content;

    template<typename T>
    void testReadFieldInternal(PLY::FileFormat format, const string &s, T expected);
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
    void testDuplicateElement();       ///< Two elements with the same name
    void testEarlyProperty();          ///< Property line before any element line
    void testDuplicateProperty();      ///< Two properties with the same name
    void testMissingEnd();             ///< Header ends without @c end_header
    void testWrongPosition();          ///< Dereferences a non-current iterator
    void testWrongRange();             ///< Dereferences an iterator from a non-current range
    void testDereferenceEnd();         ///< Dereferences a past-the-end iterator
    void testSkipToStarted();          ///< @c skipTo an element we're partway through
    void testSkipToWrongType();        ///< @c skipTo with wrong element type
    void testSkipToMissed();           ///< @c skipTo an element we've already read
    void testSkipToMissing();          ///< @c skipTo an element that doesn't exist
    /** @} */

    /**
     * @name Positive tests
     * @{
     */
    void testFormatAscii();            ///< Checks that format is detected for @c ascii
    void testFormatLittleEndian();     ///< Checks that format is detected for @c binary_little_endian
    void testFormatBigEndian();        ///< Checks that format is detected for @c binary_big_endian
    void testListType();               ///< Checks that a list property is parsed
    void testSimpleType();             ///< Checks that a non-list property is parsed
    void testComment();                ///< Checks that comments are ignored
    void testReadFieldAscii();         ///< Tests parsing of ASCII fields of each type
    void testReadFieldLittleEndian();  ///< Tests parsing of little endian binary fields of each type
    void testReadFieldBigEndian();     ///< Tests parsing of big endian binary fields of each type
    void testSkip();                   ///< Tests @ref PLY::ElementRangeReaderBase::skip
    void testSkipTo();                 ///< Tests @ref PLY::Reader::skipTo
    void testSkipToEmpty();            ///< Tests @ref PLY::Reader::skipTo with empty elements
    /** @} */
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestPlyReader, TestSet::perBuild());

void TestPlyReader::testEmpty()
{
    content.str("");
    PLY::Reader(&content).readHeader();
}

void TestPlyReader::testBadSignature()
{
    content.str("ply no not really");
    PLY::Reader(&content).readHeader();
}

void TestPlyReader::testBadFormatFormat()
{
    content.str("ply\nformat asciinotreally 1.0\nelement vertex 1\nend_header\n");
    PLY::Reader(&content).readHeader();
}

void TestPlyReader::testBadFormatVersion()
{
    content.str("ply\nformat ascii 1.01\nelement vertex 1\nend_header\n");
    PLY::Reader(&content).readHeader();
}

void TestPlyReader::testBadFormatLength()
{
    content.str("ply\nformat\nelement vertex 1\nend_header\n");
    PLY::Reader(&content).readHeader();
}

void TestPlyReader::testBadElementCount()
{
    content.str("ply\nformat ascii 1.0\nelement vertex -1\nend_header\n");
    PLY::Reader(&content).readHeader();
}

void TestPlyReader::testBadElementOverflow()
{
    content.str("ply\nformat ascii 1.0\nelement vertex 123456789012345678901234567890\nend_header\n");
    PLY::Reader(&content).readHeader();
}

void TestPlyReader::testBadElementHex()
{
    content.str("ply\nformat ascii 1.0\nelement vertex 0xDEADBEEF\nend_header\n");
    PLY::Reader(&content).readHeader();
}

void TestPlyReader::testBadElementLength()
{
    content.str("ply\nformat ascii 1.0\nelement\nend_header\n");
    PLY::Reader(&content).readHeader();
}

void TestPlyReader::testBadPropertyLength()
{
    content.str("ply\nformat ascii 1.0\nelement vertex 0\nproperty int int int x\nend_header\n");
    PLY::Reader(&content).readHeader();
}

void TestPlyReader::testBadPropertyListLength()
{
    content.str("ply\nformat ascii 1.0\nelement vertex 0\nproperty list int x\nend_header\n");
    PLY::Reader(&content).readHeader();
}

void TestPlyReader::testBadPropertyListType()
{
    content.str("ply\nformat ascii 1.0\nelement vertex 0\nproperty list float int x\nend_header\n");
    PLY::Reader(&content).readHeader();
}

void TestPlyReader::testBadPropertyType()
{
    content.str("ply\nformat ascii 1.0\nelement vertex 0\nproperty int1 x\nend_header\n");
    PLY::Reader(&content).readHeader();
}

void TestPlyReader::testBadHeaderToken()
{
    content.str("ply\nformat ascii 1.0\nelement vertex 0\nfoo\nend_header\n");
    PLY::Reader(&content).readHeader();
}

void TestPlyReader::testDuplicateElement()
{
    content.str("ply\nformat ascii 1.0\nelement vertex 0\nelement vertex 0\nend_header\n");
    PLY::Reader(&content).readHeader();
}

void TestPlyReader::testEarlyProperty()
{
    content.str("ply\nformat ascii 1.0\nproperty int x\nelement vertex 0\nend_header\n");
    PLY::Reader(&content).readHeader();
}

void TestPlyReader::testDuplicateProperty()
{
    content.str("ply\nformat ascii 1.0\nelement vertex 0\nproperty int x\nproperty float x\nend_header\n");
    PLY::Reader(&content).readHeader();
}

void TestPlyReader::testMissingEnd()
{
    content.str("ply\nformat ascii 1.0\nelement vertex 0\nproperty int x\n");
    PLY::Reader(&content).readHeader();
}

void TestPlyReader::testWrongPosition()
{
    content.str("ply\nformat ascii 1.0\nelement vertex 2\nproperty int x\nend_header\n"
                "1\n2\n");
    PLY::Reader r(&content);
    r.readHeader();
    PLY::ElementRangeReader<PLY::EmptyBuilder> &reader = r.skipTo<PLY::EmptyBuilder>("vertex");
    PLY::ElementRangeReader<PLY::EmptyBuilder>::iterator i = reader.begin();
    *i;
    *i; // tries to re-read: should throw
}

void TestPlyReader::testDereferenceEnd()
{
    content.str("ply\nformat ascii 1.0\nelement vertex 2\nproperty int x\nend_header\n"
                "1\n2\n");
    PLY::Reader r(&content);
    r.readHeader();
    PLY::ElementRangeReader<PLY::EmptyBuilder> &reader = r.skipTo<PLY::EmptyBuilder>("vertex");
    PLY::ElementRangeReader<PLY::EmptyBuilder>::iterator i = reader.begin();
    *i++;
    *i++;
    *i; // dereferences past the end, should throw
}

void TestPlyReader::testWrongRange()
{
    content.str("ply\nformat ascii 1.0\n"
                "element vertex 2\nproperty int x\n"
                "element dummy 2\nproperty int y\n"
                "end_header\n"
                "1\n2\n3\n4\n");
    PLY::Reader r(&content);
    r.readHeader();
    PLY::ElementRangeReader<PLY::EmptyBuilder> &vertex = r.skipTo<PLY::EmptyBuilder>("vertex");
    PLY::ElementRangeReader<PLY::EmptyBuilder>::iterator i, j;
    i = vertex.begin();
    j = i;
    *j++;
    *j++;
    r.skipTo<PLY::EmptyBuilder>("dummy");
    *i; // accesses from vertex, should throw
}

void TestPlyReader::testSkipToStarted()
{
    content.str("ply\nformat ascii 1.0\n"
                "element vertex 2\nproperty int x\n"
                "element dummy 2\nproperty int y\n"
                "end_header\n"
                "1\n2\n3\n4\n");
    PLY::Reader r(&content);
    r.readHeader();
    PLY::ElementRangeReader<PLY::EmptyBuilder> &vertex = r.skipTo<PLY::EmptyBuilder>("vertex");
    PLY::ElementRangeReader<PLY::EmptyBuilder>::iterator i = vertex.begin();
    *i++;
    r.skipTo<PLY::EmptyBuilder>("vertex");
}

void TestPlyReader::testSkipToWrongType()
{
    // TODO rewrite
}

void TestPlyReader::testSkipToMissed()
{
    content.str("ply\nformat ascii 1.0\n"
                "element vertex 2\nproperty int x\n"
                "element dummy 2\nproperty int y\n"
                "end_header\n"
                "1\n2\n3\n4\n");
    PLY::Reader r(&content);
    r.readHeader();
    r.skipTo<PLY::EmptyBuilder>("dummy");
    r.skipTo<PLY::EmptyBuilder>("vertex");
}

void TestPlyReader::testSkipToMissing()
{
    content.str("ply\nformat ascii 1.0\n"
                "element vertex 2\nproperty int x\n"
                "end_header\n"
                "1\n2\n");
    PLY::Reader r(&content);
    r.readHeader();
    r.skipTo<PLY::EmptyBuilder>("missing");
}

void TestPlyReader::testFormatAscii()
{
    content.str("ply\nformat ascii 1.0\nelement vertex 0\nproperty int x\nend_header");
    PLY::Reader r(&content);
    r.readHeader();
    CPPUNIT_ASSERT_EQUAL(PLY::FILE_FORMAT_ASCII, r.format);
}

void TestPlyReader::testFormatLittleEndian()
{
    content.str("ply\nformat binary_little_endian 1.0\nelement vertex 0\nproperty int x\nend_header");
    PLY::Reader r(&content);
    r.readHeader();
    CPPUNIT_ASSERT_EQUAL(PLY::FILE_FORMAT_LITTLE_ENDIAN, r.format);
}

void TestPlyReader::testFormatBigEndian()
{
    content.str("ply\nformat binary_big_endian 1.0\nelement vertex 0\nproperty int x\nend_header");
    PLY::Reader r(&content);
    r.readHeader();
    CPPUNIT_ASSERT_EQUAL(PLY::FILE_FORMAT_BIG_ENDIAN, r.format);
}

void TestPlyReader::testListType()
{
    content.str("ply\nformat ascii 1.0\nelement vertex 0\nproperty list int float y\nproperty list uint8 float64 x\nend_header");
    PLY::Reader r(&content);
    r.readHeader();
    CPPUNIT_ASSERT_EQUAL(1, int(distance(r.begin(), r.end())));
    const PLY::ElementRangeReaderBase &e = *r.begin();
    CPPUNIT_ASSERT_EQUAL(size_t(2), e.getProperties().size());
    const PLY::PropertyType &y = *e.getProperties().begin();
    const PLY::PropertyType &x = *boost::next(e.getProperties().begin());

    CPPUNIT_ASSERT_EQUAL(string("y"), y.name);
    CPPUNIT_ASSERT(y.isList);
    CPPUNIT_ASSERT_EQUAL(PLY::INT32, y.lengthType);
    CPPUNIT_ASSERT_EQUAL(PLY::FLOAT32, y.valueType);

    CPPUNIT_ASSERT_EQUAL(string("x"), x.name);
    CPPUNIT_ASSERT(x.isList);
    CPPUNIT_ASSERT_EQUAL(PLY::UINT8, x.lengthType);
    CPPUNIT_ASSERT_EQUAL(PLY::FLOAT64, x.valueType);
}

void TestPlyReader::testSimpleType()
{
    content.str("ply\nformat ascii 1.0\nelement vertex 0\nproperty float y\nproperty uint16 x\nend_header");
    PLY::Reader r(&content);
    r.readHeader();
    CPPUNIT_ASSERT_EQUAL(1, int(distance(r.begin(), r.end())));
    const PLY::ElementRangeReaderBase &e = *r.begin();
    CPPUNIT_ASSERT_EQUAL(size_t(2), e.getProperties().size());
    const PLY::PropertyType &y = *e.getProperties().begin();
    const PLY::PropertyType &x = *boost::next(e.getProperties().begin());

    CPPUNIT_ASSERT_EQUAL(string("y"), y.name);
    CPPUNIT_ASSERT(!y.isList);
    CPPUNIT_ASSERT_EQUAL(PLY::FLOAT32, y.valueType);

    CPPUNIT_ASSERT_EQUAL(string("x"), x.name);
    CPPUNIT_ASSERT(!x.isList);
    CPPUNIT_ASSERT_EQUAL(PLY::UINT16, x.valueType);
}

void TestPlyReader::testComment()
{
    content.str("ply\ncomment Hello world\nobj_info Created by hand\nformat ascii 1.0\nelement vertex 0\nend_header");
    PLY::Reader(&content).readHeader();
}

template<typename T>
void TestPlyReader::testReadFieldInternal(PLY::FileFormat format, const string &s, T expected)
{
    content.str(s);
    PLY::Reader r(&content);
    r.format = format;
    r.in.exceptions(ios::failbit | ios::badbit);
    T actual = r.readField<T>();
    CPPUNIT_ASSERT_EQUAL(expected, actual);
    // tellg causes a failure if eofbit is set - not sure why it was defined
    // this way, but it is.
    if (r.in.rdstate() == ios::eofbit)
        r.in.clear();
    CPPUNIT_ASSERT_EQUAL(s.size(), size_t(r.in.tellg()));
}

void TestPlyReader::testReadFieldAscii()
{
    testReadFieldInternal(PLY::FILE_FORMAT_ASCII, "0", uint8_t(0));
    testReadFieldInternal(PLY::FILE_FORMAT_ASCII, "100", uint8_t(100));
    testReadFieldInternal(PLY::FILE_FORMAT_ASCII, "-100", int8_t(-100));
    testReadFieldInternal(PLY::FILE_FORMAT_ASCII, "10000", uint16_t(10000));
    testReadFieldInternal(PLY::FILE_FORMAT_ASCII, "-10000", int16_t(-10000));
    testReadFieldInternal(PLY::FILE_FORMAT_ASCII, "4294967295", uint32_t(0xFFFFFFFF));
    testReadFieldInternal(PLY::FILE_FORMAT_ASCII, "-2147483648", int32_t(-0x80000000LL));
    testReadFieldInternal(PLY::FILE_FORMAT_ASCII, "1", 1.0f);
    testReadFieldInternal(PLY::FILE_FORMAT_ASCII, "1.0", 1.0f);
    testReadFieldInternal(PLY::FILE_FORMAT_ASCII, "1.0e00", 1.0f);
    testReadFieldInternal(PLY::FILE_FORMAT_ASCII, "1", 1.0);
    testReadFieldInternal(PLY::FILE_FORMAT_ASCII, "1.0", 1.0);
    testReadFieldInternal(PLY::FILE_FORMAT_ASCII, "1.0e0", 1.0);

    // Value too short
    CPPUNIT_ASSERT_THROW(
        testReadFieldInternal(PLY::FILE_FORMAT_ASCII, "", 1.0f),
        PLY::FormatError);
}

void TestPlyReader::testReadFieldLittleEndian()
{
    testReadFieldInternal(PLY::FILE_FORMAT_LITTLE_ENDIAN, "\xCA", uint8_t(0xCA));
    testReadFieldInternal(PLY::FILE_FORMAT_LITTLE_ENDIAN, "\xCA", int8_t(0xCA - 0x100));
    testReadFieldInternal(PLY::FILE_FORMAT_LITTLE_ENDIAN, "\xCA\xFE", uint16_t(0xFECA));
    testReadFieldInternal(PLY::FILE_FORMAT_LITTLE_ENDIAN, "\xCA\xFE", int16_t(0xFECA - 0x10000));
    testReadFieldInternal(PLY::FILE_FORMAT_LITTLE_ENDIAN, "\xCA\xFE\xBA\xBE", uint32_t(0xBEBAFECA));
    testReadFieldInternal(PLY::FILE_FORMAT_LITTLE_ENDIAN, "\xCA\xFE\xBA\xBE", int32_t(0xBEBAFECA - 0x100000000LL));
    testReadFieldInternal(PLY::FILE_FORMAT_LITTLE_ENDIAN, string("\x00\x00\x80\x3F", 4), 1.0f);
    testReadFieldInternal(PLY::FILE_FORMAT_LITTLE_ENDIAN, string("\x00\x00\x00\x00\x00\x00\xF0\x3F", 8), 1.0);

    // Value too short
    CPPUNIT_ASSERT_THROW(
        testReadFieldInternal(PLY::FILE_FORMAT_LITTLE_ENDIAN, string("\x00\x00\x80", 3), 1.0f),
        PLY::FormatError);
}

void TestPlyReader::testReadFieldBigEndian()
{
    testReadFieldInternal(PLY::FILE_FORMAT_BIG_ENDIAN, "\xCA", uint8_t(0xCA));
    testReadFieldInternal(PLY::FILE_FORMAT_BIG_ENDIAN, "\xCA", int8_t(0xCA - 0x100));
    testReadFieldInternal(PLY::FILE_FORMAT_BIG_ENDIAN, "\xCA\xFE", uint16_t(0xCAFE));
    testReadFieldInternal(PLY::FILE_FORMAT_BIG_ENDIAN, "\xCA\xFE", int16_t(0xCAFE - 0x10000));
    testReadFieldInternal(PLY::FILE_FORMAT_BIG_ENDIAN, "\xCA\xFE\xBA\xBE", uint32_t(0xCAFEBABE));
    testReadFieldInternal(PLY::FILE_FORMAT_BIG_ENDIAN, "\xCA\xFE\xBA\xBE", int32_t(0xCAFEBABE - 0x100000000LL));
    testReadFieldInternal(PLY::FILE_FORMAT_BIG_ENDIAN, string("\x3F\x80\x00\x00", 4), 1.0f);
    testReadFieldInternal(PLY::FILE_FORMAT_BIG_ENDIAN, string("\x3F\xF0\x00\x00\x00\x00\x00\x00", 8), 1.0);

    // Value too short
    CPPUNIT_ASSERT_THROW(
        testReadFieldInternal(PLY::FILE_FORMAT_BIG_ENDIAN, string("\x3F\x80\x00", 3), 1.0f),
        PLY::FormatError);
}

void TestPlyReader::testSkip()
{
    content.str("ply\nformat ascii 1.0\n"
                "element vertex 2\nproperty int x\n"
                "element dummy 2\nproperty int y\n"
                "element face 1\nproperty list int int z\n"
                "end_header\n"
                "1\n2\n3\n4\n1 5\n");
    PLY::Reader r(&content);
    r.readHeader();
    PLY::Reader::iterator x = r.begin();
    PLY::ElementRangeReader<PLY::EmptyBuilder> &vertex = dynamic_cast<PLY::ElementRangeReader<PLY::EmptyBuilder> &>(*x++);
    PLY::ElementRangeReader<PLY::EmptyBuilder> &dummy = dynamic_cast<PLY::ElementRangeReader<PLY::EmptyBuilder> &>(*x++);
    PLY::ElementRangeReader<PLY::EmptyBuilder> &face = dynamic_cast<PLY::ElementRangeReader<PLY::EmptyBuilder> &>(*x++);
    vertex.skip();
    CPPUNIT_ASSERT_EQUAL(&*r.currentReader, static_cast<PLY::ElementRangeReaderBase *>(&dummy));
    CPPUNIT_ASSERT_EQUAL(std::tr1::uintmax_t(0), r.currentPos);
    *dummy.begin();
    CPPUNIT_ASSERT_EQUAL(std::tr1::uintmax_t(1), r.currentPos);
    dummy.skip();
    CPPUNIT_ASSERT_EQUAL(&*r.currentReader, static_cast<PLY::ElementRangeReaderBase *>(&face));
    CPPUNIT_ASSERT_EQUAL(std::tr1::uintmax_t(0), r.currentPos);
}

void TestPlyReader::testSkipTo()
{
    content.str("ply\nformat ascii 1.0\n"
                "element vertex 2\nproperty int x\n"
                "element dummy 2\nproperty int y\n"
                "element face 1\nproperty list int int z\n"
                "end_header\n"
                "1\n2\n3\n4\n1 5\n");
    PLY::Reader r(&content);
    r.readHeader();
    PLY::Reader::iterator x = r.begin();
    PLY::ElementRangeReader<PLY::EmptyBuilder> &vertex = dynamic_cast<PLY::ElementRangeReader<PLY::EmptyBuilder> &>(*x++);
    // Check that it has the right type (otherwise an exception is thrown)
    (void) & dynamic_cast<PLY::ElementRangeReader<PLY::EmptyBuilder> &>(*x++);
    PLY::ElementRangeReader<PLY::EmptyBuilder> &face = dynamic_cast<PLY::ElementRangeReader<PLY::EmptyBuilder> &>(*x++);
    CPPUNIT_ASSERT(&vertex == &r.skipTo<PLY::EmptyBuilder>("vertex"));
    CPPUNIT_ASSERT_EQUAL(&*r.currentReader, static_cast<PLY::ElementRangeReaderBase *>(&vertex));
    CPPUNIT_ASSERT_EQUAL(std::tr1::uintmax_t(0), r.currentPos);
    *vertex.begin();

    CPPUNIT_ASSERT(&face == &r.skipTo<PLY::EmptyBuilder>("face"));
    CPPUNIT_ASSERT_EQUAL(&*r.currentReader, static_cast<PLY::ElementRangeReaderBase *>(&face));
    CPPUNIT_ASSERT_EQUAL(std::tr1::uintmax_t(0), r.currentPos);
}

void TestPlyReader::testSkipToEmpty()
{
    content.str("ply\nformat ascii 1.0\n"
                "element vertex 2\nproperty int x\n"
                "element dummy 0\nproperty int y\n"
                "element face 1\nproperty list int int z\n"
                "end_header\n"
                "1\n2\n1 5\n");
    PLY::Reader r(&content);
    r.readHeader();
    PLY::Reader::iterator x = r.begin();
    PLY::ElementRangeReader<PLY::EmptyBuilder> &vertex = dynamic_cast<PLY::ElementRangeReader<PLY::EmptyBuilder> &>(*x++);
    PLY::ElementRangeReader<PLY::EmptyBuilder> &dummy = dynamic_cast<PLY::ElementRangeReader<PLY::EmptyBuilder> &>(*x++);
    PLY::ElementRangeReader<PLY::EmptyBuilder> &face = dynamic_cast<PLY::ElementRangeReader<PLY::EmptyBuilder> &>(*x++);
    PLY::ElementRangeReader<PLY::EmptyBuilder>::iterator i;
    i = vertex.begin();
    *i++;
    *i++;
    CPPUNIT_ASSERT_EQUAL(&*r.currentReader, static_cast<PLY::ElementRangeReaderBase *>(&face));
    CPPUNIT_ASSERT_EQUAL(std::tr1::uintmax_t(0), r.currentPos);
    CPPUNIT_ASSERT(&dummy == &r.skipTo<PLY::EmptyBuilder>("dummy"));
    CPPUNIT_ASSERT_EQUAL(&*r.currentReader, static_cast<PLY::ElementRangeReaderBase *>(&face));
    CPPUNIT_ASSERT_EQUAL(std::tr1::uintmax_t(0), r.currentPos);
}
