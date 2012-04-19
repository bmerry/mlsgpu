/**
 * @file
 *
 * Test code for @ref binary_io.h.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif

#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/extensions/HelperMacros.h>
#include "../src/tr1_cstdint.h"
#include <string>
#include <sstream>
#include <algorithm>
#include <limits>
#include "testmain.h"
#include "../src/binary_io.h"

/// Tests for @ref readBinary and @ref writeBinary
class TestBinary : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestBinary);
    CPPUNIT_TEST(testInt8);
    CPPUNIT_TEST(testInt16);
    CPPUNIT_TEST(testInt32);
    CPPUNIT_TEST(testInt64);
    CPPUNIT_TEST(testUint8);
    CPPUNIT_TEST(testUint16);
    CPPUNIT_TEST(testUint32);
    CPPUNIT_TEST(testUint64);
    CPPUNIT_TEST(testFloat);
    CPPUNIT_TEST(testDouble);
    CPPUNIT_TEST_SUITE_END();

    /**
     * Tests that reading @a bin gives back @a value.
     */
    template<typename T, typename Endian>
    void testRead(const T &value, const std::string &bin);

    /**
     * Tests that writing @a value gives @a bin.
     */
    template<typename T, typename endian>
    void testWrite(const T &value, const std::string &bin);

    /**
     * Checks both direction and both endianness.
     *
     * @param value         Raw binary
     * @param size          Length of @a littleEndian
     * @param littleEndian  Little-endian string representation matching @a value.
     *
     * Note that @a size is redundant since it must equal <code>sizeof(value)</code>.
     * It is provided explicitly as a check that the type has the intended size.
     */
    template<typename T>
    void testCase(const T &value, std::size_t size, const char *littleEndian);

    void testInt8();
    void testInt16();
    void testInt32();
    void testInt64();
    void testUint8();
    void testUint16();
    void testUint32();
    void testUint64();
    void testFloat();
    void testDouble();
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestBinary, TestSet::perBuild());

template<typename T, typename Endian>
void TestBinary::testRead(const T &value, const std::string &bin)
{
    std::istringstream in(bin);
    T actual;
    readBinary<T>(in, actual, Endian());
    CPPUNIT_ASSERT_EQUAL(actual, value);
    CPPUNIT_ASSERT_EQUAL(std::size_t(in.tellg()), std::size_t(bin.size()));
}

template<typename T, typename Endian>
void TestBinary::testWrite(const T &value, const std::string &bin)
{
    std::ostringstream out(bin);
    writeBinary<T>(out, value, Endian());
    std::string actual = out.str();

    // Compare byte-by-byte rather than strings, since they're non-printable
    CPPUNIT_ASSERT_EQUAL(actual.size(), bin.size());
    for (std::size_t i = 0; i < bin.size(); i++)
    {
        CPPUNIT_ASSERT_EQUAL((unsigned int) (unsigned char) bin[i],
                             (unsigned int) (unsigned char) actual[i]);
    }
}

template<typename T>
void TestBinary::testCase(const T &value, std::size_t size, const char *littleEndian)
{
    std::string bin(littleEndian, size);
    testRead<T, boost::true_type>(value, bin);
    testWrite<T, boost::true_type>(value, bin);

    // Switch to big endian
    std::reverse(bin.begin(), bin.end());
    testRead<T, boost::false_type>(value, bin);
    testWrite<T, boost::false_type>(value, bin);
}

void TestBinary::testInt8()
{
    testCase(std::tr1::int8_t(0), 1, "\x00");
    testCase(std::tr1::int8_t(-128), 1, "\x80");
    testCase(std::tr1::int8_t(127), 1, "\x7F");
    testCase(std::tr1::int8_t(-100), 1, "\x9C");
}

void TestBinary::testInt16()
{
    testCase(std::tr1::int16_t(0), 2, "\x00\x00");
    testCase(std::tr1::int16_t(-32768), 2, "\x00\x80");
    testCase(std::tr1::int16_t(32767), 2, "\xFF\x7F");
    testCase(std::tr1::int16_t(-10000), 2, "\xF0\xD8");
}

void TestBinary::testInt32()
{
    testCase(std::tr1::int32_t(INT32_C(0)), 4, "\x00\x00\x00\x00");
    testCase(std::tr1::int32_t(INT32_C(-2147483647) - 1), 4, "\x00\x00\x00\x80");
    testCase(std::tr1::int32_t(INT32_C(2147483647)), 4, "\xFF\xFF\xFF\x7F");
    testCase(std::tr1::int32_t(INT32_C(-1000000000)), 4, "\x00\x36\x65\xC4");
}

void TestBinary::testInt64()
{
    testCase(std::tr1::int64_t(INT64_C(0)), 8, "\x00\x00\x00\x00\x00\x00\x00\x00");
    testCase(std::tr1::int64_t(INT64_C(-9223372036854775807) - 1),
             8, "\x00\x00\x00\x00\x00\x00\x00\x80");
    testCase(std::tr1::int64_t(INT64_C(9223372036854775807)),
             8, "\xFF\xFF\xFF\xFF\xFF\xFF\xFF\x7F");
    testCase(std::tr1::int64_t(INT64_C(-1000000000000000000)),
             8, "\x00\x00\x9C\x58\x4C\x49\x1F\xF2");
}

void TestBinary::testUint8()
{
    testCase(std::tr1::uint8_t(0), 1, "\x00");
    testCase(std::tr1::uint8_t(0xDE), 1, "\xDE");
    testCase(std::tr1::uint8_t(0xFF), 1, "\xFF");
}

void TestBinary::testUint16()
{
    testCase(std::tr1::uint16_t(0), 2, "\x00\x00");
    testCase(std::tr1::uint16_t(0xDEAD), 2, "\xAD\xDE");
    testCase(std::tr1::uint16_t(0xFFFF), 2, "\xFF\xFF");
}

void TestBinary::testUint32()
{
    testCase(std::tr1::uint32_t(UINT32_C(0)), 4, "\x00\x00\x00\x00");
    testCase(std::tr1::uint32_t(UINT32_C(0xDEADBEEF)), 4, "\xEF\xBE\xAD\xDE");
    testCase(std::tr1::uint32_t(UINT32_C(0xFFFFFFFF)), 4, "\xFF\xFF\xFF\xFF");
}

void TestBinary::testUint64()
{
    testCase(std::tr1::uint64_t(UINT64_C(0)),
             8, "\x00\x00\x00\x00\x00\x00\x00\x00");
    testCase(std::tr1::uint64_t(UINT64_C(0xDEADBEEFCAFEBABE)),
             8, "\xBE\xBA\xFE\xCA\xEF\xBE\xAD\xDE");
    testCase(std::tr1::uint64_t(UINT64_C(0xFFFFFFFFFFFFFFFF)),
             8, "\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF");
}

void TestBinary::testFloat()
{
    testCase(0.0f, 4, "\x00\x00\x00\x00");
    testCase(1.0f, 4, "\x00\x00\x80\x3F");
    testCase(-1.0f, 4, "\x00\x00\x80\xBF");
    testCase(-1.0353395f, 4, "\x01\x86\x84\xBF");
    testCase(std::numeric_limits<float>::infinity(), 4, "\x00\x00\x80\x7F");
}

void TestBinary::testDouble()
{
    testCase(0.0, 8, "\x00\x00\x00\x00\x00\x00\x00\x00");
    testCase(1.0, 8, "\x00\x00\x00\x00\x00\x00\xF0\x3F");
    testCase(-1.0, 8, "\x00\x00\x00\x00\x00\x00\xF0\xBF");
    testCase(3.14, 8, "\x1F\x85\xEB\x51\xB8\x1E\x09\x40");
    testCase(std::numeric_limits<double>::infinity(), 8, "\x00\x00\x00\x00\x00\x00\xF0\x7F");
}
