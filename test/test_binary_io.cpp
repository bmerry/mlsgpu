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
#include <tr1/cstdint>
#include <string>
#include <sstream>
#include <algorithm>
#include <limits>
#include "testmain.h"
#include "../src/binary_io.h"

class TestWriteBinary : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestWriteBinary);
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
     * Checks that writing @a value gives the string @a littleEndian when in
     * little endian mode and its reverse in big endian mode. The string length
     * is taken as @a size rather than from a NULL terminator, so
     * it is binary-safe.
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
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestWriteBinary, TestSet::perBuild());

template<typename T>
void TestWriteBinary::testCase(const T &value, std::size_t size, const char *littleEndian)
{
    std::ostringstream little;
    std::string expectedLittle(littleEndian, size);
    CPPUNIT_ASSERT_EQUAL(sizeof(T), size);

    writeBinary(little, value, boost::true_type());
    std::string actualLittle = little.str();
    // Test the strings one character at a time, to give more useful errors
    CPPUNIT_ASSERT_EQUAL(expectedLittle.size(), actualLittle.size());
    for (std::size_t i = 0; i < actualLittle.size(); i++)
    {
        CPPUNIT_ASSERT_EQUAL((unsigned int) (unsigned char) expectedLittle[i],
                             (unsigned int) (unsigned char) actualLittle[i]);
    }

    std::ostringstream big;
    std::string expectedBig = expectedLittle;
    std::reverse(expectedBig.begin(), expectedBig.end());

    writeBinary(big, value, boost::false_type());
    std::string actualBig = big.str();
    // Test the strings one character at a time, to give more useful errors
    CPPUNIT_ASSERT_EQUAL(expectedBig.size(), actualBig.size());
    for (std::size_t i = 0; i < actualBig.size(); i++)
    {
        CPPUNIT_ASSERT_EQUAL((unsigned int) expectedBig[i], (unsigned int) actualBig[i]);
    }
}

void TestWriteBinary::testInt8()
{
    testCase(std::tr1::int8_t(0), 1, "\x00");
    testCase(std::tr1::int8_t(-128), 1, "\x80");
    testCase(std::tr1::int8_t(127), 1, "\x7F");
    testCase(std::tr1::int8_t(-100), 1, "\x9C");
}

void TestWriteBinary::testInt16()
{
    testCase(std::tr1::int16_t(0), 2, "\x00\x00");
    testCase(std::tr1::int16_t(-32768), 2, "\x00\x80");
    testCase(std::tr1::int16_t(32767), 2, "\xFF\x7F");
    testCase(std::tr1::int16_t(-10000), 2, "\xF0\xD8");
}

void TestWriteBinary::testInt32()
{
    testCase(std::tr1::int32_t(INT32_C(0)), 4, "\x00\x00\x00\x00");
    testCase(std::tr1::int32_t(INT32_C(-2147483647) - 1), 4, "\x00\x00\x00\x80");
    testCase(std::tr1::int32_t(INT32_C(2147483647)), 4, "\xFF\xFF\xFF\x7F");
    testCase(std::tr1::int32_t(INT32_C(-1000000000)), 4, "\x00\x36\x65\xC4");
}

void TestWriteBinary::testInt64()
{
    testCase(std::tr1::int64_t(INT64_C(0)), 8, "\x00\x00\x00\x00\x00\x00\x00\x00");
    testCase(std::tr1::int64_t(INT64_C(-9223372036854775807) - 1),
             8, "\x00\x00\x00\x00\x00\x00\x00\x80");
    testCase(std::tr1::int64_t(INT64_C(9223372036854775807)),
             8, "\xFF\xFF\xFF\xFF\xFF\xFF\xFF\x7F");
    testCase(std::tr1::int64_t(INT64_C(-1000000000000000000)),
             8, "\x00\x00\x9C\x58\x4C\x49\x1F\xF2");
}

void TestWriteBinary::testUint8()
{
    testCase(std::tr1::uint8_t(0), 1, "\x00");
    testCase(std::tr1::uint8_t(0xDE), 1, "\xDE");
    testCase(std::tr1::uint8_t(0xFF), 1, "\xFF");
}

void TestWriteBinary::testUint16()
{
    testCase(std::tr1::uint16_t(0), 2, "\x00\x00");
    testCase(std::tr1::uint16_t(0xDEAD), 2, "\xAD\xDE");
    testCase(std::tr1::uint16_t(0xFFFF), 2, "\xFF\xFF");
}

void TestWriteBinary::testUint32()
{
    testCase(std::tr1::uint32_t(UINT32_C(0)), 4, "\x00\x00\x00\x00");
    testCase(std::tr1::uint32_t(UINT32_C(0xDEADBEEF)), 4, "\xEF\xBE\xAD\xDE");
    testCase(std::tr1::uint32_t(UINT32_C(0xFFFFFFFF)), 4, "\xFF\xFF\xFF\xFF");
}

void TestWriteBinary::testUint64()
{
    testCase(std::tr1::uint64_t(UINT64_C(0)),
             8, "\x00\x00\x00\x00\x00\x00\x00\x00");
    testCase(std::tr1::uint64_t(UINT64_C(0xDEADBEEFCAFEBABE)),
             8, "\xBE\xBA\xFE\xCA\xEF\xBE\xAD\xDE");
    testCase(std::tr1::uint64_t(UINT64_C(0xFFFFFFFFFFFFFFFF)),
             8, "\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF");
}

void TestWriteBinary::testFloat()
{
    testCase(0.0f, 4, "\x00\x00\x00\x00");
    testCase(1.0f, 4, "\x00\x00\x80\x3F");
    testCase(-1.0f, 4, "\x00\x00\x80\xBF");
    testCase(-1.0353395f, 4, "\x01\x86\x84\xBF");
    testCase(std::numeric_limits<float>::infinity(), 4, "\x00\x00\x80\x7F");
}

void TestWriteBinary::testDouble()
{
    testCase(0.0, 8, "\x00\x00\x00\x00\x00\x00\x00\x00");
    testCase(1.0, 8, "\x00\x00\x00\x00\x00\x00\xF0\x3F");
    testCase(-1.0, 8, "\x00\x00\x00\x00\x00\x00\xF0\xBF");
    testCase(3.14, 8, "\x1F\x85\xEB\x51\xB8\x1E\x09\x40");
    testCase(std::numeric_limits<double>::infinity(), 8, "\x00\x00\x00\x00\x00\x00\xF0\x7F");
}
