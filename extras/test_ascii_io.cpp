/**
 * @file
 *
 * Test code for @ref ascii_io.h.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif

#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/extensions/HelperMacros.h>
#include "../src/tr1_cstdint.h"
#include <boost/tr1/cmath.hpp>
#include <limits>
#include "../extras/ascii_io.h"
#include "../test/testutil.h"

using namespace std;
using namespace std::tr1;
using boost::bad_lexical_cast;

class TestStringToNumber : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestStringToNumber);
    CPPUNIT_TEST(testInt8);
    CPPUNIT_TEST(testUint8);
    CPPUNIT_TEST(testInt16);
    CPPUNIT_TEST(testUint16);
    CPPUNIT_TEST(testInt32);
    CPPUNIT_TEST(testUint32);
    CPPUNIT_TEST(testInt64);
    CPPUNIT_TEST(testFloat);
    CPPUNIT_TEST(testDouble);
    CPPUNIT_TEST_SUITE_END();
public:
    void testInt8();
    void testUint8();
    void testInt16();
    void testUint16();
    void testInt32();
    void testUint32();
    void testInt64();
    void testFloat();
    void testDouble();
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestStringToNumber, TestSet::perBuild());

void TestStringToNumber::testInt8()
{
    CPPUNIT_ASSERT_EQUAL(int8_t(0), stringToNumber<int8_t>("0"));
    CPPUNIT_ASSERT_EQUAL(int8_t(-128), stringToNumber<int8_t>("-128"));
    CPPUNIT_ASSERT_EQUAL(int8_t(127), stringToNumber<int8_t>("127"));
    CPPUNIT_ASSERT_EQUAL(int8_t(127), stringToNumber<int8_t>("0127"));
    CPPUNIT_ASSERT_THROW(stringToNumber<int8_t>(""), bad_lexical_cast);
    CPPUNIT_ASSERT_THROW(stringToNumber<int8_t>("128"), bad_lexical_cast);
    CPPUNIT_ASSERT_THROW(stringToNumber<int8_t>("-129"), bad_lexical_cast);
    CPPUNIT_ASSERT_THROW(stringToNumber<int8_t>("0x1"), bad_lexical_cast);
    CPPUNIT_ASSERT_THROW(stringToNumber<int8_t>("100a"), bad_lexical_cast);
    CPPUNIT_ASSERT_THROW(stringToNumber<int8_t>("18446744073709551616"), bad_lexical_cast);
}

void TestStringToNumber::testUint8()
{
    CPPUNIT_ASSERT_EQUAL(uint8_t(0), stringToNumber<uint8_t>("0"));
    CPPUNIT_ASSERT_EQUAL(uint8_t(255), stringToNumber<uint8_t>("255"));
    CPPUNIT_ASSERT_EQUAL(uint8_t(100), stringToNumber<uint8_t>("0100"));
    CPPUNIT_ASSERT_THROW(stringToNumber<uint8_t>(""), bad_lexical_cast);
    CPPUNIT_ASSERT_THROW(stringToNumber<uint8_t>("256"), bad_lexical_cast);
    CPPUNIT_ASSERT_THROW(stringToNumber<uint8_t>("-1"), bad_lexical_cast);
    CPPUNIT_ASSERT_THROW(stringToNumber<uint8_t>("0x1"), bad_lexical_cast);
    CPPUNIT_ASSERT_THROW(stringToNumber<uint8_t>("100a"), bad_lexical_cast);
    CPPUNIT_ASSERT_THROW(stringToNumber<uint8_t>("18446744073709551616"), bad_lexical_cast);
}

void TestStringToNumber::testInt16()
{
    CPPUNIT_ASSERT_EQUAL(int16_t(0), stringToNumber<int16_t>("0"));
    CPPUNIT_ASSERT_EQUAL(int16_t(-32768), stringToNumber<int16_t>("-32768"));
    CPPUNIT_ASSERT_EQUAL(int16_t(32767), stringToNumber<int16_t>("32767"));
    CPPUNIT_ASSERT_EQUAL(int16_t(10000), stringToNumber<int16_t>("010000"));
    CPPUNIT_ASSERT_THROW(stringToNumber<int16_t>(""), bad_lexical_cast);
    CPPUNIT_ASSERT_THROW(stringToNumber<int16_t>("32768"), bad_lexical_cast);
    CPPUNIT_ASSERT_THROW(stringToNumber<int16_t>("-32769"), bad_lexical_cast);
    CPPUNIT_ASSERT_THROW(stringToNumber<int16_t>("0x1"), bad_lexical_cast);
    CPPUNIT_ASSERT_THROW(stringToNumber<int16_t>("100a"), bad_lexical_cast);
    CPPUNIT_ASSERT_THROW(stringToNumber<int16_t>("18446744073709551616"), bad_lexical_cast);
}

void TestStringToNumber::testUint16()
{
    CPPUNIT_ASSERT_EQUAL(uint16_t(0), stringToNumber<uint16_t>("0"));
    CPPUNIT_ASSERT_EQUAL(uint16_t(65535), stringToNumber<uint16_t>("65535"));
    CPPUNIT_ASSERT_EQUAL(uint16_t(10000), stringToNumber<uint16_t>("010000"));
    CPPUNIT_ASSERT_THROW(stringToNumber<uint16_t>(""), bad_lexical_cast);
    CPPUNIT_ASSERT_THROW(stringToNumber<uint16_t>("65536"), bad_lexical_cast);
    CPPUNIT_ASSERT_THROW(stringToNumber<uint16_t>("-1"), bad_lexical_cast);
    CPPUNIT_ASSERT_THROW(stringToNumber<uint16_t>("0x1"), bad_lexical_cast);
    CPPUNIT_ASSERT_THROW(stringToNumber<uint16_t>("100a"), bad_lexical_cast);
    CPPUNIT_ASSERT_THROW(stringToNumber<uint16_t>("18446744073709551616"), bad_lexical_cast);
}

void TestStringToNumber::testInt32()
{
    CPPUNIT_ASSERT_EQUAL(int32_t(0), stringToNumber<int32_t>("0"));
    CPPUNIT_ASSERT_EQUAL(int32_t(-2147483647 - 1), stringToNumber<int32_t>("-2147483648"));
    CPPUNIT_ASSERT_EQUAL(int32_t(2147483647), stringToNumber<int32_t>("2147483647"));
    CPPUNIT_ASSERT_EQUAL(int32_t(1000000000), stringToNumber<int32_t>("01000000000"));
    CPPUNIT_ASSERT_THROW(stringToNumber<int32_t>(""), bad_lexical_cast);
    CPPUNIT_ASSERT_THROW(stringToNumber<int32_t>("2147483648"), bad_lexical_cast);
    CPPUNIT_ASSERT_THROW(stringToNumber<int32_t>("-2147483649"), bad_lexical_cast);
    CPPUNIT_ASSERT_THROW(stringToNumber<int32_t>("0x1"), bad_lexical_cast);
    CPPUNIT_ASSERT_THROW(stringToNumber<int32_t>("100a"), bad_lexical_cast);
    CPPUNIT_ASSERT_THROW(stringToNumber<int32_t>("18446744073709551616"), bad_lexical_cast);
}

void TestStringToNumber::testUint32()
{
    CPPUNIT_ASSERT_EQUAL(uint32_t(0), stringToNumber<uint32_t>("0"));
    CPPUNIT_ASSERT_EQUAL(uint32_t(4294967295U), stringToNumber<uint32_t>("4294967295"));
    CPPUNIT_ASSERT_EQUAL(uint32_t(1000000000), stringToNumber<uint32_t>("01000000000"));
    CPPUNIT_ASSERT_THROW(stringToNumber<uint32_t>(""), bad_lexical_cast);
    CPPUNIT_ASSERT_THROW(stringToNumber<uint32_t>("4294967296"), bad_lexical_cast);
    CPPUNIT_ASSERT_THROW(stringToNumber<uint32_t>("-1"), bad_lexical_cast);
    CPPUNIT_ASSERT_THROW(stringToNumber<uint32_t>("0x1"), bad_lexical_cast);
    CPPUNIT_ASSERT_THROW(stringToNumber<uint32_t>("100a"), bad_lexical_cast);
    CPPUNIT_ASSERT_THROW(stringToNumber<uint32_t>("18446744073709551616"), bad_lexical_cast);
}

void TestStringToNumber::testInt64()
{
    CPPUNIT_ASSERT_EQUAL(int64_t(0), stringToNumber<int64_t>("0"));
    CPPUNIT_ASSERT_EQUAL(int64_t(INT64_C(-9223372036854775807) - 1), stringToNumber<int64_t>("-9223372036854775808"));
    CPPUNIT_ASSERT_EQUAL(int64_t(INT64_C(9223372036854775807)), stringToNumber<int64_t>("9223372036854775807"));
    CPPUNIT_ASSERT_EQUAL(int64_t(INT64_C(1000000000000000000)), stringToNumber<int64_t>("01000000000000000000"));
    CPPUNIT_ASSERT_THROW(stringToNumber<int64_t>(""), bad_lexical_cast);
    CPPUNIT_ASSERT_THROW(stringToNumber<int64_t>("9223372036854775808"), bad_lexical_cast);
    CPPUNIT_ASSERT_THROW(stringToNumber<int64_t>("-9223372036854775809"), bad_lexical_cast);
    CPPUNIT_ASSERT_THROW(stringToNumber<int64_t>("0x1"), bad_lexical_cast);
    CPPUNIT_ASSERT_THROW(stringToNumber<int64_t>("100a"), bad_lexical_cast);
    CPPUNIT_ASSERT_THROW(stringToNumber<int64_t>("18446744073709551616"), bad_lexical_cast);
}

void TestStringToNumber::testFloat()
{
    CPPUNIT_ASSERT_EQUAL(0.0f, stringToNumber<float>("0"));
    CPPUNIT_ASSERT_EQUAL(0.0f, stringToNumber<float>("0."));
    CPPUNIT_ASSERT_EQUAL(0.0f, stringToNumber<float>(".0"));
    CPPUNIT_ASSERT_EQUAL(0.0f, stringToNumber<float>("0.0"));
    CPPUNIT_ASSERT_EQUAL(0.0f, stringToNumber<float>("0.0e0"));
    CPPUNIT_ASSERT_EQUAL(1.25f, stringToNumber<float>("1.25"));
    CPPUNIT_ASSERT_EQUAL(1.234e03f, stringToNumber<float>("+1.234e03"));
    CPPUNIT_ASSERT_THROW(stringToNumber<float>("."), bad_lexical_cast);
    CPPUNIT_ASSERT_THROW(stringToNumber<float>("0x14"), bad_lexical_cast);
    CPPUNIT_ASSERT_THROW(stringToNumber<float>("inf"), bad_lexical_cast);
    CPPUNIT_ASSERT_THROW(stringToNumber<float>("INF"), bad_lexical_cast);
    CPPUNIT_ASSERT_THROW(stringToNumber<float>("INFINITY"), bad_lexical_cast);
    CPPUNIT_ASSERT_THROW(stringToNumber<float>("nan"), bad_lexical_cast);
    CPPUNIT_ASSERT_THROW(stringToNumber<float>("NaN"), bad_lexical_cast);
    CPPUNIT_ASSERT_THROW(stringToNumber<float>("NAN(0)"), bad_lexical_cast);
    if (std::numeric_limits<float>::is_iec559)
    {
        CPPUNIT_ASSERT_EQUAL(-1.17549435e-38f, stringToNumber<float>("-1.17549435e-38"));
    }
}

void TestStringToNumber::testDouble()
{
    CPPUNIT_ASSERT_EQUAL(0.0, stringToNumber<double>("0"));
    CPPUNIT_ASSERT_EQUAL(0.0, stringToNumber<double>("0."));
    CPPUNIT_ASSERT_EQUAL(0.0, stringToNumber<double>(".0"));
    CPPUNIT_ASSERT_EQUAL(0.0, stringToNumber<double>("0.0"));
    CPPUNIT_ASSERT_EQUAL(0.0, stringToNumber<double>("0.0e0"));
    CPPUNIT_ASSERT_EQUAL(1.25, stringToNumber<double>("1.25"));
    CPPUNIT_ASSERT_EQUAL(1.234e03, stringToNumber<double>("+1.234e03"));
    CPPUNIT_ASSERT_THROW(stringToNumber<double>("."), bad_lexical_cast);
    CPPUNIT_ASSERT_THROW(stringToNumber<double>("0x14"), bad_lexical_cast);
    CPPUNIT_ASSERT_THROW(stringToNumber<double>("inf"), bad_lexical_cast);
    CPPUNIT_ASSERT_THROW(stringToNumber<double>("INF"), bad_lexical_cast);
    CPPUNIT_ASSERT_THROW(stringToNumber<double>("INFINITY"), bad_lexical_cast);
    CPPUNIT_ASSERT_THROW(stringToNumber<double>("nan"), bad_lexical_cast);
    CPPUNIT_ASSERT_THROW(stringToNumber<double>("NaN"), bad_lexical_cast);
    CPPUNIT_ASSERT_THROW(stringToNumber<double>("NAN(0)"), bad_lexical_cast);
    if (std::numeric_limits<double>::is_iec559)
    {
        CPPUNIT_ASSERT_EQUAL(-2.2250738585072014e-308, stringToNumber<double>("-2.2250738585072014e-308"));
    }
}

class TestNumberToString : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestNumberToString);
    CPPUNIT_TEST(testInt8);
    CPPUNIT_TEST(testUint8);
    CPPUNIT_TEST(testInt16);
    CPPUNIT_TEST(testUint16);
    CPPUNIT_TEST(testInt32);
    CPPUNIT_TEST(testUint32);
    CPPUNIT_TEST(testInt64);
    CPPUNIT_TEST(testUint64);
    CPPUNIT_TEST(testFloat);
    CPPUNIT_TEST(testDouble);
    CPPUNIT_TEST_SUITE_END();
public:
    void testInt8();
    void testUint8();
    void testInt16();
    void testUint16();
    void testInt32();
    void testUint32();
    void testInt64();
    void testUint64();
    void testFloat();
    void testDouble();
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestNumberToString, TestSet::perBuild());

void TestNumberToString::testInt8()
{
    CPPUNIT_ASSERT_EQUAL(string("0"), numberToString(int8_t(0)));
    CPPUNIT_ASSERT_EQUAL(string("127"), numberToString(int8_t(127)));
    CPPUNIT_ASSERT_EQUAL(string("-128"), numberToString(int8_t(-128)));
}

void TestNumberToString::testUint8()
{
    CPPUNIT_ASSERT_EQUAL(string("0"), numberToString(uint8_t(0)));
    CPPUNIT_ASSERT_EQUAL(string("255"), numberToString(uint8_t(255)));
}

void TestNumberToString::testInt16()
{
    CPPUNIT_ASSERT_EQUAL(string("0"), numberToString(int16_t(0)));
    CPPUNIT_ASSERT_EQUAL(string("32767"), numberToString(int16_t(32767)));
    CPPUNIT_ASSERT_EQUAL(string("-32768"), numberToString(int16_t(-32768)));
}

void TestNumberToString::testUint16()
{
    CPPUNIT_ASSERT_EQUAL(string("0"), numberToString(uint16_t(0)));
    CPPUNIT_ASSERT_EQUAL(string("65535"), numberToString(uint16_t(65535)));
}

void TestNumberToString::testInt32()
{
    CPPUNIT_ASSERT_EQUAL(string("0"), numberToString(int32_t(0)));
    CPPUNIT_ASSERT_EQUAL(string("2147483647"), numberToString(int32_t(2147483647)));
    CPPUNIT_ASSERT_EQUAL(string("-2147483648"), numberToString(int32_t(-2147483647 - 1)));
}

void TestNumberToString::testUint32()
{
    CPPUNIT_ASSERT_EQUAL(string("0"), numberToString(uint32_t(0)));
    CPPUNIT_ASSERT_EQUAL(string("4294967295"), numberToString(uint32_t(4294967295U)));
}

void TestNumberToString::testInt64()
{
    CPPUNIT_ASSERT_EQUAL(string("0"), numberToString(int64_t(0)));
    CPPUNIT_ASSERT_EQUAL(string("9223372036854775807"), numberToString(int64_t(INT64_C(9223372036854775807))));
    CPPUNIT_ASSERT_EQUAL(string("-9223372036854775808"), numberToString(int64_t(INT64_C(-9223372036854775807) - 1)));
}

void TestNumberToString::testUint64()
{
    CPPUNIT_ASSERT_EQUAL(string("0"), numberToString(uint64_t(0)));
    CPPUNIT_ASSERT_EQUAL(string("18446744073709551615"), numberToString(uint64_t(UINT64_C(18446744073709551615))));
}

void TestNumberToString::testFloat()
{
    float v;

    v = 1.2f;
    CPPUNIT_ASSERT_EQUAL(v, stringToNumber<float>(numberToString(v)));
    v = std::numeric_limits<float>::min();
    CPPUNIT_ASSERT_EQUAL(v, stringToNumber<float>(numberToString(v)));
    v = std::numeric_limits<float>::max();
    CPPUNIT_ASSERT_EQUAL(v, stringToNumber<float>(numberToString(v)));
    v = std::numeric_limits<float>::epsilon();
    CPPUNIT_ASSERT_EQUAL(v, stringToNumber<float>(numberToString(v)));

    CPPUNIT_ASSERT_THROW(numberToString(std::numeric_limits<float>::infinity()), boost::bad_lexical_cast);
    CPPUNIT_ASSERT_THROW(numberToString(std::numeric_limits<float>::quiet_NaN()), boost::bad_lexical_cast);
}

void TestNumberToString::testDouble()
{
    double v;

    v = 1.2;
    CPPUNIT_ASSERT_EQUAL(v, stringToNumber<double>(numberToString(v)));
    v = std::numeric_limits<double>::min();
    CPPUNIT_ASSERT_EQUAL(v, stringToNumber<double>(numberToString(v)));
    v = std::numeric_limits<double>::max();
    CPPUNIT_ASSERT_EQUAL(v, stringToNumber<double>(numberToString(v)));
    v = std::numeric_limits<double>::epsilon();
    CPPUNIT_ASSERT_EQUAL(v, stringToNumber<double>(numberToString(v)));

    CPPUNIT_ASSERT_THROW(numberToString(std::numeric_limits<double>::infinity()), boost::bad_lexical_cast);
    CPPUNIT_ASSERT_THROW(numberToString(std::numeric_limits<double>::quiet_NaN()), boost::bad_lexical_cast);
}
