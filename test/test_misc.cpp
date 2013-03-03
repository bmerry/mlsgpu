/**
 * @file
 *
 * Test code for @ref misc.h.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/extensions/HelperMacros.h>
#include <string>
#include "../src/tr1_cstdint.h"
#include "../src/misc.h"
#include "testutil.h"

using namespace std;

using std::tr1::uint16_t;
using std::tr1::int16_t;
using std::tr1::uint32_t;
using std::tr1::int32_t;
using std::tr1::uint64_t;
using std::tr1::int64_t;

class TestMulDiv : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestMulDiv);
    CPPUNIT_TEST(testBigA);
    CPPUNIT_TEST(testBigC);
#if DEBUG
    CPPUNIT_TEST(testExceptions);
#endif
    CPPUNIT_TEST_SUITE_END();
public:
    void testBigA();         ///< Test with a large value for A
    void testBigC();         ///< Test with a large value for C
    void testExceptions();   ///< Test exception tests
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestMulDiv, TestSet::perBuild());

void TestMulDiv::testBigA()
{
    uint32_t a = 0x98765432u;
    uint32_t b = 17;
    uint32_t c = 20;
    uint32_t expected = uint64_t(a) * uint64_t(b) / uint64_t(c);
    CPPUNIT_ASSERT_EQUAL(expected, mulDiv(a, b, c));
    CPPUNIT_ASSERT_EQUAL(a, mulDiv(a, c, c));
    CPPUNIT_ASSERT_EQUAL(uint32_t(0), mulDiv(a, uint32_t(0), c));
}

void TestMulDiv::testBigC()
{
    uint32_t a = 12345;
    uint32_t b = 23456;
    uint32_t c = 34567;
    uint32_t expected = uint64_t(a) * uint64_t(b) / uint64_t(c);
    CPPUNIT_ASSERT_EQUAL(expected, mulDiv(a, b, c));
    CPPUNIT_ASSERT_EQUAL(a, mulDiv(a, c, c));
    CPPUNIT_ASSERT_EQUAL(uint32_t(0), mulDiv(a, uint32_t(0), c));
}

void TestMulDiv::testExceptions()
{
    CPPUNIT_ASSERT_THROW(mulDiv(5, 6, 5), std::invalid_argument); // b > c
    CPPUNIT_ASSERT_THROW(mulDiv(5, -1, 5), std::invalid_argument); // b < 0
    CPPUNIT_ASSERT_THROW(mulDiv(5, 0, 0), std::invalid_argument); // c <= 0
    CPPUNIT_ASSERT_THROW(mulDiv(-1, 4, 6), std::invalid_argument); // a < 0
    CPPUNIT_ASSERT_THROW(mulDiv(int32_t(5), int32_t(3), int32_t(100000)), std::out_of_range);  // c too big
}


/// Tests for @ref mulSat.
class TestMulSat : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestMulSat);
    CPPUNIT_TEST(testOverflow);
    CPPUNIT_TEST(testNoOverflow);
#if DEBUG
    CPPUNIT_TEST(testExceptions);
#endif
    CPPUNIT_TEST(testZero);
    CPPUNIT_TEST_SUITE_END();
public:
    void testOverflow();      ///< Test saturation
    void testNoOverflow();    ///< Test cases where no saturation is needed
    void testExceptions();    ///< Test that preconditions are checked
    void testZero();          ///< Test with one or both values being zero
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestMulSat, TestSet::perBuild());

void TestMulSat::testOverflow()
{
    CPPUNIT_ASSERT_EQUAL(uint16_t(UINT16_MAX), mulSat<uint16_t>(1000, 1000));
    CPPUNIT_ASSERT_EQUAL(int16_t(INT16_MAX), mulSat<int16_t>(32767, 2));
    CPPUNIT_ASSERT_EQUAL(uint32_t(UINT32_MAX), mulSat<uint32_t>(65536, 65536));
    CPPUNIT_ASSERT_EQUAL(uint32_t(UINT32_MAX), mulSat<uint32_t>(2, UINT32_MAX / 2 + 1));
    CPPUNIT_ASSERT_EQUAL(int64_t(INT64_MAX), mulSat<int64_t>(INT64_MAX, INT64_MAX));
}

void TestMulSat::testNoOverflow()
{
    CPPUNIT_ASSERT_EQUAL(UINT64_C(10000000000000000000), mulSat(UINT64_C(10000000000), UINT64_C(1000000000)));
    CPPUNIT_ASSERT_EQUAL(UINT32_MAX, mulSat(UINT32_C(65536), UINT32_C(65537)));
}

void TestMulSat::testExceptions()
{
    CPPUNIT_ASSERT_THROW(mulSat(-1, 1), std::invalid_argument);
    CPPUNIT_ASSERT_THROW(mulSat(1, -1), std::invalid_argument);
}

void TestMulSat::testZero()
{
    CPPUNIT_ASSERT_EQUAL(0, mulSat(0, 0));
    CPPUNIT_ASSERT_EQUAL(0, mulSat(0, 1000000));
    CPPUNIT_ASSERT_EQUAL(0, mulSat(INT_MAX, 0));
}


/// Tests for @ref divUp.
class TestDivUp : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestDivUp);
    CPPUNIT_TEST(testSimple);
#if DEBUG
    CPPUNIT_TEST(testExceptions);
#endif
    CPPUNIT_TEST(testTypes);
    CPPUNIT_TEST_SUITE_END();
public:
    void testExceptions();         ///< Test pre-conditions
    void testSimple();             ///< Test normal operation
    void testTypes();              ///< Test case where arguments have different types
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestDivUp, TestSet::perBuild());

void TestDivUp::testSimple()
{
    CPPUNIT_ASSERT_EQUAL(0, divUp(0, 4));
    CPPUNIT_ASSERT_EQUAL(5, divUp(17, 4));
    CPPUNIT_ASSERT_EQUAL(4, divUp(16, 4));
    CPPUNIT_ASSERT_EQUAL(1, divUp(17, 19));
    CPPUNIT_ASSERT_EQUAL(100, divUp(1890, 19));
    CPPUNIT_ASSERT_EQUAL(INT_MAX, divUp(INT_MAX, 1));
    CPPUNIT_ASSERT_EQUAL(UINT64_C(1000000001), divUp(UINT64_C(10000000000000000001), UINT64_C(10000000000)));
    CPPUNIT_ASSERT_EQUAL(UINT64_C(10000000001), divUp(UINT64_C(10000000000000000001), UINT64_C(1000000000)));
}

void TestDivUp::testExceptions()
{
    CPPUNIT_ASSERT_THROW(divUp(4, 0), std::invalid_argument);
    CPPUNIT_ASSERT_THROW(divUp(-1, 4), std::invalid_argument);
    CPPUNIT_ASSERT_THROW(divUp(INT_MAX - 3, 5), std::out_of_range);
}

void TestDivUp::testTypes()
{
    CPPUNIT_ASSERT_EQUAL(UINT64_C(10000000001), divUp(UINT64_C(10000000000000000001), UINT32_C(1000000000)));
}


/// Tests for @ref divDown
class TestDivDown : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestDivDown);
    CPPUNIT_TEST(testSimple);
#if DEBUG
    CPPUNIT_TEST(testZero);
    CPPUNIT_TEST(testOverflow);
#endif
    CPPUNIT_TEST_SUITE_END();
public:
    void testSimple();           ///< Test normal use cases
    void testZero();             ///< Test exception handling on divide-by-zero
    void testOverflow();         ///< Test exception handling on @c INT_MIN
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestDivDown, TestSet::perBuild());

void TestDivDown::testSimple()
{
    // Positive cases
    CPPUNIT_ASSERT_EQUAL(2, divDown(20, 10));
    CPPUNIT_ASSERT_EQUAL(2, divDown(21, 10));
    CPPUNIT_ASSERT_EQUAL(2, divDown(29, 10));
    CPPUNIT_ASSERT_EQUAL(UINT32_MAX, divDown(UINT32_MAX, 1));
    CPPUNIT_ASSERT_EQUAL(UINT64_MAX / 2, divDown(UINT64_MAX, 2));
    CPPUNIT_ASSERT_EQUAL(UINT64_C(2), divDown(UINT64_MAX, UINT64_MAX / 2));

    // Negative cases
    CPPUNIT_ASSERT_EQUAL(-2, divDown(-20, 10));
    CPPUNIT_ASSERT_EQUAL(-3, divDown(-21, 10));
    CPPUNIT_ASSERT_EQUAL(-3, divDown(-29, 10));
    CPPUNIT_ASSERT_EQUAL(-1000000001, divDown(-2000000001, 2));
    CPPUNIT_ASSERT_EQUAL(-INT64_MAX, divDown(-INT64_MAX, 1));
}

void TestDivDown::testZero()
{
    CPPUNIT_ASSERT_THROW(divDown(0, 0), std::invalid_argument);
    CPPUNIT_ASSERT_THROW(divDown(100, 0), std::invalid_argument);
    CPPUNIT_ASSERT_THROW(divDown(-100, 0), std::invalid_argument);
}

void TestDivDown::testOverflow()
{
    // This test is only valid on two's complement machines
    if (INT32_MIN < -INT32_MAX)
    {
        CPPUNIT_ASSERT_THROW(divDown(INT32_MIN, 1), std::out_of_range);
        CPPUNIT_ASSERT_THROW(divDown(INT32_MIN, 1000000), std::out_of_range);
    }
}

/// Tests for @ref DownDivider
class TestDownDivider : public CppUnit::TestFixture
{
public:
    CPPUNIT_TEST_SUITE(TestDownDivider);
    CPPUNIT_TEST(testNormal);
#if DEBUG
    CPPUNIT_TEST(testDivZero);
#endif
    CPPUNIT_TEST_SUITE_END();

private:
    /// Test a single division
    void testHelper(std::tr1::int32_t x, std::tr1::uint32_t d);
    void testNormal();    ///< Test a number of test cases
    void testDivZero();   ///< Test constructor with zero dividend
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestDownDivider, TestSet::perBuild());

void TestDownDivider::testHelper(std::tr1::int32_t x, std::tr1::uint32_t d)
{
    DownDivider div(d);
    std::tr1::int64_t q = div(x);
    std::tr1::int64_t dl = d;
    CPPUNIT_ASSERT(q * dl <= x && (q + 1) * dl > x);
}

void TestDownDivider::testNormal()
{
    const std::tr1::uint32_t ds[] =
    {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000,
        32767, 32768, 32769,
        65535, 65536, 65537,
        2147483647U, 2147483648U, 2147483649U,
        3000000000U,
        4294967294U, 4294967295U
    };
    const std::tr1::int32_t xs[] =
    {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000,
        32767, 32768, 32769,
        65535, 65536, 65537,
        1073741823, 1073741824, 1073741825,
        2147483645, 2147483646, 2147483647
    };

    for (std::size_t i = 0; i < sizeof(ds) / sizeof(ds[0]); i++)
    {
        for (std::size_t j = 0; j < sizeof(xs) / sizeof(xs[0]); j++)
        {
            testHelper(xs[j], ds[i]);
            testHelper(-xs[j], ds[i]);
        }
        if (ds[i] != 1)
            testHelper(INT32_MIN, ds[i]); // overflows if d == 1
    }
}

void TestDownDivider::testDivZero()
{
    CPPUNIT_ASSERT_THROW(DownDivider(0), std::invalid_argument);
}

/// Tests for @ref floatToBits.
class TestFloatToBits : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestFloatToBits);
    CPPUNIT_TEST(testSimple);
    CPPUNIT_TEST_SUITE_END();
public:
    void testSimple();   ///< Some smoke tests
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestFloatToBits, TestSet::perBuild());

void TestFloatToBits::testSimple()
{
    CPPUNIT_ASSERT_EQUAL(UINT32_C(0x3F800000), floatToBits(1.0f));
    CPPUNIT_ASSERT_EQUAL(UINT32_C(0x00000000), floatToBits(0.0f));
}


/// Tests for temporary file creation
class TestTmpFile : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestTmpFile);
    CPPUNIT_TEST(testCreate);
#if DEBUG
    CPPUNIT_TEST(testBadPath);
#endif
    CPPUNIT_TEST_SUITE_END();

private:
    boost::filesystem::path removePath; ///< Path to remove in teardown

public:
    void testCreate();      ///< Test basic creation
    void testBadPath();     ///< Test exception handling when the path is wrong

    virtual void tearDown();
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestTmpFile, TestSet::perBuild());

void TestTmpFile::testCreate()
{
    std::string testString = "TestTmpFile\n"; // \n to ensure binary mode

    setTmpFileDir(boost::filesystem::path("."));
    boost::filesystem::ofstream out;
    boost::filesystem::path path;
    createTmpFile(removePath, out);
    CPPUNIT_ASSERT(!removePath.empty());
    CPPUNIT_ASSERT(out);
    out << "TestTmpFile\n"; // The \n is to ensure it is in binary mode
    out.close();

    boost::filesystem::ifstream in(removePath, std::ios::in | std::ios::binary);
    char buffer[64];
    in.read(buffer, sizeof(buffer));
    CPPUNIT_ASSERT_EQUAL(int(testString.size()), int(in.gcount()));
    std::string actualString(buffer, testString.size());
    CPPUNIT_ASSERT_EQUAL(testString, actualString);
    in.close();
}

void TestTmpFile::testBadPath()
{
    setTmpFileDir("//\\bad");
    boost::filesystem::ofstream out;
    CPPUNIT_ASSERT_THROW(createTmpFile(removePath, out), std::ios::failure);
}

void TestTmpFile::tearDown()
{
    setTmpFileDir(boost::filesystem::path());
    if (!removePath.empty())
    {
        boost::filesystem::remove(removePath);
    }
}
