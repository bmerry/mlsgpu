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
#include "../src/tr1_cstdint.h"
#include "testmain.h"
#include "../src/misc.h"

using namespace std;

using std::tr1::uint16_t;
using std::tr1::int16_t;
using std::tr1::uint32_t;
using std::tr1::int32_t;
using std::tr1::uint64_t;
using std::tr1::int64_t;

/// Tests for @ref mulSat.
class TestMulSat : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestMulSat);
    CPPUNIT_TEST(testOverflow);
    CPPUNIT_TEST(testNoOverflow);
    CPPUNIT_TEST(testExceptions);
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
    CPPUNIT_TEST(testExceptions);
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
    CPPUNIT_TEST(testZero);
    CPPUNIT_TEST(testOverflow);
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
