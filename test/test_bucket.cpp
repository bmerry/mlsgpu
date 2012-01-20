/**
 * @file
 *
 * Test code for @ref bucket.cpp.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/extensions/HelperMacros.h>
#include "testmain.h"
#include "../src/bucket.h"

/**
 * Tests for SplatRange.
 */
class TestSplatRange : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestSplatRange);
    CPPUNIT_TEST(testConstructor);
    CPPUNIT_TEST(testAppendEmpty);
    CPPUNIT_TEST(testAppendOverflow);
    CPPUNIT_TEST(testAppendMiddle);
    CPPUNIT_TEST(testAppendEnd);
    CPPUNIT_TEST(testAppendGap);
    CPPUNIT_TEST(testAppendNewScan);
    CPPUNIT_TEST_SUITE_END();

public:
    void testConstructor();          ///< Test the constructors
    void testAppendEmpty();          ///< Appending to an empty range
    void testAppendOverflow();       ///< An append that would overflow the size
    void testAppendMiddle();         ///< Append to the middle of an existing range
    void testAppendEnd();            ///< Extend a range
    void testAppendGap();            ///< Append outside (and not adjacent) to an existing range
    void testAppendNewScan();        ///< Append with a different scan
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestSplatRange, TestSet::perBuild());

void TestSplatRange::testConstructor()
{
    SplatRange empty;
    SplatRange single(3, 6);

    CPPUNIT_ASSERT_EQUAL(SplatRange::size_type(0), empty.size);

    CPPUNIT_ASSERT_EQUAL(SplatRange::size_type(1), single.size);
    CPPUNIT_ASSERT_EQUAL(SplatRange::scan_type(3), single.scan);
    CPPUNIT_ASSERT_EQUAL(SplatRange::index_type(6), single.start);
}

void TestSplatRange::testAppendEmpty()
{
    SplatRange range;
    bool success;

    success = range.append(3, 6);
    CPPUNIT_ASSERT_EQUAL(true, success);
    CPPUNIT_ASSERT_EQUAL(SplatRange::size_type(1), range.size);
    CPPUNIT_ASSERT_EQUAL(SplatRange::scan_type(3), range.scan);
    CPPUNIT_ASSERT_EQUAL(SplatRange::index_type(6), range.start);
}

void TestSplatRange::testAppendOverflow()
{
    SplatRange range;
    range.scan = 3;
    range.start = 0x90000000U;
    range.size = 0xFFFFFFFFU;
    bool success = range.append(3, range.start + range.size);
    CPPUNIT_ASSERT_EQUAL(false, success);
    CPPUNIT_ASSERT_EQUAL(SplatRange::size_type(0xFFFFFFFFU), range.size);
    CPPUNIT_ASSERT_EQUAL(SplatRange::scan_type(3), range.scan);
    CPPUNIT_ASSERT_EQUAL(SplatRange::index_type(0x90000000U), range.start);
}

void TestSplatRange::testAppendMiddle()
{
    SplatRange range;
    range.scan = 4;
    range.start = UINT64_C(0x123456781234);
    range.size = 0x10000;
    bool success = range.append(4, UINT64_C(0x12345678FFFF));
    CPPUNIT_ASSERT_EQUAL(true, success);
    CPPUNIT_ASSERT_EQUAL(SplatRange::size_type(0x10000), range.size);
    CPPUNIT_ASSERT_EQUAL(SplatRange::scan_type(4), range.scan);
    CPPUNIT_ASSERT_EQUAL(SplatRange::index_type(UINT64_C(0x123456781234)), range.start);
}

void TestSplatRange::testAppendEnd()
{
    SplatRange range;
    range.scan = 4;
    range.start = UINT64_C(0x123456781234);
    range.size = 0x10000;
    bool success = range.append(4, range.start + range.size);
    CPPUNIT_ASSERT_EQUAL(true, success);
    CPPUNIT_ASSERT_EQUAL(SplatRange::size_type(0x10001), range.size);
    CPPUNIT_ASSERT_EQUAL(SplatRange::scan_type(4), range.scan);
    CPPUNIT_ASSERT_EQUAL(SplatRange::index_type(UINT64_C(0x123456781234)), range.start);
}

void TestSplatRange::testAppendGap()
{
    SplatRange range;
    range.scan = 4;
    range.start = UINT64_C(0x123456781234);
    range.size = 0x10000;
    bool success = range.append(4, range.start + range.size + 1);
    CPPUNIT_ASSERT_EQUAL(false, success);
    CPPUNIT_ASSERT_EQUAL(SplatRange::size_type(0x10000), range.size);
    CPPUNIT_ASSERT_EQUAL(SplatRange::scan_type(4), range.scan);
    CPPUNIT_ASSERT_EQUAL(SplatRange::index_type(UINT64_C(0x123456781234)), range.start);
}

void TestSplatRange::testAppendNewScan()
{
    SplatRange range;
    range.scan = 4;
    range.start = UINT64_C(0x123456781234);
    range.size = 0x10000;
    bool success = range.append(5, range.start + range.size);
    CPPUNIT_ASSERT_EQUAL(false, success);
    CPPUNIT_ASSERT_EQUAL(SplatRange::size_type(0x10000), range.size);
    CPPUNIT_ASSERT_EQUAL(SplatRange::scan_type(4), range.scan);
    CPPUNIT_ASSERT_EQUAL(SplatRange::index_type(UINT64_C(0x123456781234)), range.start);
}


class TestSplatRangeCounter : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestSplatRangeCounter);
    CPPUNIT_TEST(testEmpty);
    CPPUNIT_TEST(testSimple);
    CPPUNIT_TEST_SUITE_END();
public:
    void testEmpty();           ///< Tests initial state
    void testSimple();          ///< Tests state after various types of additions
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestSplatRangeCounter, TestSet::perBuild());

void TestSplatRangeCounter::testEmpty()
{
    SplatRangeCounter c;
    CPPUNIT_ASSERT_EQUAL(std::tr1::uint64_t(0), c.countRanges());
    CPPUNIT_ASSERT_EQUAL(std::tr1::uint64_t(0), c.countSplats());
}

void TestSplatRangeCounter::testSimple()
{
    SplatRangeCounter c;

    c.append(3, 5);
    c.append(3, 6);
    c.append(3, 6);
    c.append(4, 7);
    c.append(5, 2);
    c.append(5, 4);
    c.append(5, 5);
    CPPUNIT_ASSERT_EQUAL(std::tr1::uint64_t(4), c.countRanges());
    CPPUNIT_ASSERT_EQUAL(std::tr1::uint64_t(7), c.countSplats());
}
