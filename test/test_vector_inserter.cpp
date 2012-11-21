/**
 * @file
 *
 * Tests for @ref VectorInserter.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif

#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/extensions/HelperMacros.h>
#include <stxxl.h>
#include "testutil.h"
#include "../src/vector_inserter.h"

class TestVectorInserter : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestVectorInserter);
    CPPUNIT_TEST(testPushBackSmall);
    CPPUNIT_TEST(testPushBackBig);
    CPPUNIT_TEST_SUITE_END();

private:
    void testPushBack(int size);  ///< Helper function for testing @ref VectorInserter::push_back

    void testPushBackSmall();  ///< Put in less than a block
    void testPushBackBig();    ///< Put in many blocks
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestVectorInserter, TestSet::perBuild());

void TestVectorInserter::testPushBack(int size)
{
    const unsigned int blockSize = 128 * 1024;
    stxxl::VECTOR_GENERATOR<int, 4, 8, blockSize>::result v;
    VectorInserter<int, blockSize> a;

    for (int i = 0; i < size; i++)
        a.push_back(i);
    a.move(v);

    CPPUNIT_ASSERT_EQUAL(size, int(v.size()));
    for (int i = 0; i < size; i++)
        CPPUNIT_ASSERT_EQUAL(i, v[i]);
}

void TestVectorInserter::testPushBackSmall()
{
    testPushBack(10000);
}

void TestVectorInserter::testPushBackBig()
{
    testPushBack(16 * 1024 * 1024);
}
