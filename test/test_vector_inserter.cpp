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
#include <vector>
#include <list>
#include <boost/iterator/counting_iterator.hpp>
#include "testutil.h"
#include "../src/tr1_cstdint.h"
#include "../src/vector_inserter.h"

/**
 * Tests for @ref VectorInserter.
 */
class TestVectorInserter : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestVectorInserter);
    CPPUNIT_TEST(testPushBackSmall);
    CPPUNIT_TEST(testPushBackBig);
    CPPUNIT_TEST(testAppendRandomAccess);
    CPPUNIT_TEST(testAppendGeneral);
    CPPUNIT_TEST(testSize);
    CPPUNIT_TEST(testReuse);
    CPPUNIT_TEST_SUITE_END();

protected:
    /**
     * Helper function for testing @ref VectorInserter::push_back. Pushes @a size
     * 64-bit unsigned integers into an inserter, constructs the vector, and checks
     * that it has the correct content.
     */
    void testPushBack(std::tr1::uint64_t size);

    /**
     * Helper function for testing @ref VectorInserter::push_back. Pushes various
     * length ranges into the container, constructs the vector, and checks that
     * it has the correct content. The @a Container determines the iterator type
     * used.
     */
    template<typename Container>
    void testAppend();

private:
    void testPushBackSmall();       ///< Put in less than a block
    void testPushBackBig();         ///< Put in many blocks
    void testAppendRandomAccess();  ///< Test @ref VectorInserter::append with random-access iterator
    void testAppendGeneral();       ///< Test @ref VectorInserter::append with non-random-access iterator
    void testSize();                ///< Test @ref VectorInserter::size
    void testReuse();               ///< Test that the inserter is properly emptied and can be reused
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestVectorInserter, TestSet::perBuild());

/**
 * Tests for VectorInserter that take a long time.
 */
class TestVectorInserterStress : public TestVectorInserter
{
    CPPUNIT_TEST_SUITE(TestVectorInserterStress);
    CPPUNIT_TEST(testPushBackHuge);
    CPPUNIT_TEST_SUITE_END();

private:
    void testPushBackHuge();        ///< Test with >2^32 elements, to check for overflow
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestVectorInserterStress, TestSet::perNightly());

void TestVectorInserter::testPushBack(std::tr1::uint64_t size)
{
    const unsigned int blockSize = 128 * 1024;
    typedef stxxl::VECTOR_GENERATOR<std::tr1::uint64_t, 4, 8, blockSize>::result vector_type;
    vector_type v;
    VectorInserter<std::tr1::uint64_t, blockSize> a("test.inserter");

    for (std::tr1::uint64_t i = 0; i < size; i++)
        a.push_back(i);
    a.move(v);

    MLSGPU_ASSERT_EQUAL(size, v.size());
    /* We use the more complicate stream interface for checking this so that
     * the stress test will be more efficient.
     */
    stxxl::stream::streamify_traits<vector_type::const_iterator>::stream_type stream
        = stxxl::stream::streamify(v.cbegin(), v.cend());
    for (std::tr1::uint64_t i = 0; i < size; i++)
    {
        CPPUNIT_ASSERT_EQUAL(i, *stream);
        ++stream;
    }
}

void TestVectorInserter::testPushBackSmall()
{
    testPushBack(10000);
}

void TestVectorInserter::testPushBackBig()
{
    testPushBack(1024 * 1024);
    testPushBack(1024 * 1024 + 12345);
}

template<typename Container>
void TestVectorInserter::testAppend()
{
    const unsigned int blockSize = 128 * 1024;
    stxxl::VECTOR_GENERATOR<int, 4, 8, blockSize>::result v;
    VectorInserter<int, blockSize> a("test.inserter");

    // Specifies the size of the vector after each append
    static const int cuts[] = {0, 1000, blockSize, blockSize, blockSize + 10, 5 * blockSize - 5, 5 * blockSize + 5};
    int cur = 0; // next value to append
    for (std::size_t i = 0; i < sizeof(cuts) / sizeof(cuts[0]); i++)
    {
        Container c;
        while (cur < cuts[i])
            c.push_back(cur++);
        a.append(c.begin(), c.end());
    }
    a.move(v);

    MLSGPU_ASSERT_EQUAL(cur, v.size());
    for (int i = 0; i < cur; i++)
        CPPUNIT_ASSERT_EQUAL(i, *(v.cbegin() + i));
}

void TestVectorInserter::testAppendRandomAccess()
{
    testAppend<std::vector<int> >();
}

void TestVectorInserter::testAppendGeneral()
{
    testAppend<std::list<int> >();
}

void TestVectorInserter::testSize()
{
    const unsigned int blockSize = 128 * 1024;
    VectorInserter<int, blockSize> v("test.inserter");

    MLSGPU_ASSERT_EQUAL(0, v.size());
    v.push_back(3);
    MLSGPU_ASSERT_EQUAL(1, v.size());
    v.append(boost::counting_iterator<int>(0),
             boost::counting_iterator<int>(1000000));
    MLSGPU_ASSERT_EQUAL(1000001, v.size());
}

void TestVectorInserter::testReuse()
{
    const unsigned int blockSize = 128 * 1024;
    stxxl::VECTOR_GENERATOR<int, 4, 8, blockSize>::result v1, v2;
    VectorInserter<int, blockSize> a("test.inserter");

    // Push [0, blockSize) into v1, [100, 200) into v2
    a.append(boost::counting_iterator<int>(0),
             boost::counting_iterator<int>(blockSize));
    a.move(v1);
    a.append(boost::counting_iterator<int>(100),
             boost::counting_iterator<int>(200));
    a.move(v2);

    MLSGPU_ASSERT_EQUAL(blockSize, v1.size());
    for (int i = 0; i < (int) blockSize; i++)
        CPPUNIT_ASSERT_EQUAL(i, *(v1.cbegin() + i));
    MLSGPU_ASSERT_EQUAL(100, v2.size());
    for (int i = 0; i < 100; i++)
        CPPUNIT_ASSERT_EQUAL(i + 100, *(v2.cbegin() + i));
}

void TestVectorInserterStress::testPushBackHuge()
{
    testPushBack((std::tr1::uint64_t(1) << 32) + 100);
}
