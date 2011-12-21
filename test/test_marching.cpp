/**
 * @file
 *
 * Tests for @ref Marching.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif

#ifndef __CL_ENABLE_EXCEPTIONS
# define __CL_ENABLE_EXCEPTIONS
#endif

#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/extensions/HelperMacros.h>
#include <cstddef>
#include <vector>
#include <CL/cl.hpp>
#include "testmain.h"
#include "test_clh.h"
#include "../src/marching.h"

using namespace std;

class TestMarching : public CLH::Test::TestFixture
{
    CPPUNIT_TEST_SUITE(TestMarching);
    CPPUNIT_TEST(testConstructor);
    CPPUNIT_TEST_SUITE_END();

private:
    /// Read the contents of a buffer and return it (synchronously) as a vector.
    template<typename T>
    vector<T> bufferToVector(const cl::Buffer &buffer);

    void testConstructor();    ///< Basic sanity tests on the tables
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestMarching, TestSet::perBuild());

template<typename T>
vector<T> TestMarching::bufferToVector(const cl::Buffer &buffer)
{
    size_t size = buffer.getInfo<CL_MEM_SIZE>();
    CPPUNIT_ASSERT(size % sizeof(T) == 0);
    vector<T> ans(size / sizeof(T));
    queue.enqueueReadBuffer(buffer, CL_TRUE, 0, size, &ans[0]);
    return ans;
}

void TestMarching::testConstructor()
{
    Marching marching(context, device, 2, 2, 2);

    vector<cl_uchar2> countTable = bufferToVector<cl_uchar2>(marching.countTable);
    vector<cl_ushort2> startTable = bufferToVector<cl_ushort2>(marching.startTable);
    vector<cl_uchar> dataTable = bufferToVector<cl_uchar>(marching.dataTable);

    CPPUNIT_ASSERT_EQUAL(256, int(countTable.size()));
    CPPUNIT_ASSERT_EQUAL(257, int(startTable.size()));
    CPPUNIT_ASSERT_EQUAL(int(startTable.back().s1), int(dataTable.size()));
    CPPUNIT_ASSERT_EQUAL(0, int(startTable.front().s0));
    for (unsigned int i = 0; i < 256; i++)
    {
        int sv = startTable[i].s0;
        int si = startTable[i].s1;
        int ev = startTable[i + 1].s0;
        int ei = startTable[i + 1].s1;
        CPPUNIT_ASSERT_EQUAL(int(countTable[i].s0), ev - sv);
        CPPUNIT_ASSERT_EQUAL(int(countTable[i].s1), ei - si);
        CPPUNIT_ASSERT(countTable[i].s1 % 3 == 0);
        for (int j = sv; j < ev; j++)
        {
            if (j > sv)
                CPPUNIT_ASSERT(dataTable[j - 1] < dataTable[j]);
            CPPUNIT_ASSERT(dataTable[j] < 19);
        }
        for (int j = si; j < ei; j++)
        {
            CPPUNIT_ASSERT(dataTable[j] < ev - sv);
        }
    }
}
