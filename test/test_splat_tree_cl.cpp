/**
 * @file
 *
 * Test code for @ref SplatTreeCL.
 */

#ifndef __CL_ENABLE_EXCEPTIONS
# define __CL_ENABLE_EXCEPTIONS
#endif

#if HAVE_CONFIG_H
# include <config.h>
#endif

#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/extensions/HelperMacros.h>
#include "testmain.h"
#include "test_clh.h"
#include "../src/splat_tree_cl.h"

using namespace std;

class TestSplatTreeCL : public CLH::Test::TestFixture
{
    CPPUNIT_TEST_SUITE(TestSplatTreeCL);
    CPPUNIT_TEST(testLevelShift);
    CPPUNIT_TEST(testMakeCode);
    CPPUNIT_TEST_SUITE_END();
private:
    cl::Program program;

    int callLevelShift(cl_int ilox, cl_int iloy, cl_int iloz, cl_int ihix, cl_int ihiy, cl_int ihiz);
    int callMakeCode(cl_int x, cl_int y, cl_int z, cl_int level);

    void testLevelShift();  ///< Tests @ref levelShift in @ref octree.cl.
    void testMakeCode();    ///< Tests @ref makeCode in @ref octree.cl.
public:
    TestSplatTreeCL();
    virtual void setUp();
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestSplatTreeCL, TestSet::perBuild());

TestSplatTreeCL::TestSplatTreeCL() : program() {}

void TestSplatTreeCL::setUp()
{
    CLH::Test::TestFixture::setUp();
    map<string, string> defines;
    defines["UNIT_TESTS"] = "1";
    program = CLH::build(context, "kernels/octree.cl", defines);
}

int TestSplatTreeCL::callLevelShift(cl_int ilox, cl_int iloy, cl_int iloz, cl_int ihix, cl_int ihiy, cl_int ihiz)
{
    cl_int ans;
    cl::Buffer out(context, CL_MEM_WRITE_ONLY, sizeof(cl_uint));
    cl::Kernel kernel(program, "testLevelShift");
    cl_int3 ilo, ihi;
    ilo.x = ilox; ilo.y = iloy; ilo.z = iloz;
    ihi.x = ihix; ihi.y = ihiy; ihi.z = ihiz;
    kernel.setArg(0, out);
    kernel.setArg(1, ilo);
    kernel.setArg(2, ihi);
    queue.enqueueTask(kernel);
    queue.enqueueReadBuffer(out, CL_TRUE, 0, sizeof(cl_int), &ans);
    return ans;
}

int TestSplatTreeCL::callMakeCode(cl_int x, cl_int y, cl_int z, cl_int level)
{
    cl_uint ans;
    cl::Buffer out(context, CL_MEM_WRITE_ONLY, sizeof(cl_uint));
    cl::Kernel kernel(program, "testMakeCode");
    cl_int3 xyz;
    xyz.s0 = x; xyz.s1 = y; xyz.s2 = z;
    kernel.setArg(0, out);
    kernel.setArg(1, xyz);
    kernel.setArg(2, level);
    queue.enqueueTask(kernel);
    queue.enqueueReadBuffer(out, CL_TRUE, 0, sizeof(cl_uint), &ans);
    return ans;
}

void TestSplatTreeCL::testLevelShift()
{
    CPPUNIT_ASSERT_EQUAL(0, callLevelShift(0, 0, 0,  0, 0, 0)); // single cell
    CPPUNIT_ASSERT_EQUAL(0, callLevelShift(1, 1, 1,  0, 0, 0)); // empty
    CPPUNIT_ASSERT_EQUAL(0, callLevelShift(0, 1, 2,  1, 2, 3)); // 2x2x2
    CPPUNIT_ASSERT_EQUAL(1, callLevelShift(0, 1, 2,  2, 2, 3)); // 3x2x2
    CPPUNIT_ASSERT_EQUAL(1, callLevelShift(0, 1, 2,  1, 3, 3)); // 2x3x2
    CPPUNIT_ASSERT_EQUAL(1, callLevelShift(0, 1, 2,  1, 2, 4)); // 2x2x3
    CPPUNIT_ASSERT_EQUAL(3, callLevelShift(31, 0, 0, 36, 0, 0)); // 011111 -> 100100
    CPPUNIT_ASSERT_EQUAL(3, callLevelShift(27, 0, 0, 32, 0, 0)); // 011011 -> 100000
    CPPUNIT_ASSERT_EQUAL(4, callLevelShift(48, 0, 0, 79, 0, 0)); // 0110000 -> 1001111
}

void TestSplatTreeCL::testMakeCode()
{
    CPPUNIT_ASSERT_EQUAL(0, callMakeCode(0, 0, 0, 0));
    CPPUNIT_ASSERT_EQUAL(2, callMakeCode(0, 0, 0, 1));
    CPPUNIT_ASSERT_EQUAL(2 + 7, callMakeCode(1, 1, 1, 1));
    CPPUNIT_ASSERT_EQUAL(128, callMakeCode(0, 0, 0, 3));
    CPPUNIT_ASSERT_EQUAL(128 + 174, callMakeCode(2, 5, 3, 3));
    CPPUNIT_ASSERT_EQUAL(128 + 511, callMakeCode(7, 7, 7, 3));
    CPPUNIT_ASSERT_EQUAL(8192, callMakeCode(0, 0, 0, 5));
}
