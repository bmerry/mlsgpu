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
    CPPUNIT_TEST(testMakeCode);
    CPPUNIT_TEST_SUITE_END();
private:
    cl::Program program;

    cl_uint callMakeCode(cl_int x, cl_int y, cl_int z, cl_int level);

    void testMakeCode();  /// Tests @ref makeCode in @ref octree.cl.
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

cl_uint TestSplatTreeCL::callMakeCode(cl_int x, cl_int y, cl_int z, cl_int level)
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

void TestSplatTreeCL::testMakeCode()
{
    CPPUNIT_ASSERT_EQUAL(0U, callMakeCode(0, 0, 0, 0));
    CPPUNIT_ASSERT_EQUAL(2U, callMakeCode(0, 0, 0, 1));
    CPPUNIT_ASSERT_EQUAL(2U + 7, callMakeCode(1, 1, 1, 1));
    CPPUNIT_ASSERT_EQUAL(128U, callMakeCode(0, 0, 0, 3));
    CPPUNIT_ASSERT_EQUAL(128U + 174, callMakeCode(2, 5, 3, 3));
    CPPUNIT_ASSERT_EQUAL(128U + 511, callMakeCode(7, 7, 7, 3));
    CPPUNIT_ASSERT_EQUAL(8192U, callMakeCode(0, 0, 0, 5));
}
