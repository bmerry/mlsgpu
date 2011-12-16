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
#include <climits>
#include <cstddef>
#include <vector>
#include "testmain.h"
#include "test_clh.h"
#include "test_splat_tree.h"
#include "../src/splat_tree_cl.h"

using namespace std;

class TestSplatTreeCL : public TestSplatTree, public CLH::Test::Mixin
{
    CPPUNIT_TEST_SUB_SUITE(TestSplatTreeCL, TestSplatTree);
    CPPUNIT_TEST(testLevelShift);
    CPPUNIT_TEST(testPointBoxDist2);
    CPPUNIT_TEST(testMakeCode);
    CPPUNIT_TEST(testLowerBound);
    CPPUNIT_TEST_SUITE_END();

protected:
    virtual void build(
        std::size_t &numLevels,
        std::vector<SplatTree::command_type> &commands,
        std::vector<SplatTree::command_type> &start,
        const std::vector<Splat> &splats, const Grid &grid);

private:
    cl::Program program;

    int callLevelShift(cl_int ilox, cl_int iloy, cl_int iloz, cl_int ihix, cl_int ihiy, cl_int ihiz);
    float callPointBoxDist2(float px, float py, float pz, float lx, float ly, float lz, float hx, float hy, float hz);
    int callMakeCode(cl_int x, cl_int y, cl_int z);
    cl_uint callLowerBound(const cl::Buffer &keys, cl_uint keysLen, cl_uint key);

    void testLevelShift();     ///< Test @ref levelShift in @ref octree.cl.
    void testPointBoxDist2();  ///< Test @ref pointBoxDist2 in @ref octree.cl.
    void testMakeCode();       ///< Test @ref makeCode in @ref octree.cl.
    void testLowerBound();     ///< Test @ref lowerBound in @ref octree.cl.
public:
    virtual void setUp();
    virtual void tearDown();
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestSplatTreeCL, TestSet::perBuild());

void TestSplatTreeCL::setUp()
{
    TestSplatTree::setUp();
    setUpCL();
    map<string, string> defines;
    defines["UNIT_TESTS"] = "1";
    program = CLH::build(context, "kernels/octree.cl", defines);
}

void TestSplatTreeCL::tearDown()
{
    tearDownCL();
    TestSplatTree::tearDown();
}

void TestSplatTreeCL::build(
    std::size_t &numLevels,
    std::vector<SplatTree::command_type> &commands,
    std::vector<SplatTree::command_type> &start,
    const std::vector<Splat> &splats, const Grid &grid)
{
    SplatTreeCL tree(context, 9, 1001);
    std::vector<cl::Event> events(1);
    tree.enqueueBuild(queue, &splats[0], splats.size(), grid, CL_FALSE, NULL, NULL, &events[0]);

    std::size_t commandsSize = tree.getCommands().getInfo<CL_MEM_SIZE>();
    std::size_t startSize = tree.getStart().getInfo<CL_MEM_SIZE>();
    commands.resize(commandsSize / sizeof(SplatTree::command_type));
    start.resize(startSize / sizeof(SplatTree::command_type));
    queue.enqueueReadBuffer(tree.getCommands(), CL_TRUE, 0, commandsSize, &commands[0]);
    queue.enqueueReadBuffer(tree.getStart(), CL_TRUE, 0, startSize, &start[0]);
    numLevels = tree.getNumLevels();
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

float TestSplatTreeCL::callPointBoxDist2(float px, float py, float pz, float lx, float ly, float lz, float hx, float hy, float hz)
{
    cl_float ans;
    cl::Buffer out(context, CL_MEM_WRITE_ONLY, sizeof(cl_uint));
    cl::Kernel kernel(program, "testPointBoxDist2");
    cl_float3 pos, lo, hi;
    pos.x = px; pos.y = py; pos.z = pz;
    lo.x = lx; lo.y = ly; lo.z = lz;
    hi.x = hx; hi.y = hy; hi.z = hz;
    kernel.setArg(0, out);
    kernel.setArg(1, pos);
    kernel.setArg(2, lo);
    kernel.setArg(3, hi);
    queue.enqueueTask(kernel);
    queue.enqueueReadBuffer(out, CL_TRUE, 0, sizeof(cl_float), &ans);
    return ans;
}

int TestSplatTreeCL::callMakeCode(cl_int x, cl_int y, cl_int z)
{
    cl_uint ans;
    cl::Buffer out(context, CL_MEM_WRITE_ONLY, sizeof(cl_uint));
    cl::Kernel kernel(program, "testMakeCode");
    cl_int3 xyz;
    xyz.s0 = x; xyz.s1 = y; xyz.s2 = z;
    kernel.setArg(0, out);
    kernel.setArg(1, xyz);
    queue.enqueueTask(kernel);
    queue.enqueueReadBuffer(out, CL_TRUE, 0, sizeof(cl_uint), &ans);
    return ans;
}

cl_uint TestSplatTreeCL::callLowerBound(const cl::Buffer &keys, cl_uint keysLen, cl_uint key)
{
    cl_uint ans;
    cl::Buffer out(context, CL_MEM_WRITE_ONLY, sizeof(cl_uint));
    cl::Kernel kernel(program, "testLowerBound");
    kernel.setArg(0, out);
    kernel.setArg(1, keys);
    kernel.setArg(2, keysLen);
    kernel.setArg(3, key);
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

void TestSplatTreeCL::testPointBoxDist2()
{
    // Point inside the box
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0,
        callPointBoxDist2(0.5f, 0.5f, 0.5f,  0.0f, 0.0f, 0.0f,  1.0f, 1.0f, 1.0f), 1e-4);
    // Above one face
    CPPUNIT_ASSERT_DOUBLES_EQUAL(4.0,
        callPointBoxDist2(0.25f, 0.5f, 3.0f,  -1.5f, 0.0f, 0.5f,  1.5f, 0.75f, 1.0f), 1e-4);
    // Nearest point is a corner
    CPPUNIT_ASSERT_DOUBLES_EQUAL(14.0,
        callPointBoxDist2(9.0f, 11.f, -10.f,  -1.0f, 0.0f, -7.0f,  8.0f, 9.0f, 8.0f), 1e-4);
}

void TestSplatTreeCL::testMakeCode()
{
    CPPUNIT_ASSERT_EQUAL(0, callMakeCode(0, 0, 0));
    CPPUNIT_ASSERT_EQUAL(7, callMakeCode(1, 1, 1));
    CPPUNIT_ASSERT_EQUAL(174, callMakeCode(2, 5, 3));
    CPPUNIT_ASSERT_EQUAL(511, callMakeCode(7, 7, 7));
}

void TestSplatTreeCL::testLowerBound()
{
    cl_uint hKeys[] =
    {
        1, 1, 4, 7, 7, 7, 8, 9, 10, 10, 12, 12, 14
    };
    cl::Buffer dKeys(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(hKeys), hKeys);
    cl_uint keysLen = sizeof(hKeys) / sizeof(hKeys[0]);
    cl_uint ans;

    for (unsigned int i = 0; i < 16; i++)
    {
        ans = callLowerBound(dKeys, keysLen, i);
        CPPUNIT_ASSERT(ans <= keysLen);
        CPPUNIT_ASSERT(ans == keysLen || hKeys[ans] >= i);
        CPPUNIT_ASSERT(ans == 0 || hKeys[ans - 1] < i);
    }
}
