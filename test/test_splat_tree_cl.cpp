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
#include <cstddef>
#include <vector>
#include <cmath>
#include "testmain.h"
#include "test_clh.h"
#include "test_splat_tree.h"
#include "../src/splat_tree_cl.h"

using namespace std;

/// Tests for @ref SplatTreeCL
class TestSplatTreeCL : public TestSplatTree, public CLH::Test::Mixin
{
    CPPUNIT_TEST_SUB_SUITE(TestSplatTreeCL, TestSplatTree);
    CPPUNIT_TEST(testLevelShift);
    CPPUNIT_TEST(testPointBoxDist2);
    CPPUNIT_TEST(testMakeCode);
    CPPUNIT_TEST_SUITE_END();

protected:
    virtual void build(
        std::size_t &numLevels,
        std::vector<SplatTree::command_type> &commands,
        std::vector<SplatTree::command_type> &start,
        const std::vector<Splat> &splats,
        int maxLevels, int subsamplingShift, std::size_t maxSplats,
        const Grid::size_type size[3], const Grid::difference_type offset[3]);

private:
    cl::Program octreeProgram;  ///< Program compiled from @ref octree.cl.

    int callLevelShift(cl_int ilox, cl_int iloy, cl_int iloz, cl_int ihix, cl_int ihiy, cl_int ihiz);
    float callPointBoxDist2(float px, float py, float pz, float lx, float ly, float lz, float hx, float hy, float hz);
    int callMakeCode(cl_int x, cl_int y, cl_int z);

    void testLevelShift();     ///< Test @ref levelShift in @ref octree.cl.
    void testPointBoxDist2();  ///< Test @ref pointBoxDist2 in @ref octree.cl.
    void testMakeCode();       ///< Test @ref makeCode in @ref octree.cl.
public:
    virtual void setUp();
    virtual void tearDown();
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestSplatTreeCL, TestSet::perCommit());

void TestSplatTreeCL::setUp()
{
    TestSplatTree::setUp();
    setUpCL();
    map<string, string> defines;
    defines["UNIT_TESTS"] = "1";
    defines["WGS_X"] = "4";  // arbitrary values to make it compile
    defines["WGS_Y"] = "4";
    defines["WGS_Z"] = "4";
    octreeProgram = CLH::build(context, "kernels/octree.cl", defines);
}

void TestSplatTreeCL::tearDown()
{
    octreeProgram = NULL;
    tearDownCL();
    TestSplatTree::tearDown();
}

void TestSplatTreeCL::build(
    std::size_t &numLevels,
    std::vector<SplatTree::command_type> &commands,
    std::vector<SplatTree::command_type> &start,
    const std::vector<Splat> &splats,
    int maxLevels, int subsamplingShift, std::size_t maxSplats,
    const Grid::size_type size[3], const Grid::difference_type offset[3])
{
    SplatTreeCL tree(context, maxLevels, maxSplats);
    std::vector<cl::Event> events(1);
    cl::Buffer splatBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                           splats.size() * sizeof(Splat), (void *) &splats[0]);
    tree.enqueueBuild(queue, splatBuffer, splats.size(), size, offset, subsamplingShift, NULL, &events[0]);

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
    cl::Kernel kernel(octreeProgram, "testLevelShift");
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
    cl::Kernel kernel(octreeProgram, "testPointBoxDist2");
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
    cl::Kernel kernel(octreeProgram, "testMakeCode");
    cl_int3 xyz;
    xyz.s0 = x; xyz.s1 = y; xyz.s2 = z;
    kernel.setArg(0, out);
    kernel.setArg(1, xyz);
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
    CPPUNIT_ASSERT_EQUAL(2, callLevelShift(31, 0, 0, 35, 0, 0)); // 011111 -> 100001
    CPPUNIT_ASSERT_EQUAL(3, callLevelShift(31, 0, 0, 36, 0, 0)); // 011111 -> 100100
    CPPUNIT_ASSERT_EQUAL(3, callLevelShift(27, 0, 0, 32, 0, 0)); // 011011 -> 100000
    CPPUNIT_ASSERT_EQUAL(5, callLevelShift(48, 0, 0, 79, 0, 0)); // 0110000 -> 1001111
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
