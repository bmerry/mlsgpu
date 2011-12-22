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
#include <cmath>
#include <limits>
#include <tr1/random>
#include <boost/math/constants/constants.hpp>
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
    CPPUNIT_TEST(testSolveQuadratic);
    CPPUNIT_TEST(testFitSphere);
    CPPUNIT_TEST(testProjectDistOrigin);
    CPPUNIT_TEST_SUITE_END();

protected:
    virtual void build(
        std::size_t &numLevels,
        std::vector<SplatTree::command_type> &commands,
        std::vector<SplatTree::command_type> &start,
        const std::vector<Splat> &splats, const Grid &grid);

private:
    cl::Program octreeProgram;  ///< Program compiled from @ref octree.cl.
    cl::Program mlsProgram;     ///< Program compiled from @ref mls.cl.

    /**
     * Variant of @c CppUnit::assertDoublesEqual that accepts relative or absolute error.
     *
     * - If @a expected or @a actual are both NaN, passes.
     * - If one of @a expected or @a actual is NaN, fails.
     * - If @a expected or @a actual is an infinity, fails unless they're equal.
     * - Otherwise, passes if |@a expected - @a actual| <= @a eps or
     *   if |@a expected - @a actual| <= @a eps * @a expected.
     *
     * @param expected     Expected value
     * @param actual       Actual value
     * @param eps          Error tolerance
     * @param sourceLine   Pass @c CPPUNIT_SOURCELINE
     *
     * @see @ref ASSERT_DOUBLES_EQUAL
     */
    static void assertDoublesRelEqual(double expected, double actual, double eps, const CppUnit::SourceLine &sourceLine);

    /**
     * Generate an algebraic sphere from a geometric one.
     * @param xc, yc, zc       Sphere center.
     * @param r                Sphere radius.
     * @param grad             Magnitude of gradient at surface (negative to invert the sphere).
     * @return A 5-element vector of algebraic sphere parameters.
     */
    std::vector<float> makeSphere(float xc, float yc, float zc, float r, float grad);

    /**
     * Generate an algebraic sphere representing a plane.
     * @param px, py, pz       A point on the plane.
     * @param dx, dy, dz       Normal to the plane (need not be unit length).
     * @return A 5-element vector of algebraic sphere parameters.
     */
    std::vector<float> makePlane(float px, float py, float pz, float dx, float dy, float dz);

    int callLevelShift(cl_int ilox, cl_int iloy, cl_int iloz, cl_int ihix, cl_int ihiy, cl_int ihiz);
    float callPointBoxDist2(float px, float py, float pz, float lx, float ly, float lz, float hx, float hy, float hz);
    int callMakeCode(cl_int x, cl_int y, cl_int z);
    float callSolveQuadratic(float a, float b, float c);
    float callProjectDistOrigin(float p0, float p1, float p2, float p3, float p4);
    float callProjectDistOrigin(const std::vector<float> &params);

    /**
     * Wrapper around @c TestFitSphere in @ref mls.cl.
     * @param splats     Two or more splats. The radius is ignored, and the weight should be
     *                   placed directly into the quality slot. The positions
     *                   are in the local coordinate system in which the sphere is fitted.
     * @return A 5-element vector of algebraic sphere parameters.
     */
    std::vector<float> callFitSphere(const std::vector<Splat> &splats);

    void testLevelShift();     ///< Test @ref levelShift in @ref octree.cl.
    void testPointBoxDist2();  ///< Test @ref pointBoxDist2 in @ref octree.cl.
    void testMakeCode();       ///< Test @ref makeCode in @ref octree.cl.
    void testSolveQuadratic(); ///< Test @ref solveQuadratic in @ref mls.cl.
    void testProjectDistOrigin();  ///< Test @ref projectDistOrigin in @ref mls.cl.
    void testFitSphere();      ///< Test @ref fitSphere in @ref mls.cl.
public:
    virtual void setUp();
    virtual void tearDown();
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestSplatTreeCL, TestSet::perCommit());

/**
 * Macro wrapper around @ref TestSplatTreeCL::assertDoublesRelEqual.
 */
#define ASSERT_DOUBLES_EQUAL(actual, expected, eps) \
    TestSplatTreeCL::assertDoublesRelEqual(actual, expected, eps, CPPUNIT_SOURCELINE())

void TestSplatTreeCL::assertDoublesRelEqual(double expected, double actual, double eps, const CppUnit::SourceLine &sourceLine)
{
    string expectedStr = boost::lexical_cast<string>(expected);
    string actualStr = boost::lexical_cast<string>(actual);
    if ((tr1::isnan)(expected) && (tr1::isnan)(actual))
        return;
    if ((tr1::isnan)(expected))
        CppUnit::Asserter::failNotEqual(expectedStr, actualStr, sourceLine, "expected is NaN");
    if ((tr1::isnan)(actual))
        CppUnit::Asserter::failNotEqual(expectedStr, actualStr, sourceLine, "actual is NaN");
    if (expected == actual)
        return;
    if ((tr1::isfinite)(expected) && (tr1::isfinite)(actual))
    {
        double err = abs(expected - actual);
        if (err > eps * abs(expected) && err > eps)
            CppUnit::Asserter::failNotEqual(expectedStr, actualStr, sourceLine,
                                            "Delta   : " + boost::lexical_cast<string>(err));
    }
    else
        CppUnit::Asserter::failNotEqual(expectedStr, actualStr, sourceLine, "");
}

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
    mlsProgram = CLH::build(context, "kernels/mls.cl", defines);
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
    tree.enqueueBuild(queue, &splats[0], splats.size(), grid, 0, CL_FALSE, NULL, NULL, &events[0]);

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

float TestSplatTreeCL::callSolveQuadratic(float a, float b, float c)
{
    cl_float ans;
    cl::Buffer out(context, CL_MEM_WRITE_ONLY, sizeof(cl_float));
    cl::Kernel kernel(mlsProgram, "testSolveQuadratic");
    kernel.setArg(0, out);
    kernel.setArg(1, a);
    kernel.setArg(2, b);
    kernel.setArg(3, c);
    queue.enqueueTask(kernel);
    queue.enqueueReadBuffer(out, CL_TRUE, 0, sizeof(cl_float), &ans);
    return ans;
}

float TestSplatTreeCL::callProjectDistOrigin(float p0, float p1, float p2, float p3, float p4)
{
    cl_float ans;
    cl::Buffer out(context, CL_MEM_WRITE_ONLY, sizeof(cl_float));
    cl::Kernel kernel(mlsProgram, "testProjectDistOrigin");
    kernel.setArg(0, out);
    kernel.setArg(1, p0);
    kernel.setArg(2, p1);
    kernel.setArg(3, p2);
    kernel.setArg(4, p3);
    kernel.setArg(5, p4);
    queue.enqueueTask(kernel);
    queue.enqueueReadBuffer(out, CL_TRUE, 0, sizeof(cl_float), &ans);
    return ans;
}

float TestSplatTreeCL::callProjectDistOrigin(const std::vector<float> &params)
{
    CPPUNIT_ASSERT_EQUAL(std::vector<float>::size_type(5), params.size());
    return callProjectDistOrigin(params[0], params[1], params[2], params[3], params[4]);
}

std::vector<float> TestSplatTreeCL::callFitSphere(const std::vector<Splat> &splats)
{
    std::vector<float> ans(5);
    cl::Buffer in(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(Splat) * splats.size(), (void *) &splats[0]);
    cl::Buffer out(context, CL_MEM_WRITE_ONLY, 5 * sizeof(cl_float));
    cl::Kernel kernel(mlsProgram, "testFitSphere");
    kernel.setArg(0, out);
    kernel.setArg(1, in);
    kernel.setArg(2, cl_uint(splats.size()));
    queue.enqueueTask(kernel);
    queue.enqueueReadBuffer(out, CL_TRUE, 0, 5 * sizeof(cl_float), &ans[0]);
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

void TestSplatTreeCL::testSolveQuadratic()
{
    float n = std::numeric_limits<float>::quiet_NaN();
    float eps = std::numeric_limits<float>::epsilon() * 4;

    // Cases with no roots
    ASSERT_DOUBLES_EQUAL(n, callSolveQuadratic(1, -2, 2), eps);
    ASSERT_DOUBLES_EQUAL(n, callSolveQuadratic(-1, 2, -2), eps);
    ASSERT_DOUBLES_EQUAL(n, callSolveQuadratic(1e20, -2e10, 1.0001), eps);
    ASSERT_DOUBLES_EQUAL(n, callSolveQuadratic(1, 0, 1), eps);
    ASSERT_DOUBLES_EQUAL(n, callSolveQuadratic(-1, 0, -1), eps);
    // Constant functions (no roots or infinitely many roots)
    ASSERT_DOUBLES_EQUAL(n, callSolveQuadratic(0, 0, 0), eps);
    ASSERT_DOUBLES_EQUAL(n, callSolveQuadratic(0, 0, 4), eps);
    ASSERT_DOUBLES_EQUAL(n, callSolveQuadratic(0, 0, -3), eps);
    ASSERT_DOUBLES_EQUAL(n, callSolveQuadratic(0, 0, -1e20), eps);
    ASSERT_DOUBLES_EQUAL(n, callSolveQuadratic(0, 0, 1e20), eps);
    // Linear functions
    ASSERT_DOUBLES_EQUAL(-1.5, callSolveQuadratic(0, 2, 3), eps);
    ASSERT_DOUBLES_EQUAL(2.5, callSolveQuadratic(0, -2, 5), eps);
    ASSERT_DOUBLES_EQUAL(0.0, callSolveQuadratic(0, 5, 0), eps);
    ASSERT_DOUBLES_EQUAL(0.0, callSolveQuadratic(0, 1e20, 0), eps);
    ASSERT_DOUBLES_EQUAL(0.0, callSolveQuadratic(0, 1e-20, 0), eps);
    ASSERT_DOUBLES_EQUAL(1e-20, callSolveQuadratic(0, 1e10, 1e-10), eps);
    ASSERT_DOUBLES_EQUAL(-1e20, callSolveQuadratic(0, 1e-10, 1e10), eps);
    // Repeated roots
    ASSERT_DOUBLES_EQUAL(1.0, callSolveQuadratic(1, -2, 1), eps);
    ASSERT_DOUBLES_EQUAL(1.0, callSolveQuadratic(10, -20, 10), eps);
    ASSERT_DOUBLES_EQUAL(1e4, callSolveQuadratic(1, -2e4, 1e8), eps);
    ASSERT_DOUBLES_EQUAL(0.0, callSolveQuadratic(1, 0, 0), eps);
    ASSERT_DOUBLES_EQUAL(0.0, callSolveQuadratic(1e30, 0, 0), eps);
    ASSERT_DOUBLES_EQUAL(0.0, callSolveQuadratic(1e-20, 0, 0), eps);
    // Regular two-root solutions
    ASSERT_DOUBLES_EQUAL(3.0, callSolveQuadratic(1, -5, 6), eps);
    ASSERT_DOUBLES_EQUAL(2.0, callSolveQuadratic(-2, 10, -12), eps);
    ASSERT_DOUBLES_EQUAL(2.0, callSolveQuadratic(1, 1, -6), eps);
    ASSERT_DOUBLES_EQUAL(-3.0, callSolveQuadratic(-0.1f, -0.1f, 0.6f), eps);
    ASSERT_DOUBLES_EQUAL(3.0, callSolveQuadratic(1e-12, -5e-12, 6e-12), eps);
    ASSERT_DOUBLES_EQUAL(-2e-12, callSolveQuadratic(1, 5e-12, 6e-24), eps);
    // Corner cases for stability
    ASSERT_DOUBLES_EQUAL(1.0, callSolveQuadratic(1, -1 - 1e-6, 1e-6), eps);
    ASSERT_DOUBLES_EQUAL(1e6, callSolveQuadratic(1, -1 - 1e6, 1e6), eps);
    ASSERT_DOUBLES_EQUAL(1e20, callSolveQuadratic(1e-20, -2, 1e20), eps);
    ASSERT_DOUBLES_EQUAL(-1e-6, callSolveQuadratic(1e-6, 1, 1e-6), eps);
}

std::vector<float> TestSplatTreeCL::makeSphere(float xc, float yc, float zc, float r, float grad)
{
    // (x - xc)^2 + (y - yc)^2 + (z - zc)^2 - r^2 = 0
    // gradient = 2[x - xc, y - yc, z - zc]
    // |gradient| = 2r

    float scale = grad * 0.5f / r;
    std::vector<float> params(5);
    params[0] = -2.0f * xc * scale;
    params[1] = -2.0f * yc * scale;
    params[2] = -2.0f * zc * scale;
    params[3] = scale;
    params[4] = (xc * xc + yc * yc + zc * zc - r * r) * scale;
    return params;
}

std::vector<float> TestSplatTreeCL::makePlane(float px, float py, float pz, float dx, float dy, float dz)
{
    std::vector<float> params(5);
    params[0] = dx;
    params[1] = dy;
    params[2] = dz;
    params[3] = 0.0f;
    params[4] = -(dx * px + dy * py + dz * pz);
    return params;
}

void TestSplatTreeCL::testProjectDistOrigin()
{
    float eps = std::numeric_limits<float>::epsilon() * 4;
    // General sphere case (3^2 + 4^2 + 12^2 = 13^2)
    ASSERT_DOUBLES_EQUAL(7.0f, callProjectDistOrigin(makeSphere(3.0f, 4.0f, 12.0f, 6.0f, 1.0f)), eps);
    ASSERT_DOUBLES_EQUAL(7.0f, callProjectDistOrigin(makeSphere(3.0f, 4.0f, 12.0f, 6.0f, 2.5f)), eps);
    ASSERT_DOUBLES_EQUAL(-7.0f, callProjectDistOrigin(makeSphere(3.0f, 4.0f, 12.0f, 6.0f, -2.5f)), eps);
    ASSERT_DOUBLES_EQUAL(0.0f, callProjectDistOrigin(makeSphere(3.0f, 4.0f, 12.0f, 13.0f, 2.5f)), eps);
    ASSERT_DOUBLES_EQUAL(-5.0f, callProjectDistOrigin(makeSphere(3.0f, 4.0f, 12.0f, 18.0f, 2.5f)), eps);
    // Origin at center of sphere
    ASSERT_DOUBLES_EQUAL(-6.0f, callProjectDistOrigin(makeSphere(0.0f, 0.0f, 0.0f, 6.0f, 2.5f)), eps);
    ASSERT_DOUBLES_EQUAL(5.0f, callProjectDistOrigin(makeSphere(0.0f, 0.0f, 0.0f, 5.0f, -1.5f)), eps);

    // Plane
    ASSERT_DOUBLES_EQUAL(-5.0f / 1.5f, callProjectDistOrigin(makePlane(1.0f, 2.0f, 3.0f, 1.0f, 0.5f, 1.0f)), eps);
    ASSERT_DOUBLES_EQUAL(5.0f / 1.5f, callProjectDistOrigin(makePlane(-1.0f, -2.0f, -3.0f, 1.0f, 0.5f, 1.0f)), eps);
}

void TestSplatTreeCL::testFitSphere()
{
    using std::tr1::variate_generator;
    using std::tr1::uniform_real;
    using std::tr1::mt19937;
    static const double pi = boost::math::constants::pi<double>();
    mt19937 engine;
    variate_generator<mt19937 &, uniform_real<double> > zGen(engine, uniform_real<double>(-1.0, 1.0));
    variate_generator<mt19937 &, uniform_real<double> > tGen(engine, uniform_real<double>(-pi, pi));
    variate_generator<mt19937 &, uniform_real<double> > wGen(engine, uniform_real<double>(0.0, 1.0));

    const std::size_t N = 20;
    const float center[3] = {1.0f, 2.0f, 3.5f};
    const float radius = 6.5f;
    const float eps = std::numeric_limits<float>::epsilon() * 10;

    std::vector<Splat> splats(N);
    for (std::size_t i = 0; i < N; i++)
    {
        double z = zGen();
        double t = tGen();
        double xy_len = sqrt(1.0 - z * z);
        double x = cos(t) * xy_len;
        double y = sin(t) * xy_len;

        splats[i].normal[0] = x;
        splats[i].normal[1] = y;
        splats[i].normal[2] = z;
        splats[i].position[0] = center[0] + x * radius;
        splats[i].position[1] = center[1] + y * radius;
        splats[i].position[2] = center[2] + z * radius;
        splats[i].quality = wGen();
    }
    std::vector<float> params = callFitSphere(splats);
    for (std::size_t i = 0; i < N; i++)
    {
        float x = splats[i].position[0];
        float y = splats[i].position[1];
        float z = splats[i].position[2];
        float v = params[0] * x + params[1] * y + params[2] * z + params[3] * (x * x + y * y + z * z) + params[4];
        ASSERT_DOUBLES_EQUAL(0.0f, v, eps);

        const float g[3] = {
            2 * params[3] * x + params[0],
            2 * params[3] * y + params[1],
            2 * params[3] * z + params[2]
        };
        ASSERT_DOUBLES_EQUAL(splats[i].normal[0], g[0], eps);
        ASSERT_DOUBLES_EQUAL(splats[i].normal[1], g[1], eps);
        ASSERT_DOUBLES_EQUAL(splats[i].normal[2], g[2], eps);
    }
}
