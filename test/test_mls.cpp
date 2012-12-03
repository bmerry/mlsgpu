/**
 * @file
 *
 * Tests for @ref MlsFunctor and related kernel code.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif

#ifndef __CL_ENABLE_EXCEPTIONS
# define __CL_ENABLE_EXCEPTIONS
#endif

#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/extensions/HelperMacros.h>
#include <vector>
#include <cstddef>
#include <limits>
#include <boost/tr1/random.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>
#include <CL/cl.hpp>
#include "testutil.h"
#include "test_clh.h"
#include "../src/clh.h"
#include "../src/splat.h"
#include "../src/mls.h"
#include "../src/splat_tree_cl.h"

using namespace std;

/**
 * Tests for @ref MlsFunctor and related kernel code.
 * @todo Add tests for plane fitting.
 */
class TestMls : public CLH::Test::TestFixture
{
    CPPUNIT_TEST_SUITE(TestMls);
    CPPUNIT_TEST(testMakeCode);
    CPPUNIT_TEST(testDecode);
    CPPUNIT_TEST(testSolveQuadratic);
    CPPUNIT_TEST(testFitSphere);
    CPPUNIT_TEST(testProjectDistOriginSphere);
    CPPUNIT_TEST(testProcessCorners);
    CPPUNIT_TEST_SUITE_END();

private:
    cl::Program mlsProgram;     ///< Program compiled from @ref mls.cl.

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

    /**
     * Generate splats over the surface of a sphere, with random positions and
     * weights.  The radii of the splats are randomly selected between @a
     * radius and 2 * @a radius.
     *
     * @param N          Number of splats to generate.
     * @param center     Center of the sphere.
     * @param radius     Radius of the sphere.
     */
    std::vector<Splat> sphereSplats(std::size_t N, const float center[3], float radius);

    int callMakeCode(cl_int x, cl_int y, cl_int z);
    cl_int3 callDecode(cl_uint code);
    float callSolveQuadratic(float a, float b, float c);
    float callProjectDistOriginSphere(float p0, float p1, float p2, float p3, float p4);
    float callProjectDistOriginSphere(const std::vector<float> &params);

    /**
     * Wrapper around @c testFitSphere in @ref mls.cl.
     * @param splats     Two or more splats. The radius is ignored, and the weight should be
     *                   placed directly into the quality slot. The positions
     *                   are in the local coordinate system in which the sphere is fitted.
     * @return A 5-element vector of algebraic sphere parameters.
     */
    std::vector<float> callFitSphere(const std::vector<Splat> &splats);

public:
    virtual void setUp();
    virtual void tearDown();

    void testMakeCode();           ///< Test @ref makeCode in @ref mls.cl.
    void testDecode();             ///< Test @ref decode in @ref mls.cl.
    void testSolveQuadratic();     ///< Test @ref solveQuadratic in @ref mls.cl.
    void testProjectDistOriginSphere();  ///< Test @ref projectDistOriginSphere in @ref mls.cl.
    void testFitSphere();          ///< Test @ref fitSphere in @ref mls.cl.

    void testProcessCorners();     ///< Test the @ref processCorners kernel.

    // TODO: test boundary handling
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestMls, TestSet::perCommit());

void TestMls::setUp()
{
    CLH::Test::TestFixture::setUp();
    map<string, string> defines;
    defines["UNIT_TESTS"] = "1";
    defines["WGS_X"] = boost::lexical_cast<std::string>(MlsFunctor::wgs[0]);
    defines["WGS_Y"] = boost::lexical_cast<std::string>(MlsFunctor::wgs[1]);
    defines["WGS_Z"] = boost::lexical_cast<std::string>(MlsFunctor::wgs[2]);
    defines["FIT_SPHERE"] = "1";
    defines["FIT_PLANE"] = "0";
    mlsProgram = CLH::build(context, "kernels/mls.cl", defines);
}

void TestMls::tearDown()
{
    mlsProgram = NULL;
    CLH::Test::TestFixture::tearDown();
}

std::vector<float> TestMls::makeSphere(float xc, float yc, float zc, float r, float grad)
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

std::vector<float> TestMls::makePlane(float px, float py, float pz, float dx, float dy, float dz)
{
    std::vector<float> params(5);
    params[0] = dx;
    params[1] = dy;
    params[2] = dz;
    params[3] = 0.0f;
    params[4] = -(dx * px + dy * py + dz * pz);
    return params;
}

int TestMls::callMakeCode(cl_int x, cl_int y, cl_int z)
{
    cl_uint ans;
    cl::Buffer out(context, CL_MEM_WRITE_ONLY, sizeof(cl_uint));
    cl::Kernel kernel(mlsProgram, "testMakeCode");
    cl_int3 xyz;
    xyz.s[0] = x; xyz.s[1] = y; xyz.s[2] = z;
    kernel.setArg(0, out);
    kernel.setArg(1, xyz);
    queue.enqueueTask(kernel);
    queue.enqueueReadBuffer(out, CL_TRUE, 0, sizeof(cl_uint), &ans);
    return ans;
}

cl_int3 TestMls::callDecode(cl_uint code)
{
    cl_int3 ans;
    cl::Buffer out(context, CL_MEM_WRITE_ONLY, sizeof(cl_int3));
    cl::Kernel kernel(mlsProgram, "testDecode");
    kernel.setArg(0, out);
    kernel.setArg(1, code);
    queue.enqueueTask(kernel);
    queue.enqueueReadBuffer(out, CL_TRUE, 0, sizeof(cl_int3), &ans);
    return ans;
}

float TestMls::callSolveQuadratic(float a, float b, float c)
{
    CPPUNIT_ASSERT(b >= 0.0f);
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

float TestMls::callProjectDistOriginSphere(float p0, float p1, float p2, float p3, float p4)
{
    cl_float ans;
    cl::Buffer out(context, CL_MEM_WRITE_ONLY, sizeof(cl_float));
    cl::Kernel kernel(mlsProgram, "testProjectDistOriginSphere");
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

float TestMls::callProjectDistOriginSphere(const std::vector<float> &params)
{
    CPPUNIT_ASSERT_EQUAL(std::vector<float>::size_type(5), params.size());
    return callProjectDistOriginSphere(params[0], params[1], params[2], params[3], params[4]);
}

std::vector<float> TestMls::callFitSphere(const std::vector<Splat> &splats)
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

void TestMls::testMakeCode()
{
    CPPUNIT_ASSERT_EQUAL(0, callMakeCode(0, 0, 0));
    CPPUNIT_ASSERT_EQUAL(7, callMakeCode(1, 1, 1));
    CPPUNIT_ASSERT_EQUAL(174, callMakeCode(2, 5, 3));
    CPPUNIT_ASSERT_EQUAL(511, callMakeCode(7, 7, 7));
}

void TestMls::testDecode()
{
    cl_int3 out;

    out = callDecode(0xB1);  // 010 110 001
    CPPUNIT_ASSERT_EQUAL(cl_int(1), out.s[0]);
    CPPUNIT_ASSERT_EQUAL(cl_int(6), out.s[1]);
    CPPUNIT_ASSERT_EQUAL(cl_int(2), out.s[2]);

    out = callDecode(0xAAAAAAAA);  // 010 101 010 101 010 101 010 101 010 101 010
    CPPUNIT_ASSERT_EQUAL(cl_int(0x2AA), out.s[0]);
    CPPUNIT_ASSERT_EQUAL(cl_int(0x555), out.s[1]);
    CPPUNIT_ASSERT_EQUAL(cl_int(0x2AA), out.s[2]);

    out = callDecode(0x24924924);
    CPPUNIT_ASSERT_EQUAL(cl_int(0), out.s[0]);
    CPPUNIT_ASSERT_EQUAL(cl_int(0), out.s[1]);
    CPPUNIT_ASSERT_EQUAL(cl_int(0x3FF), out.s[2]);

    out = callDecode(0);
    CPPUNIT_ASSERT_EQUAL(cl_int(0), out.s[1]);
    CPPUNIT_ASSERT_EQUAL(cl_int(0), out.s[1]);
}

void TestMls::testSolveQuadratic()
{
    float n = std::numeric_limits<float>::quiet_NaN();
    float eps = std::numeric_limits<float>::epsilon() * 4;

    // Cases with no roots
    MLSGPU_ASSERT_DOUBLES_EQUAL(n, callSolveQuadratic(-1, 2, -2), eps);
    MLSGPU_ASSERT_DOUBLES_EQUAL(n, callSolveQuadratic(-1e20, 2e10, -1.0001), eps);
    MLSGPU_ASSERT_DOUBLES_EQUAL(n, callSolveQuadratic(1, 0, 1), eps);
    MLSGPU_ASSERT_DOUBLES_EQUAL(n, callSolveQuadratic(-1, 0, -1), eps);
    // Constant functions (no roots or infinitely many roots)
    MLSGPU_ASSERT_DOUBLES_EQUAL(n, callSolveQuadratic(0, 0, 0), eps);
    MLSGPU_ASSERT_DOUBLES_EQUAL(n, callSolveQuadratic(0, 0, 4), eps);
    MLSGPU_ASSERT_DOUBLES_EQUAL(n, callSolveQuadratic(0, 0, -3), eps);
    MLSGPU_ASSERT_DOUBLES_EQUAL(n, callSolveQuadratic(0, 0, -1e20), eps);
    MLSGPU_ASSERT_DOUBLES_EQUAL(n, callSolveQuadratic(0, 0, 1e20), eps);
    // Linear functions
    MLSGPU_ASSERT_DOUBLES_EQUAL(-1.5, callSolveQuadratic(0, 2, 3), eps);
    MLSGPU_ASSERT_DOUBLES_EQUAL(0.0, callSolveQuadratic(0, 5, 0), eps);
    MLSGPU_ASSERT_DOUBLES_EQUAL(0.0, callSolveQuadratic(0, 1e20, 0), eps);
    MLSGPU_ASSERT_DOUBLES_EQUAL(0.0, callSolveQuadratic(0, 1e-20, 0), eps);
    MLSGPU_ASSERT_DOUBLES_EQUAL(1e-20, callSolveQuadratic(0, 1e10, 1e-10), eps);
    MLSGPU_ASSERT_DOUBLES_EQUAL(-1e20, callSolveQuadratic(0, 1e-10, 1e10), eps);
    // Repeated roots
    MLSGPU_ASSERT_DOUBLES_EQUAL(1.0, callSolveQuadratic(-1, 2, -1), eps);
    MLSGPU_ASSERT_DOUBLES_EQUAL(1.0, callSolveQuadratic(-10, 20, -10), eps);
    MLSGPU_ASSERT_DOUBLES_EQUAL(1e4, callSolveQuadratic(-1, 2e4, -1e8), eps);
    MLSGPU_ASSERT_DOUBLES_EQUAL(0.0, callSolveQuadratic(1, 0, 0), eps);
    MLSGPU_ASSERT_DOUBLES_EQUAL(0.0, callSolveQuadratic(1e30, 0, 0), eps);
    MLSGPU_ASSERT_DOUBLES_EQUAL(0.0, callSolveQuadratic(1e-20, 0, 0), eps);
    // Regular two-root solutions
    MLSGPU_ASSERT_DOUBLES_EQUAL(2.0, callSolveQuadratic(-1, 5, -6), eps);
    MLSGPU_ASSERT_DOUBLES_EQUAL(2.0, callSolveQuadratic(-2, 10, -12), eps);
    MLSGPU_ASSERT_DOUBLES_EQUAL(2.0, callSolveQuadratic(1, 1, -6), eps);
    MLSGPU_ASSERT_DOUBLES_EQUAL(2.0, callSolveQuadratic(0.1f, 0.1f, -0.6f), eps);
    MLSGPU_ASSERT_DOUBLES_EQUAL(2.0, callSolveQuadratic(-1e-12, 5e-12, -6e-12), eps);
    MLSGPU_ASSERT_DOUBLES_EQUAL(-2e-12, callSolveQuadratic(1, 5e-12, 6e-24), eps);
    // Corner cases for stability
    MLSGPU_ASSERT_DOUBLES_EQUAL(1e-6, callSolveQuadratic(-1, 1 + 1e-6, -1e-6), eps);
    MLSGPU_ASSERT_DOUBLES_EQUAL(1.0f, callSolveQuadratic(-1, 1 + 1e6, -1e6), eps);
    MLSGPU_ASSERT_DOUBLES_EQUAL(1e20, callSolveQuadratic(-1e-20, 2, -1e20), eps);
    MLSGPU_ASSERT_DOUBLES_EQUAL(-1e-6, callSolveQuadratic(1e-6, 1, 1e-6), eps);
}

void TestMls::testProjectDistOriginSphere()
{
    float eps = std::numeric_limits<float>::epsilon() * 4;
    // General sphere case (3^2 + 4^2 + 12^2 = 13^2)
    MLSGPU_ASSERT_DOUBLES_EQUAL(7.0f, callProjectDistOriginSphere(makeSphere(3.0f, 4.0f, 12.0f, 6.0f, 1.0f)), eps);
    MLSGPU_ASSERT_DOUBLES_EQUAL(7.0f, callProjectDistOriginSphere(makeSphere(3.0f, 4.0f, 12.0f, 6.0f, 2.5f)), eps);
    MLSGPU_ASSERT_DOUBLES_EQUAL(-7.0f, callProjectDistOriginSphere(makeSphere(3.0f, 4.0f, 12.0f, 6.0f, -2.5f)), eps);
    MLSGPU_ASSERT_DOUBLES_EQUAL(0.0f, callProjectDistOriginSphere(makeSphere(3.0f, 4.0f, 12.0f, 13.0f, 2.5f)), eps);
    MLSGPU_ASSERT_DOUBLES_EQUAL(-5.0f, callProjectDistOriginSphere(makeSphere(3.0f, 4.0f, 12.0f, 18.0f, 2.5f)), eps);
    // Origin at center of sphere
    MLSGPU_ASSERT_DOUBLES_EQUAL(-6.0f, callProjectDistOriginSphere(makeSphere(0.0f, 0.0f, 0.0f, 6.0f, 2.5f)), eps);
    MLSGPU_ASSERT_DOUBLES_EQUAL(5.0f, callProjectDistOriginSphere(makeSphere(0.0f, 0.0f, 0.0f, 5.0f, -1.5f)), eps);

    // Plane
    MLSGPU_ASSERT_DOUBLES_EQUAL(-5.0f / 1.5f, callProjectDistOriginSphere(makePlane(1.0f, 2.0f, 3.0f, 1.0f, 0.5f, 1.0f)), eps);
    MLSGPU_ASSERT_DOUBLES_EQUAL(5.0f / 1.5f, callProjectDistOriginSphere(makePlane(-1.0f, -2.0f, -3.0f, 1.0f, 0.5f, 1.0f)), eps);
}

std::vector<Splat> TestMls::sphereSplats(std::size_t N, const float center[3], float radius)
{
    using std::tr1::variate_generator;
    using std::tr1::uniform_real;
    using std::tr1::mt19937;
    static const double pi = boost::math::constants::pi<double>();
    mt19937 engine;
    variate_generator<mt19937 &, uniform_real<double> > zGen(engine, uniform_real<double>(-1.0, 1.0));
    variate_generator<mt19937 &, uniform_real<double> > tGen(engine, uniform_real<double>(-pi, pi));
    variate_generator<mt19937 &, uniform_real<double> > wGen(engine, uniform_real<double>(0.0, 1.0));
    variate_generator<mt19937 &, uniform_real<double> > rGen(engine, uniform_real<double>(radius, 2.0 * radius));

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
        splats[i].radius = rGen();
        splats[i].position[0] = center[0] + x * radius;
        splats[i].position[1] = center[1] + y * radius;
        splats[i].position[2] = center[2] + z * radius;
        splats[i].quality = wGen();
    }
    return splats;
}

void TestMls::testFitSphere()
{
    const std::size_t N = 20;
    const float center[3] = {1.0f, 2.0f, 3.5f};
    const float radius = 6.5f;
    const float eps = std::numeric_limits<float>::epsilon() * 16;

    std::vector<Splat> splats = sphereSplats(N, center, radius);
    std::vector<float> params = callFitSphere(splats);
    // Check that all the input splats are on the fitted sphere and with the right gradient
    for (std::size_t i = 0; i < N; i++)
    {
        double x = splats[i].position[0];
        double y = splats[i].position[1];
        double z = splats[i].position[2];
        double v = params[0] * x + params[1] * y + params[2] * z + params[3] * (x * x + y * y + z * z) + params[4];
        MLSGPU_ASSERT_DOUBLES_EQUAL(0.0f, v, eps);

        const float g[3] = {
            float(2 * params[3] * x + params[0]),
            float(2 * params[3] * y + params[1]),
            float(2 * params[3] * z + params[2])
        };
        MLSGPU_ASSERT_DOUBLES_EQUAL(splats[i].normal[0], g[0], eps);
        MLSGPU_ASSERT_DOUBLES_EQUAL(splats[i].normal[1], g[1], eps);
        MLSGPU_ASSERT_DOUBLES_EQUAL(splats[i].normal[2], g[2], eps);
    }
}

static inline double sqr(double x)
{
    return x * x;
}

void TestMls::testProcessCorners()
{
    const std::size_t N = 50;
    const float center[3] = {10.0f, 20.0f, 35.0f};
    const float radius = 65.0f;

    // Not all compilers believe size[0] is a constant expression, so we
    // need to pull out the terms as constant expressions.
    const Grid::size_type sizeX = 19;
    const Grid::size_type sizeY = 24;
    const Grid::size_type sizeZ = 28;
    const Grid::size_type size[3] = {sizeX, sizeY, sizeZ};
    const Grid::size_type zFirst = MlsFunctor::wgs[2], zLast = 26;
    const Grid::difference_type offset[3] = { 20, 15, 33 };

    unsigned int subsampling = MlsFunctor::subsamplingMin;
    while ((Grid::size_type(2) << subsampling) < *max_element(size, size + 3))
        subsampling++;

    std::vector<Splat> hSplats = sphereSplats(N, center, radius);
    // splats will contain raw radii, but the kernel requires inverse-squared radii
    BOOST_FOREACH(Splat &splat, hSplats)
    {
        splat.radius = 1.0f / (splat.radius * splat.radius);
    }

    /* Build a simple octree. The intersection information isn't truly accurate, but it
     * allows us to cover 3 cases:
     * - sufficient hits
     * - insufficient but non-zero hits
     * - zero hits
     */
    std::vector<SplatTreeCL::command_type> hStart(8, 0);
    std::vector<SplatTreeCL::command_type> hCommands;

    // First N - 2 splats in one range, then the last two in another
    hCommands.push_back(N - 1);
    for (unsigned int i = 0; i < N - 2; i++)
        hCommands.push_back(i);
    hCommands.push_back(-2 - (SplatTreeCL::command_type) hCommands.size()); // chain to next
    hStart[6] = -1;    // no hits
    hStart[7] = hCommands.size(); // last 2 hits

    hCommands.push_back(hCommands.size() + 3);
    hCommands.push_back(N - 2);
    hCommands.push_back(N - 1);
    hCommands.push_back(-1);

    cl::Buffer dSplats(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       N * sizeof(Splat), &hSplats[0]);
    cl::Buffer dStart(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                      hStart.size() * sizeof(SplatTreeCL::command_type), &hStart[0]);
    cl::Buffer dCommands(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                         hCommands.size() * sizeof(SplatTreeCL::command_type), &hCommands[0]);

    MlsFunctor generator(context, MLS_SHAPE_SPHERE);
    Grid::size_type zStride;
    cl::Image2D dCorners = generator.allocateSlices(size[0], size[1], zLast - zFirst, zStride);

    generator.set(offset, dSplats, dCommands, dStart, subsampling);
    generator.enqueue(queue, dCorners, size, zFirst, zLast, zStride, NULL, NULL);
    queue.finish();

    // Read back and verify results
    for (Grid::size_type z = zFirst; z < zLast; z++)
    {
        cl_float hCorners[sizeY][sizeX];
        cl::size_t<3> origin, region;
        origin[0] = 0; origin[1] = zStride * (z - zFirst); origin[2] = 0;
        region[0] = size[0]; region[1] = size[1]; region[2] = 1;
        queue.enqueueReadImage(dCorners, CL_TRUE, origin, region, 0, 0, &hCorners[0][0]);

        for (unsigned int y = 0; y < size[1]; y++)
            for (unsigned int x = 0; x < size[0]; x++)
            {
                float cx = x + offset[0];
                float cy = y + offset[1];
                float cz = z + offset[2];
                float expected = sqrt(sqr(cx - center[0]) + sqr(cy - center[1]) + sqr(cz - center[2]))
                    - radius;
                if ((z >> subsampling) == 1 && (y >> subsampling) == 1)
                    expected = std::numeric_limits<float>::quiet_NaN(); // the special cases in hStart
                if (fabs(expected) > std::sqrt(3.0f))
                    expected = std::numeric_limits<float>::quiet_NaN(); // divergence test
                float actual = hCorners[y][x];
                MLSGPU_ASSERT_DOUBLES_EQUAL(expected, actual, 1e-5);
            }
    }
}
