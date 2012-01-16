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
#include <map>
#include <string>
#include <cmath>
#include <fstream>
#include <boost/array.hpp>
#include <CL/cl.hpp>
#include "testmain.h"
#include "test_clh.h"
#include "../src/clh.h"
#include "../src/marching.h"
#include "../src/fast_ply.h"

using namespace std;

/**
 * Function object to model a signed distance from a sphere.
 */
class SphereFunc
{
private:
    std::size_t width, height, depth;
    float cx, cy, cz;
    float radius;
    vector<float> sliceData;

public:
    void operator()(const cl::CommandQueue &queue, const cl::Image2D &slice,
                    cl_uint z,
                    const std::vector<cl::Event> *events,
                    cl::Event *event)
    {
        for (cl_uint y = 0; y < height; y++)
            for (cl_uint x = 0; x < width; x++)
            {
                float d = std::sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy) + (z - cz) * (z - cz));
                sliceData[y * width + x] = d - radius;
            }

        cl::size_t<3> origin, region;
        origin[0] = 0; origin[1] = 0; origin[2] = 0;
        region[0] = width; region[1] = height; region[2] = 1;
        queue.enqueueWriteImage(slice, CL_FALSE, origin, region, width * sizeof(float), 0, &sliceData[0],
                                events, event);
    }

    SphereFunc(std::size_t width, std::size_t height, std::size_t depth,
               float cx, float cy, float cz, float radius)
        : width(width), height(height), depth(depth),
        cx(cx), cy(cy), cz(cz), radius(radius),
        sliceData(width * height)
    {
    }
};

class OutputFunctor
{
private:
    vector<cl_float> &hVertices;
    vector<boost::array<cl_uint, 3> > &hIndices;

public:
    OutputFunctor(vector<cl_float> &hVertices, vector<boost::array<cl_uint, 3> > &hIndices)
        : hVertices(hVertices), hIndices(hIndices) {}

    void operator()(const cl::CommandQueue &queue,
                    const cl::Buffer &vertices,
                    const cl::Buffer &indices,
                    std::size_t numVertices,
                    std::size_t numIndices,
                    cl::Event *event) const
    {
        cl::Event last;
        std::vector<cl::Event> wait(1);

        std::size_t oldVertices = hVertices.size();
        std::size_t oldIndices = hIndices.size();
        hVertices.resize(oldVertices + numVertices * 3);
        hIndices.resize(oldIndices + numIndices / 3);
        queue.enqueueReadBuffer(vertices, CL_FALSE, 0, numVertices * 3 * sizeof(cl_float), &hVertices[oldVertices],
                                NULL, &last);
        wait[0] = last;
        queue.enqueueReadBuffer(indices, CL_FALSE, 0, numIndices * sizeof(cl_uint), &hIndices[oldIndices],
                                &wait, &last);
        if (event != NULL)
            *event = last;
    }
};

/**
 * Tests for @ref Marching.
 */
class TestMarching : public CLH::Test::TestFixture
{
    CPPUNIT_TEST_SUITE(TestMarching);
    CPPUNIT_TEST(testConstructor);
    CPPUNIT_TEST(testComputeKey);
    // CPPUNIT_TEST(testCompactVertices);
    CPPUNIT_TEST(testSphere);
    CPPUNIT_TEST_SUITE_END();

private:
    /// Read the contents of a buffer and return it (synchronously) as a vector.
    template<typename T>
    vector<T> bufferToVector(const cl::Buffer &buffer);

    /// Build a vertex key
    static cl_ulong makeKey(cl_uint x, cl_uint y, cl_uint z, bool external);
    /// Wrapper that calls @ref computeKey and returns result
    cl_ulong callComputeKey(cl::Kernel &kernel,
                            cl_uint cx, cl_uint cy, cl_uint cz,
                            cl_uint tx, cl_uint ty, cl_uint tz);

    void testConstructor();    ///< Basic sanity tests on the tables
    void testComputeKey();     ///< Test @ref computeKey helper function
    void testSphere();         ///< Builds a sphere
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestMarching, TestSet::perCommit());

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
    Marching marching(context, device, 2, 2);

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

cl_ulong TestMarching::makeKey(cl_uint x, cl_uint y, cl_uint z, bool external)
{
    cl_ulong ans = (cl_ulong(z) << 42) | (cl_ulong(y) << 21) | (cl_ulong(x));
    if (external)
        ans |= cl_ulong(1) << 63;
    return ans;
}

cl_ulong TestMarching::callComputeKey(
    cl::Kernel &kernel,
    cl_uint cx, cl_uint cy, cl_uint cz,
    cl_uint tx, cl_uint ty, cl_uint tz)
{
    cl_ulong ans = 0;
    cl_uint3 coords = {{ cx, cy, cz }};
    cl_uint3 top = {{ tx, ty, tz }};
    cl::Buffer out(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_ulong), &ans);

    kernel.setArg(0, out);
    kernel.setArg(1, coords);
    kernel.setArg(2, top);
    queue.enqueueTask(kernel);
    queue.enqueueReadBuffer(out, CL_TRUE, 0, sizeof(cl_ulong), &ans);
    return ans;
}

void TestMarching::testComputeKey()
{
    map<string, string> defines;
    defines["UNIT_TESTS"] = "1";
    cl::Program program = CLH::build(context, "kernels/marching.cl", defines);
    cl::Kernel kernel(program, "testComputeKey");

    CPPUNIT_ASSERT_EQUAL(makeKey(0, 0, 0, true),    callComputeKey(kernel, 0, 0, 0, 32, 32, 32));
    CPPUNIT_ASSERT_EQUAL(makeKey(1, 2, 3, false),   callComputeKey(kernel, 1, 2, 3, 32, 32, 32));
    CPPUNIT_ASSERT_EQUAL(makeKey(0, 4, 5, true),    callComputeKey(kernel, 0, 4, 5, 32, 32, 32));
    CPPUNIT_ASSERT_EQUAL(makeKey(6, 0, 7, true),    callComputeKey(kernel, 6, 0, 7, 32, 32, 32));
    CPPUNIT_ASSERT_EQUAL(makeKey(9, 5, 0, false),   callComputeKey(kernel, 9, 5, 0, 32, 32, 32));
    CPPUNIT_ASSERT_EQUAL(makeKey(30, 1, 2, true),   callComputeKey(kernel, 30, 1, 2, 30, 40, 50));
    CPPUNIT_ASSERT_EQUAL(makeKey(5, 40, 3, true),   callComputeKey(kernel, 5, 40, 3, 30, 40, 50));
    CPPUNIT_ASSERT_EQUAL(makeKey(1, 2, 50, true),   callComputeKey(kernel, 1, 2, 50, 30, 40, 50));
    CPPUNIT_ASSERT_EQUAL(makeKey(1, 2, 40, false),  callComputeKey(kernel, 1, 2, 40, 30, 40, 50));
    CPPUNIT_ASSERT_EQUAL(makeKey(1, 2, 30, false),  callComputeKey(kernel, 1, 2, 30, 30, 40, 50));
    CPPUNIT_ASSERT_EQUAL(makeKey(30, 40, 50, true), callComputeKey(kernel, 30, 40, 50, 30, 40, 50));
}

void TestMarching::testSphere()
{
    const std::size_t maxWidth = 83;
    const std::size_t maxHeight = 78;
    const std::size_t width = 71;
    const std::size_t height = 75;
    const std::size_t depth = 60;
    cl::Event done;

    const float ref[3] = {0.0f, 0.0f, 0.0f};
    Grid grid(ref, 1.0f, 0, width - 1, 0, height - 1, 0, depth - 1);

    // Replace the command queue with an out-of-order one, to ensure that the
    // events are being handled correctly.
    queue = cl::CommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);

    Marching marching(context, device, maxWidth, maxHeight);
    SphereFunc input(width, height, depth, 30.0, 41.5, 27.75, 25.3);

    std::vector<cl_float> hVertices;
    std::vector<boost::array<cl_uint, 3> > hIndices;

    OutputFunctor output(hVertices, hIndices);
    marching.generate(queue, input, output, grid, 0, NULL);

    FastPly::Writer writer;
    writer.setNumVertices(hVertices.size() / 3);
    writer.setNumTriangles(hIndices.size());
    writer.open("sphere.ply");
    writer.writeVertices(0, hVertices.size() / 3, &hVertices[0]);
    writer.writeTriangles(0, hIndices.size(), &hIndices[0][0]);
}
