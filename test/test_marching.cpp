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
#include <cmath>
#include <iostream>
#include "testmain.h"
#include "test_clh.h"
#include "../src/marching.h"

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

/**
 * Tests for @ref Marching.
 */
class TestMarching : public CLH::Test::TestFixture
{
    CPPUNIT_TEST_SUITE(TestMarching);
    CPPUNIT_TEST(testConstructor);
    CPPUNIT_TEST(testSphere);
    CPPUNIT_TEST_SUITE_END();

private:
    /// Read the contents of a buffer and return it (synchronously) as a vector.
    template<typename T>
    vector<T> bufferToVector(const cl::Buffer &buffer);

    void testConstructor();    ///< Basic sanity tests on the tables
    void testSphere();         ///< Builds a sphere
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

void TestMarching::testSphere()
{
    const std::size_t width = 71;
    const std::size_t height = 75;
    const std::size_t depth = 60;
    cl::Event done;

    cl_float3 scale, bias;
    scale.s[0] = 1.0f;
    scale.s[1] = 1.0f;
    scale.s[2] = 1.0f;
    bias.s[0] = 0.0f;
    bias.s[1] = 0.0f;
    bias.s[2] = 0.0f;

    Marching marching(context, device, width, height, depth);
    SphereFunc func(width, height, depth, 30.0, 41.5, 27.75, 25.3);
    cl::Buffer vertices(context, CL_MEM_WRITE_ONLY, 1000000 * sizeof(cl_float3));
    cl::Buffer indices(context, CL_MEM_WRITE_ONLY, 1000000 * sizeof(cl_uint));
    cl_uint2 totals;

    marching.enqueue(queue, func, scale, bias, vertices, indices, &totals, NULL, &done);
    done.wait();
#if 0
    vector<cl_float3> hVertices(totals.s0);
    vector<cl_uint> hIndices(totals.s1);
    queue.enqueueReadBuffer(vertices, CL_TRUE, 0, totals.s0 * sizeof(cl_float3), &hVertices[0]);
    queue.enqueueReadBuffer(indices, CL_TRUE, 0, totals.s1 * sizeof(cl_uint), &hIndices[0]);
    cout << "ply\n"
         << "format ascii 1.0\n"
         << "element vertex " << totals.s0 << '\n'
         << "property float32 x\n"
         << "property float32 y\n"
         << "property float32 z\n"
         << "element face " << totals.s1 / 3 << '\n'
         << "property list uint8 uint32 vertex_indices\n"
         << "end_header\n";
    for (std::size_t i = 0; i < totals.s0; i++)
    {
        cout << hVertices[i].x << ' ' << hVertices[i].y << ' ' << hVertices[i].z << '\n';
    }
    for (std::size_t i = 0; i < totals.s1; i += 3)
    {
        cout << "3 " << hIndices[i] << ' ' << hIndices[i + 1] << ' ' << hIndices[i + 2] << '\n';
    }
#endif
}
