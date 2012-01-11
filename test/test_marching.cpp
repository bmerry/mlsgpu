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
#include <fstream>
#include <boost/array.hpp>
#include "testmain.h"
#include "test_clh.h"
#include "../src/marching.h"
#include "../src/ply.h"
#include "../src/ply_mesh.h"

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

class OutputFunctor
{
private:
    vector<cl_float3> &hVertices;
    vector<boost::array<cl_uint, 3> > &hIndices;

public:
    OutputFunctor(vector<cl_float3> &hVertices, vector<boost::array<cl_uint, 3> > &hIndices)
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
        hVertices.resize(oldVertices + numVertices);
        hIndices.resize(oldIndices + numIndices / 3);
        queue.enqueueReadBuffer(vertices, CL_FALSE, 0, numVertices * sizeof(cl_float3), &hVertices[oldVertices],
                                NULL, &last);
        wait[0] = last;
        queue.enqueueReadBuffer(indices, CL_FALSE, 0, numIndices * sizeof(cl_uint), &hIndices[oldIndices],
                                &wait, &last);
        if (event != NULL)
            *event = last;
    }
};

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

    std::vector<cl_float3> hVertices;
    std::vector<boost::array<cl_uint, 3> > hIndices;

    OutputFunctor output(hVertices, hIndices);
    marching.generate(queue, input, output, grid, 0, NULL);

    std::filebuf out;
    out.open("sphere.ply", ios::out | ios::binary);
    PLY::Writer writer(PLY::FILE_FORMAT_LITTLE_ENDIAN, &out);
    writer.addElement(PLY::makeElementRangeWriter(
            hVertices.begin(), hVertices.end(), hVertices.size(),
            PLY::VertexFetcher()));
    writer.addElement(PLY::makeElementRangeWriter(
            hIndices.begin(), hIndices.end(), hIndices.size(),
            PLY::TriangleFetcher()));
    writer.write();
}
