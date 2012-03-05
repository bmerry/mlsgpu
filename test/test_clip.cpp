/**
 * @file
 *
 * Tests for @ref Clip.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif

#ifndef __CL_ENABLE_EXCEPTIONS
# define __CL_ENABLE_EXCEPTIONS
#endif

#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/extensions/HelperMacros.h>
#include <CL/cl.hpp>
#include <vector>
#include <algorithm>
#include <boost/bind.hpp>
#include <boost/ref.hpp>
#include <boost/array.hpp>
#include <tr1/cstdint>
#include "../src/marching.h"
#include "../src/clip.h"
#include "../src/clh.h"
#include "test_clh.h"
#include "testmain.h"

/**
 * Tests @ref Clip.
 *
 * As input it uses an MxN regular grid of vertices. It clips it against
 * a line running through the grid (aligned to the grid). Vertex keys are
 * constructed as 0xCAFEBABExxyy0000.
 *
 * The internal and external vertices are separated along a line orthogonal
 * to the boundary clipping.
 */
class TestClip : public CLH::Test::TestFixture
{
    CPPUNIT_TEST_SUITE(TestClip);
    CPPUNIT_TEST(testSmall);
    CPPUNIT_TEST(testBig);
    CPPUNIT_TEST(testInternalOnly);
    CPPUNIT_TEST(testExternalOnly);
    CPPUNIT_TEST(testEmpty);
    CPPUNIT_TEST_SUITE_END();

private:
    /**
     * Data structure wrapping up mesh data. It exists mainly
     * @todo Make this a separate class and use it in multiple places.
     */
    struct HostKeyMesh
    {
        std::vector<cl_float3> vertices;
        std::vector<cl_ulong> vertexKeys;
        std::vector<boost::array<cl_uint, 3> > triangles;
        std::size_t numInternalVertices;

        HostKeyMesh() : vertices(), vertexKeys(), triangles(), numInternalVertices(0) {}
    };

    /**
     * Function object called by the clipper to give the output.
     */
    static void outputFunc(
        const cl::CommandQueue &queue,
        const cl::Buffer &vertices,
        const cl::Buffer &vertexKeys,
        const cl::Buffer &indices,
        std::size_t numVertices,
        std::size_t numInternalVertices,
        std::size_t numIndices,
        cl::Event *event,
        HostKeyMesh &out);

    /**
     * Function object called by the clipper to retrieve signed
     * distances.
     */
    static void distanceFunc(
        const cl::CommandQueue &queue,
        const cl::Buffer &distances,
        const cl::Buffer &vertices,
        std::size_t numVertices,
        const std::vector<cl::Event> *events,
        cl::Event *event,
        float cut);

    void testCase(int M, int N, int internalRows, int keepCols);

public:
    void testSmall()         { testCase(5, 5, 2, 3); }
    void testBig()           { testCase(1001, 1001, 950, 651); }
    void testInternalOnly()  { testCase(5, 5, 5, 2); }
    void testExternalOnly()  { testCase(7, 6, 0, 3); }
    void testEmpty()         { testCase(4, 7, 2, 0); }
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestClip, TestSet::perCommit());

void TestClip::outputFunc(
    const cl::CommandQueue &queue,
    const cl::Buffer &vertices,
    const cl::Buffer &vertexKeys,
    const cl::Buffer &indices,
    std::size_t numVertices,
    std::size_t numInternalVertices,
    std::size_t numIndices,
    cl::Event *event,
    HostKeyMesh &out)
{
    CPPUNIT_ASSERT(numVertices > 0);
    CPPUNIT_ASSERT(numIndices > 0);
    CPPUNIT_ASSERT(numIndices % 3 == 0);

    const std::size_t numTriangles = numIndices / 3;
    out.vertices.resize(numVertices);
    out.vertexKeys.resize(numVertices);
    out.triangles.resize(numTriangles);
    out.numInternalVertices = numInternalVertices;

    std::vector<cl::Event> events(3);
    queue.enqueueReadBuffer(vertices, CL_FALSE, 0, numVertices * sizeof(cl_float3),
                            &out.vertices[0], NULL, &events[0]);
    queue.enqueueReadBuffer(vertexKeys, CL_FALSE, 0, numVertices * sizeof(cl_ulong),
                            &out.vertexKeys[0], NULL, &events[1]);
    queue.enqueueReadBuffer(indices, CL_FALSE, 0, numTriangles * (3 * sizeof(cl_uint)),
                            &out.triangles[0][0], NULL, &events[2]);
    CLH::enqueueMarkerWithWaitList(queue, &events, event);
}

void TestClip::distanceFunc(
    const cl::CommandQueue &queue,
    const cl::Buffer &distances,
    const cl::Buffer &vertices,
    std::size_t numVertices,
    const std::vector<cl::Event> *events,
    cl::Event *event,
    float cut)
{
    std::vector<cl_float3> hVertices(numVertices);
    std::vector<cl_float> hDistances(numVertices);
    queue.enqueueReadBuffer(vertices, CL_TRUE, 0, numVertices * sizeof(cl_float3),
                            &hVertices[0], events, NULL);
    for (std::size_t i = 0; i < numVertices; i++)
        hDistances[i] = hVertices[i].s[0] - cut;

    queue.enqueueWriteBuffer(distances, CL_TRUE, 0, numVertices * sizeof(cl_float),
                             &hDistances[0], NULL, event);
}

void TestClip::testCase(int M, int N, int internalRows, int keepCols)
{
    HostKeyMesh in, out, expected;

    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
        {
            cl_float3 v = {{ j * 20.0f, i * 10.0f, 0.0f }};
            cl_ulong key = UINT64_C(0xCAFEBABE00000000) + (i << 24) + (j << 16);
            in.vertices.push_back(v);
            in.vertexKeys.push_back(key);
            if (j < keepCols)
            {
                expected.vertices.push_back(v);
                expected.vertexKeys.push_back(key);
            }
        }
    in.numInternalVertices = internalRows * N;
    expected.numInternalVertices = internalRows * keepCols;
    for (int i = 0; i < M - 1; i++)
        for (int j = 0; j < N - 1; j++)
        {
            int a = i * N + j;
            boost::array<cl_uint, 3> tris[2];
            tris[0][0] = a;
            tris[0][1] = a + 1;
            tris[0][2] = a + N;
            tris[1][0] = a + N;
            tris[1][1] = a + 1;
            tris[1][2] = a + N + 1;
            in.triangles.push_back(tris[0]);
            in.triangles.push_back(tris[1]);
            if (j + 1 < keepCols)
            {
                int b = i * keepCols + j;
                tris[0][0] = b;
                tris[0][1] = b + 1;
                tris[0][2] = b + keepCols;
                tris[1][0] = b + keepCols;
                tris[1][1] = b + 1;
                tris[1][2] = b + keepCols + 1;
                expected.triangles.push_back(tris[0]);
                expected.triangles.push_back(tris[1]);
            }
        }
    float cut = 20.0f * keepCols - 7.0f;

    Clip clip(context, device, in.vertices.size(), in.triangles.size());
    clip.setDistanceFunctor(boost::bind(&TestClip::distanceFunc,
                                        _1, _2, _3, _4, _5, _6, cut));
    clip.setOutput(boost::bind(&TestClip::outputFunc,
                               _1, _2, _3, _4, _5, _6, _7, _8,
                               boost::ref(out)));

    cl::Buffer dInVertices(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                           in.vertices.size() * sizeof(in.vertices[0]),
                           &in.vertices[0]);
    cl::Buffer dInVertexKeys(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                             in.vertexKeys.size() * sizeof(in.vertexKeys[0]),
                             &in.vertexKeys[0]);
    cl::Buffer dInIndices(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          in.triangles.size() * sizeof(in.triangles[0]),
                          &in.triangles[0][0]);

    cl::Event event;
    clip(queue, dInVertices, dInVertexKeys, dInIndices,
         in.vertices.size(), in.numInternalVertices, in.triangles.size() * 3, &event);
    event.wait();

    CPPUNIT_ASSERT_EQUAL(expected.vertices.size(), out.vertices.size());
    CPPUNIT_ASSERT_EQUAL(expected.vertexKeys.size(), out.vertexKeys.size());
    CPPUNIT_ASSERT_EQUAL(expected.numInternalVertices, out.numInternalVertices);
    for (size_t i = 0; i < expected.vertices.size(); i++)
    {
        CPPUNIT_ASSERT_EQUAL(expected.vertices[i].s[0], out.vertices[i].s[0]);
        CPPUNIT_ASSERT_EQUAL(expected.vertices[i].s[1], out.vertices[i].s[1]);
        CPPUNIT_ASSERT_EQUAL(expected.vertices[i].s[2], out.vertices[i].s[2]);
        CPPUNIT_ASSERT_EQUAL(expected.vertexKeys[i], out.vertexKeys[i]);
    }
    CPPUNIT_ASSERT_EQUAL(expected.triangles.size(), out.triangles.size());
    for (size_t i = 0; i < expected.triangles.size(); i++)
    {
        for (unsigned int j = 0; j < 3; j++)
            CPPUNIT_ASSERT_EQUAL(expected.triangles[i][j], out.triangles[i][j]);
    }
}
