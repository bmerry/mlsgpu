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
#include "../src/tr1_cstdint.h"
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

void TestClip::distanceFunc(
    const cl::CommandQueue &queue,
    const cl::Buffer &distances,
    const cl::Buffer &vertices,
    std::size_t numVertices,
    const std::vector<cl::Event> *events,
    cl::Event *event,
    float cut)
{

    std::vector<boost::array<cl_float, 3> > hVertices(numVertices);
    std::vector<cl_float> hDistances(numVertices);
    queue.enqueueReadBuffer(vertices, CL_TRUE, 0, numVertices * (3 * sizeof(cl_float)),
                            &hVertices[0][0], events, NULL);
    for (std::size_t i = 0; i < numVertices; i++)
        hDistances[i] = hVertices[i][0] - cut;

    queue.enqueueWriteBuffer(distances, CL_TRUE, 0, numVertices * sizeof(cl_float),
                             &hDistances[0], NULL, event);
}

void TestClip::testCase(int M, int N, int internalRows, int keepCols)
{
    HostKeyMesh in, out, expected;

    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
        {
            boost::array<cl_float, 3> v = {{ j * 20.0f, i * 10.0f, 0.0f }};
            cl_ulong key = UINT64_C(0xCAFEBABE00000000) + (i << 24) + (j << 16);
            in.vertices.push_back(v);
            if (i >= internalRows)
                in.vertexKeys.push_back(key);
            if (j < keepCols)
            {
                expected.vertices.push_back(v);
                if (i >= internalRows)
                   expected.vertexKeys.push_back(key);
            }
        }
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

    DeviceKeyMesh dIn(context, CL_MEM_READ_ONLY,
                      in.vertices.size(),
                      in.vertices.size() - in.vertexKeys.size(),
                      in.triangles.size());
    std::vector<cl::Event> wait;
    {
        cl::Event e;
        queue.enqueueWriteBuffer(dIn.vertices, CL_FALSE,
                                 0, in.vertices.size() * (3 * sizeof(cl_float)),
                                 &in.vertices[0][0], NULL, &e);
        wait.push_back(e);
    }
    if (!in.vertexKeys.empty())
    {
        cl::Event e;
        queue.enqueueWriteBuffer(dIn.vertexKeys, CL_FALSE,
                                 dIn.numInternalVertices * sizeof(cl_ulong),
                                 in.vertexKeys.size() * sizeof(cl_ulong),
                                 &in.vertexKeys[0], NULL, &e);
        wait.push_back(e);
    }
    {
        cl::Event e;
        queue.enqueueWriteBuffer(dIn.triangles, CL_FALSE,
                                 0, dIn.numTriangles * (3 * sizeof(cl_uint)),
                                 &in.triangles[0][0], NULL, &e);
        wait.push_back(e);
    }

    std::vector<cl::Event> clipEvent(1);
    std::vector<cl::Event> readEvent(3);
    DeviceKeyMesh dOut;
    clip(queue, dIn, &wait, &clipEvent[0], dOut);
    enqueueReadMesh(queue, dOut, out, &clipEvent, &readEvent[0], &readEvent[1], &readEvent[2]);
    cl::Event::waitForEvents(readEvent);

    CPPUNIT_ASSERT_EQUAL(expected.vertices.size(), out.vertices.size());
    for (size_t i = 0; i < expected.vertices.size(); i++)
    {
        CPPUNIT_ASSERT_EQUAL(expected.vertices[i][0], out.vertices[i][0]);
        CPPUNIT_ASSERT_EQUAL(expected.vertices[i][1], out.vertices[i][1]);
        CPPUNIT_ASSERT_EQUAL(expected.vertices[i][2], out.vertices[i][2]);
    }

    CPPUNIT_ASSERT_EQUAL(expected.vertexKeys.size(), out.vertexKeys.size());
    for (size_t i = 0; i < expected.vertexKeys.size(); i++)
    {
        CPPUNIT_ASSERT_EQUAL(expected.vertexKeys[i], out.vertexKeys[i]);
    }

    CPPUNIT_ASSERT_EQUAL(expected.triangles.size(), out.triangles.size());
    for (size_t i = 0; i < expected.triangles.size(); i++)
    {
        for (unsigned int j = 0; j < 3; j++)
            CPPUNIT_ASSERT_EQUAL(expected.triangles[i][j], out.triangles[i][j]);
    }
}
