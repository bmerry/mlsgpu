/*
 * mlsgpu: surface reconstruction from point clouds
 * Copyright (C) 2013  University of Cape Town
 *
 * This file is part of mlsgpu.
 *
 * mlsgpu is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/**
 * @file
 *
 * Tests for @ref MeshFilterChain.
 */

#ifndef __CL_ENABLE_EXCEPTIONS
# define __CL_ENABLE_EXCEPTIONS
#endif

#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/extensions/HelperMacros.h>
#include <vector>
#include <boost/ref.hpp>
#include <boost/smart_ptr/scoped_array.hpp>
#include <CL/cl.hpp>
#include "testutil.h"
#include "test_clh.h"
#include "../src/mesh.h"
#include "../src/mesh_filter.h"

/**
 * A mesh filter created specifically for the test. It truncates one triangle
 * from the beginning, thus potentially leaving an empty triangle list.
 */
class TruncateFilter
{
private:
    DeviceKeyMesh out;
public:
    TruncateFilter(const cl::Context &context, std::size_t maxVertices, std::size_t maxTriangles);

    void operator()(
        const cl::CommandQueue &queue,
        const DeviceKeyMesh &inMesh,
        const std::vector<cl::Event> *events,
        cl::Event *event,
        DeviceKeyMesh &outMesh) const;
};

TruncateFilter::TruncateFilter(const cl::Context &context, std::size_t maxVertices, std::size_t maxTriangles)
    : out(context, CL_MEM_READ_WRITE, MeshSizes(maxVertices, maxTriangles, 0))
{
}

void TruncateFilter::operator()(
    const cl::CommandQueue &queue,
    const DeviceKeyMesh &inMesh,
    const std::vector<cl::Event> *events,
    cl::Event *event,
    DeviceKeyMesh &outMesh) const
{
    CPPUNIT_ASSERT(&inMesh != &outMesh);
    CPPUNIT_ASSERT(inMesh.vertices());
    CPPUNIT_ASSERT(inMesh.vertexKeys());
    CPPUNIT_ASSERT(inMesh.triangles());
    CPPUNIT_ASSERT(inMesh.vertices() != out.vertices());
    CPPUNIT_ASSERT(inMesh.vertexKeys() != out.vertexKeys());
    CPPUNIT_ASSERT(inMesh.triangles() != out.triangles());
    CPPUNIT_ASSERT(inMesh.numVertices() * 3 * sizeof(cl_float) <= out.vertices.getInfo<CL_MEM_SIZE>());
    CPPUNIT_ASSERT(inMesh.numTriangles() * 3 * sizeof(cl_uint) <= out.triangles.getInfo<CL_MEM_SIZE>());

    outMesh = out;
    cl::Event verticesEvent, vertexKeysEvent, trianglesEvent;
    std::vector<cl::Event> wait;

    CLH::enqueueCopyBuffer(queue,
                           inMesh.vertices, outMesh.vertices, 0, 0,
                           inMesh.numVertices() * (3 * sizeof(cl_float)),
                           events, &verticesEvent);
    wait.push_back(verticesEvent);

    CLH::enqueueCopyBuffer(queue,
                           inMesh.vertexKeys, outMesh.vertexKeys, 0, 0,
                           inMesh.numVertices() * sizeof(cl_ulong),
                           events, &vertexKeysEvent);
    wait.push_back(vertexKeysEvent);

    if (inMesh.numTriangles() > 0)
    {
        CLH::enqueueCopyBuffer(queue,
                               inMesh.triangles, outMesh.triangles,
                               3 * sizeof(cl_uint), 0,
                               (inMesh.numTriangles() - 1) * (3 * sizeof(cl_uint)),
                               events, &trianglesEvent);
        wait.push_back(trianglesEvent);
    }

    outMesh.assign(
        inMesh.numVertices(),
        inMesh.numTriangles() ? inMesh.numTriangles() - 1 : 0,
        inMesh.numInternalVertices());
    CLH::enqueueMarkerWithWaitList(queue, &wait, event);
}

/**
 * Marching output class that reads the result into a host mesh. If the
 * provided host mesh is NULL, it asserts that it is never called.
 */
class CollectOutput
{
private:
    HostKeyMesh *hMesh;
public:
    CollectOutput() : hMesh(NULL) {}
    CollectOutput(HostKeyMesh *hMesh) : hMesh(hMesh) {}

    void operator()(
        const cl::CommandQueue &queue,
        const DeviceKeyMesh &dMesh,
        const std::vector<cl::Event> *events,
        cl::Event *event);

    void setHostMesh(HostKeyMesh *hMesh) { this->hMesh = hMesh; }
};

void CollectOutput::operator()(
        const cl::CommandQueue &queue,
        const DeviceKeyMesh &dMesh,
        const std::vector<cl::Event> *events,
        cl::Event *event)
{
    CPPUNIT_ASSERT(hMesh != NULL);
    // Resize the mesh. The truncate filter only removes data, so there is no
    // need to worry about reallocating the backing storage
    hMesh->assign(dMesh.numVertices(), dMesh.numTriangles(), dMesh.numInternalVertices());
    std::vector<cl::Event> wait(3);
    enqueueReadMesh(queue, dMesh, *hMesh, events, &wait[0], &wait[1], &wait[2]);
    CLH::enqueueMarkerWithWaitList(queue, &wait, event);
}

class TestMeshFilterChain : public CLH::Test::TestFixture
{
    CPPUNIT_TEST_SUITE(TestMeshFilterChain);
    CPPUNIT_TEST(testNoFilters);
    CPPUNIT_TEST(testFilters);
    CPPUNIT_TEST(testCull);
    CPPUNIT_TEST(testEmpty);
    CPPUNIT_TEST_SUITE_END();

private:
    DeviceKeyMesh dMesh;
    HostKeyMesh hMesh;
    MeshFilterChain filterChain;

    boost::scoped_array<char> hMeshBuffer; ///< backing store for @c hMesh

public:
    virtual void setUp();     ///< Creates a device mesh with some data
    virtual void tearDown();  ///< Destroy the meshes

    void testNoFilters();     ///< Test basic operation with no filters in the chain
    void testFilters();       ///< Test normal operation with filters in the chain
    void testCull();          ///< Test case where a filter completely eliminates the mesh
    void testEmpty();         ///< Passes an empty mesh into the front end
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestMeshFilterChain, TestSet::perBuild());

void TestMeshFilterChain::setUp()
{
    CLH::Test::TestFixture::setUp();

    dMesh.assign(5, 3, 4);
    dMesh.vertices = createBuffer(CL_MEM_READ_WRITE, dMesh.numVertices() * 3 * sizeof(cl_float));
    dMesh.vertexKeys = createBuffer(CL_MEM_READ_WRITE, dMesh.numVertices() * sizeof(cl_ulong));
    dMesh.triangles = createBuffer(CL_MEM_READ_WRITE, dMesh.numTriangles() * 3 * sizeof(cl_uint));

    hMeshBuffer.reset(new char[dMesh.getHostBytes()]);
    hMesh = HostKeyMesh(hMeshBuffer.get(), dMesh);

    filterChain.setOutput(CollectOutput(&hMesh));
}

void TestMeshFilterChain::tearDown()
{
    hMeshBuffer.reset();
    dMesh.vertices = NULL;
    dMesh.vertexKeys = NULL;
    dMesh.triangles = NULL;
    filterChain = MeshFilterChain();
    CLH::Test::TestFixture::tearDown();
}

void TestMeshFilterChain::testNoFilters()
{
    cl::Event event;
    filterChain(queue, dMesh, NULL, &event);
    queue.flush();
    event.wait();

    MLSGPU_ASSERT_EQUAL(5, hMesh.numVertices());
    MLSGPU_ASSERT_EQUAL(4, hMesh.numInternalVertices());
    MLSGPU_ASSERT_EQUAL(3, hMesh.numTriangles());
}

void TestMeshFilterChain::testFilters()
{
    TruncateFilter filter1(context, 10, 10);
    TruncateFilter filter2(context, 10, 10);
    filterChain.addFilter(boost::ref(filter1));
    filterChain.addFilter(boost::ref(filter2));

    cl::Event event;
    filterChain(queue, dMesh, NULL, &event);
    queue.flush();
    event.wait();

    MLSGPU_ASSERT_EQUAL(5, hMesh.numVertices());
    MLSGPU_ASSERT_EQUAL(4, hMesh.numInternalVertices());
    MLSGPU_ASSERT_EQUAL(1, hMesh.numTriangles());
}

void TestMeshFilterChain::testCull()
{
    TruncateFilter filter1(context, 10, 10);
    TruncateFilter filter2(context, 10, 10);
    filterChain.addFilter(boost::ref(filter1));
    filterChain.addFilter(boost::ref(filter2));
    filterChain.addFilter(boost::ref(filter1));

    cl::Event event;
    filterChain(queue, dMesh, NULL, &event);
    queue.flush();
    event.wait();

    MLSGPU_ASSERT_EQUAL(5, hMesh.numVertices());
    MLSGPU_ASSERT_EQUAL(1, hMesh.numExternalVertices());
    MLSGPU_ASSERT_EQUAL(0, hMesh.numTriangles());
}

void TestMeshFilterChain::testEmpty()
{
    TruncateFilter filter1(context, 10, 10);
    TruncateFilter filter2(context, 10, 10);
    filterChain.addFilter(boost::ref(filter1));
    filterChain.addFilter(boost::ref(filter2));
    filterChain.addFilter(boost::ref(filter1));

    dMesh.assign(0, 0, 0);

    cl::Event event;
    filterChain(queue, dMesh, NULL, &event);
    queue.flush();
    event.wait();

    MLSGPU_ASSERT_EQUAL(0, hMesh.numVertices());
    MLSGPU_ASSERT_EQUAL(0, hMesh.numInternalVertices());
    MLSGPU_ASSERT_EQUAL(0, hMesh.numTriangles());
}

class TestScaleBiasFilter : public CLH::Test::TestFixture
{
    CPPUNIT_TEST_SUITE(TestScaleBiasFilter);
    CPPUNIT_TEST(testSimple);
    CPPUNIT_TEST(testEmpty);
    CPPUNIT_TEST_SUITE_END();

private:
    void testSimple();
    void testEmpty();
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestScaleBiasFilter, TestSet::perCommit());

void TestScaleBiasFilter::testSimple()
{
    ScaleBiasFilter filter(context);
    filter.setScaleBias(3.0f, 10.0f, -20.0f, 30.0f);

    const unsigned int N = 5;
    const unsigned int T = 2;
    std::vector<boost::array<cl_float, 3> > inVertices(N);
    std::vector<boost::array<cl_uint, 3> > inTriangles(T);
    std::vector<cl_ulong> inVertexKeys(N);

    inVertices[0][0] = 1.0f;    inVertices[0][1] = 2.0f;   inVertices[0][2] = 3.0f;
    inVertices[1][0] = 0.0f;    inVertices[1][1] = 0.0f;   inVertices[1][2] = 0.0f;
    inVertices[2][0] = -100.0f; inVertices[2][1] = 100.0f; inVertices[2][2] = -150.0f;
    inVertices[3][0] = 0.01f;   inVertices[3][1] = 5.0f;   inVertices[3][2] = -0.5f;
    inVertices[4][0] = 123.4f;  inVertices[4][1] = 456.7f; inVertices[4][2] = 890.1f;

    inTriangles[0][0] = 0; inTriangles[0][1] = 1; inTriangles[0][2] = 2;
    inTriangles[1][0] = 2; inTriangles[1][1] = 1; inTriangles[1][2] = 3;

    inVertexKeys[0] = 0xDEADBEEF;
    inVertexKeys[1] = 0xCAFEBABE;
    inVertexKeys[2] = 0xBABECAFE;
    inVertexKeys[3] = 0xBEEFDEAD;
    inVertexKeys[4] = 0xABCDEF01;

    DeviceKeyMesh inMesh;
    inMesh.assign(N, T, 0);
    inMesh.vertices = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N * 3 * sizeof(cl_float),
                                 &inVertices[0][0]);
    inMesh.triangles = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, T * 3 * sizeof(cl_uint),
                                  &inTriangles[0][0]);
    inMesh.vertexKeys = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N * sizeof(cl_ulong),
                                   &inVertexKeys[0]);

    DeviceKeyMesh outMesh;
    std::vector<cl::Event> wait(1);
    std::vector<cl::Event> readWait(3);
    filter(queue, inMesh, NULL, &wait[0], outMesh);

    boost::scoped_array<char> buffer(new char[outMesh.getHostBytes()]);
    HostKeyMesh result(buffer.get(), outMesh);
    enqueueReadMesh(queue, outMesh, result, &wait, &readWait[0], &readWait[1], &readWait[2]);
    cl::Event::waitForEvents(readWait);

    MLSGPU_ASSERT_EQUAL(N, result.numVertices());
    MLSGPU_ASSERT_EQUAL(N, result.numExternalVertices());
    MLSGPU_ASSERT_EQUAL(T, result.numTriangles());
    for (unsigned int i = 0; i < N; i++)
    {
        CPPUNIT_ASSERT_DOUBLES_EQUAL(inVertices[i][0] * 3.0f + 10.0f, result.vertices[i][0], 1e-2);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(inVertices[i][1] * 3.0f - 20.0f, result.vertices[i][1], 1e-2);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(inVertices[i][2] * 3.0f + 30.0f, result.vertices[i][2], 1e-2);
        CPPUNIT_ASSERT_EQUAL(inVertexKeys[i], result.vertexKeys[i]);
    }
    for (unsigned int i = 0; i < T; i++)
    {
        CPPUNIT_ASSERT_EQUAL(inTriangles[i][0], result.triangles[i][0]);
        CPPUNIT_ASSERT_EQUAL(inTriangles[i][1], result.triangles[i][1]);
        CPPUNIT_ASSERT_EQUAL(inTriangles[i][2], result.triangles[i][2]);
    }
}

void TestScaleBiasFilter::testEmpty()
{
    ScaleBiasFilter filter(context);
    DeviceKeyMesh inMesh, outMesh;

    inMesh.assign(0, 0, 0);

    HostKeyMesh result;
    cl::Event done;
    filter(queue, inMesh, NULL, &done, outMesh);
    done.wait();
    MLSGPU_ASSERT_EQUAL(0, outMesh.numVertices());
    MLSGPU_ASSERT_EQUAL(0, outMesh.numInternalVertices());
    MLSGPU_ASSERT_EQUAL(0, outMesh.numTriangles());
}
