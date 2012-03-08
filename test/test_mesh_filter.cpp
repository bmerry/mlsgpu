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
#include <CL/cl.hpp>
#include "testmain.h"
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
    : out(context, CL_MEM_READ_WRITE, maxVertices, 0, maxTriangles)
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
    CPPUNIT_ASSERT(inMesh.numVertices * 3 * sizeof(cl_float) <= out.vertices.getInfo<CL_MEM_SIZE>());
    CPPUNIT_ASSERT(inMesh.numTriangles * 3 * sizeof(cl_uint) <= out.triangles.getInfo<CL_MEM_SIZE>());

    outMesh = out;
    cl::Event verticesEvent, vertexKeysEvent, trianglesEvent;
    std::vector<cl::Event> wait;

    CLH::enqueueCopyBuffer(queue,
                           inMesh.vertices, outMesh.vertices, 0, 0,
                           inMesh.numVertices * (3 * sizeof(cl_float)),
                           events, &verticesEvent);
    wait.push_back(verticesEvent);

    CLH::enqueueCopyBuffer(queue,
                           inMesh.vertexKeys, outMesh.vertexKeys, 0, 0,
                           inMesh.numVertices * sizeof(cl_ulong),
                           events, &vertexKeysEvent);
    wait.push_back(vertexKeysEvent);

    if (inMesh.numTriangles > 0)
    {
        CLH::enqueueCopyBuffer(queue,
                               inMesh.triangles, outMesh.triangles,
                               3 * sizeof(cl_uint), 0,
                               (inMesh.numTriangles - 1) * (3 * sizeof(cl_uint)),
                               events, &trianglesEvent);
        wait.push_back(trianglesEvent);
    }

    outMesh.numVertices = inMesh.numVertices;
    outMesh.numInternalVertices = inMesh.numInternalVertices;
    outMesh.numTriangles = inMesh.numTriangles ? inMesh.numTriangles - 1 : 0;
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

public:
    virtual void setUp();     ///< Creates a device mesh with some data

    void testNoFilters();     ///< Test basic operation with no filters in the chain
    void testFilters();       ///< Test normal operation with filters in the chain
    void testCull();          ///< Test case where a filter completely eliminates the mesh
    void testEmpty();         ///< Passes an empty mesh into the front end
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestMeshFilterChain, TestSet::perBuild());

void TestMeshFilterChain::setUp()
{
    CLH::Test::TestFixture::setUp();

    dMesh = DeviceKeyMesh(context, CL_MEM_READ_WRITE, 5, 4, 3);
    // TODO: specify the member contents
    filterChain.setOutput(CollectOutput(&hMesh));
}

void TestMeshFilterChain::testNoFilters()
{
    cl::Event event;
    filterChain(queue, dMesh, NULL, &event);
    queue.flush();
    event.wait();

    CPPUNIT_ASSERT(hMesh.vertices.size() == 5);
    CPPUNIT_ASSERT(hMesh.vertexKeys.size() == 1);
    CPPUNIT_ASSERT(hMesh.triangles.size() == 3);
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

    CPPUNIT_ASSERT(hMesh.vertices.size() == 5);
    CPPUNIT_ASSERT(hMesh.vertexKeys.size() == 1);
    CPPUNIT_ASSERT(hMesh.triangles.size() == 1);
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

    CPPUNIT_ASSERT(hMesh.vertices.size() == 5);
    CPPUNIT_ASSERT(hMesh.vertexKeys.size() == 1);
    CPPUNIT_ASSERT(hMesh.triangles.size() == 0);
}

void TestMeshFilterChain::testEmpty()
{
    TruncateFilter filter1(context, 10, 10);
    TruncateFilter filter2(context, 10, 10);
    filterChain.addFilter(boost::ref(filter1));
    filterChain.addFilter(boost::ref(filter2));
    filterChain.addFilter(boost::ref(filter1));

    dMesh.numTriangles = 0;
    dMesh.numVertices = 0;
    dMesh.numInternalVertices = 0;

    cl::Event event;
    filterChain(queue, dMesh, NULL, &event);
    queue.flush();
    event.wait();

    CPPUNIT_ASSERT(hMesh.vertices.size() == 0);
    CPPUNIT_ASSERT(hMesh.vertexKeys.size() == 0);
    CPPUNIT_ASSERT(hMesh.triangles.size() == 0);
}
