/**
 * @file
 *
 * Tests for @ref Serialize namespace.
 */
#if HAVE_CONFIG_H
# include <config.h>
#endif

#ifndef __CL_ENABLE_EXCEPTIONS
# define __CL_ENABLE_EXCEPTIONS
#endif

#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/extensions/HelperMacros.h>
#include <boost/thread.hpp>
#include <boost/smart_ptr/scoped_array.hpp>
#include <limits>
#include <vector>
#include <iterator>
#include <algorithm>
#include <mpi.h>
#include "../testutil.h"
#include "../../src/serialize.h"
#include "../../src/grid.h"
#include "../../src/bucket.h"
#include "../../src/mesher.h"
#include "../../src/splat_set.h"
#include "../../src/tr1_cstdint.h"

#define SERIALIZE_TEST(name) \
    CPPUNIT_TEST_SUITE_ADD_TEST( (new GenericTestCaller<TestFixtureType>( \
        context.getTestNameFor(#name), \
        boost::bind(&TestFixtureType::serializeTest, _1, \
                    &TestFixtureType::name ## Send, \
                    &TestFixtureType::name ## Recv), \
        context.makeFixture() ) ) )

class TestSerialize : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestSerialize);
    SERIALIZE_TEST(testGrid);
    SERIALIZE_TEST(testChunkId);
    SERIALIZE_TEST(testSubset);
    SERIALIZE_TEST(testSplats);
    SERIALIZE_TEST(testMesherWork);
    CPPUNIT_TEST_SUITE_END();
private:
    /**
     * Test driver. One process runs the sender method while the root
     * process runs the receiver. The receiver does the assertions.
     */
    void serializeTest(
        void (TestSerialize::* sender)(MPI_Comm, int),
        void (TestSerialize::* receiver)(MPI_Comm, int));

    void testGridSend(MPI_Comm comm, int dest);
    void testGridRecv(MPI_Comm comm, int source);
    void testChunkIdSend(MPI_Comm comm, int dest);
    void testChunkIdRecv(MPI_Comm comm, int source);
    void testSubsetSend(MPI_Comm comm, int dest);
    void testSubsetRecv(MPI_Comm comm, int source);
    void testSplatsSend(MPI_Comm comm, int dest);
    void testSplatsRecv(MPI_Comm comm, int source);
    void testMesherWorkSend(MPI_Comm comm, int dest);
    void testMesherWorkRecv(MPI_Comm comm, int source);
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestSerialize, TestSet::perBuild());

void TestSerialize::serializeTest(
    void (TestSerialize::* sender)(MPI_Comm, int),
    void (TestSerialize::* receiver)(MPI_Comm, int))
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 1)
        (this->*sender)(MPI_COMM_WORLD, 0);
    else if (rank == 0)
        (this->*receiver)(MPI_COMM_WORLD, 1);
}

void TestSerialize::testGridSend(MPI_Comm comm, int dest)
{
    const float ref[3] = {1.0f, -2.2f, 3.141f};
    Grid g(ref, 2.5f, -1, 100, -1000000000, 1000000000, 50, 52);
    Serialize::send(g, comm, dest);
}

void TestSerialize::testGridRecv(MPI_Comm comm, int source)
{
    Grid g;
    Serialize::recv(g, comm, source);
    MLSGPU_ASSERT_EQUAL(1.0f, g.getReference()[0]);
    MLSGPU_ASSERT_EQUAL(-2.2f, g.getReference()[1]);
    MLSGPU_ASSERT_EQUAL(3.141f, g.getReference()[2]);
    MLSGPU_ASSERT_EQUAL(2.5f, g.getSpacing());
    MLSGPU_ASSERT_EQUAL(-1, g.getExtent(0).first);
    MLSGPU_ASSERT_EQUAL(100, g.getExtent(0).second);
    MLSGPU_ASSERT_EQUAL(-1000000000, g.getExtent(1).first);
    MLSGPU_ASSERT_EQUAL(1000000000, g.getExtent(1).second);
    MLSGPU_ASSERT_EQUAL(50, g.getExtent(2).first);
    MLSGPU_ASSERT_EQUAL(52, g.getExtent(2).second);
}

void TestSerialize::testChunkIdSend(MPI_Comm comm, int dest)
{
    ChunkId chunkId;

    chunkId.gen = 12345;
    chunkId.coords[0] = 234;
    chunkId.coords[1] = 0;
    chunkId.coords[2] = std::numeric_limits<Grid::size_type>::max();

    Serialize::send(chunkId, comm, dest);
}

void TestSerialize::testChunkIdRecv(MPI_Comm comm, int source)
{
    ChunkId chunkId;

    Serialize::recv(chunkId, comm, source);

    MLSGPU_ASSERT_EQUAL(12345, chunkId.gen);
    MLSGPU_ASSERT_EQUAL(234, chunkId.coords[0]);
    MLSGPU_ASSERT_EQUAL(0, chunkId.coords[1]);
    MLSGPU_ASSERT_EQUAL(std::numeric_limits<Grid::size_type>::max(), chunkId.coords[2]);
}

void TestSerialize::testSubsetSend(MPI_Comm comm, int dest)
{
    SplatSet::SubsetBase subset;

    subset.addRange(1, 5);
    subset.addRange(5, 10);
    subset.addRange(15, 20);
    subset.addRange(5000, UINT64_C(1000000000000));
    subset.flush();

    Serialize::send(subset, comm, dest);
}

void TestSerialize::testSubsetRecv(MPI_Comm comm, int source)
{
    SplatSet::SubsetBase subset;

    Serialize::recv(subset, comm, source);

    MLSGPU_ASSERT_EQUAL(3, subset.numRanges());
    MLSGPU_ASSERT_EQUAL(UINT64_C(999999995014), subset.numSplats());

    std::vector<std::pair<SplatSet::splat_id, SplatSet::splat_id> > ranges;
    std::copy(subset.begin(), subset.end(), std::back_inserter(ranges));
    MLSGPU_ASSERT_EQUAL(3, ranges.size());
    MLSGPU_ASSERT_EQUAL(1, ranges[0].first);
    MLSGPU_ASSERT_EQUAL(10, ranges[0].second);
    MLSGPU_ASSERT_EQUAL(15, ranges[1].first);
    MLSGPU_ASSERT_EQUAL(20, ranges[1].second);
    MLSGPU_ASSERT_EQUAL(5000, ranges[2].first);
    MLSGPU_ASSERT_EQUAL(UINT64_C(1000000000000), ranges[2].second);
}

/**
 * Create a splat.
 */
static Splat makeSplat(float x, float y, float z, float radius, float nx, float ny, float nz, float quality)
{
    Splat splat;
    splat.position[0] = x;
    splat.position[1] = y;
    splat.position[2] = z;
    splat.radius = radius;
    splat.normal[0] = nx;
    splat.normal[1] = ny;
    splat.normal[2] = nz;
    splat.quality = quality;
    return splat;
}

void TestSerialize::testSplatsSend(MPI_Comm comm, int dest)
{
    Splat splats[2] =
    {
        makeSplat(1.0f, 2.2f, 4.1f, 0.5f, -0.3f, -0.4f, 0.5f, 1000.0f),
        makeSplat(0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f)
    };
    Serialize::send(splats, 2, comm, dest);
}

void TestSerialize::testSplatsRecv(MPI_Comm comm, int source)
{
    Splat splats[2];

    Serialize::recv(splats, 2, comm, source);

    MLSGPU_ASSERT_EQUAL(1.0f, splats[0].position[0]);
    MLSGPU_ASSERT_EQUAL(2.2f, splats[0].position[1]);
    MLSGPU_ASSERT_EQUAL(4.1f, splats[0].position[2]);
    MLSGPU_ASSERT_EQUAL(0.5f, splats[0].radius);
    MLSGPU_ASSERT_EQUAL(-0.3f, splats[0].normal[0]);
    MLSGPU_ASSERT_EQUAL(-0.4f, splats[0].normal[1]);
    MLSGPU_ASSERT_EQUAL(0.5f, splats[0].normal[2]);
    MLSGPU_ASSERT_EQUAL(1000.0f, splats[0].quality);
    MLSGPU_ASSERT_EQUAL(0.1f, splats[1].position[0]);
    MLSGPU_ASSERT_EQUAL(0.2f, splats[1].position[1]);
    MLSGPU_ASSERT_EQUAL(0.3f, splats[1].position[2]);
    MLSGPU_ASSERT_EQUAL(0.4f, splats[1].radius);
    MLSGPU_ASSERT_EQUAL(0.5f, splats[1].normal[0]);
    MLSGPU_ASSERT_EQUAL(0.6f, splats[1].normal[1]);
    MLSGPU_ASSERT_EQUAL(0.7f, splats[1].normal[2]);
    MLSGPU_ASSERT_EQUAL(0.8f, splats[1].quality);
}

void TestSerialize::testMesherWorkSend(MPI_Comm comm, int dest)
{
    // TODO: also need to test the interaction with events. But I'm not sure
    // the test framework will handle CL very well yet.
    MesherWork work;
    MeshSizes sizes(3, 2, 1);
    boost::scoped_array<char> buffer(new char[sizes.getHostBytes()]);

    work.chunkId.gen = 12345;
    work.chunkId.coords[0] = 567;
    work.chunkId.coords[1] = 678;
    work.chunkId.coords[2] = 789;

    work.mesh = HostKeyMesh(buffer.get(), sizes);

    work.mesh.vertices[0][0] = 0.1f;
    work.mesh.vertices[0][1] = -0.2f;
    work.mesh.vertices[0][2] = 0.3f;
    work.mesh.vertices[1][0] = 1.1f;
    work.mesh.vertices[1][1] = -1.2f;
    work.mesh.vertices[1][2] = 1.3f;
    work.mesh.vertices[2][0] = 2.1f;
    work.mesh.vertices[2][1] = -2.2f;
    work.mesh.vertices[2][2] = 2.3f;

    work.mesh.triangles[0][0] = 123;
    work.mesh.triangles[0][1] = 234;
    work.mesh.triangles[0][2] = 345;
    work.mesh.triangles[1][0] = 0;
    work.mesh.triangles[1][1] = 0xFFFFFFFFu;
    work.mesh.triangles[1][2] = 0xFEDCBA98u;

    work.mesh.vertexKeys[0] = UINT64_C(0x1234567823456789);
    work.mesh.vertexKeys[1] = UINT64_C(0xFFFFFFFF11111111);

    work.hasEvents = false;

    Serialize::send(work, comm, dest);
}

void TestSerialize::testMesherWorkRecv(MPI_Comm comm, int source)
{
    MesherWork work;

    MeshSizes expectedSizes(3, 2, 1);
    boost::scoped_array<char> buffer(new char[expectedSizes.getHostBytes()]);

    Serialize::recv(work, buffer.get(), comm, source);

    MLSGPU_ASSERT_EQUAL(12345, work.chunkId.gen);
    MLSGPU_ASSERT_EQUAL(567, work.chunkId.coords[0]);
    MLSGPU_ASSERT_EQUAL(678, work.chunkId.coords[1]);
    MLSGPU_ASSERT_EQUAL(789, work.chunkId.coords[2]);

    MLSGPU_ASSERT_EQUAL(3, work.mesh.numVertices());
    MLSGPU_ASSERT_EQUAL(0.1f, work.mesh.vertices[0][0]);
    MLSGPU_ASSERT_EQUAL(-0.2f, work.mesh.vertices[0][1]);
    MLSGPU_ASSERT_EQUAL(0.3f, work.mesh.vertices[0][2]);
    MLSGPU_ASSERT_EQUAL(1.1f, work.mesh.vertices[1][0]);
    MLSGPU_ASSERT_EQUAL(-1.2f, work.mesh.vertices[1][1]);
    MLSGPU_ASSERT_EQUAL(1.3f, work.mesh.vertices[1][2]);
    MLSGPU_ASSERT_EQUAL(2.1f, work.mesh.vertices[2][0]);
    MLSGPU_ASSERT_EQUAL(-2.2f, work.mesh.vertices[2][1]);
    MLSGPU_ASSERT_EQUAL(2.3f, work.mesh.vertices[2][2]);

    MLSGPU_ASSERT_EQUAL(2, work.mesh.numTriangles());
    MLSGPU_ASSERT_EQUAL(123, work.mesh.triangles[0][0]);
    MLSGPU_ASSERT_EQUAL(234, work.mesh.triangles[0][1]);
    MLSGPU_ASSERT_EQUAL(345, work.mesh.triangles[0][2]);
    MLSGPU_ASSERT_EQUAL(0, work.mesh.triangles[1][0]);
    MLSGPU_ASSERT_EQUAL(0xFFFFFFFFu, work.mesh.triangles[1][1]);
    MLSGPU_ASSERT_EQUAL(0xFEDCBA98u, work.mesh.triangles[1][2]);

    MLSGPU_ASSERT_EQUAL(2, work.mesh.numExternalVertices());
    MLSGPU_ASSERT_EQUAL(UINT64_C(0x1234567823456789), work.mesh.vertexKeys[0]);
    MLSGPU_ASSERT_EQUAL(UINT64_C(0xFFFFFFFF11111111), work.mesh.vertexKeys[1]);

    MLSGPU_ASSERT_EQUAL(false, work.hasEvents);
}
