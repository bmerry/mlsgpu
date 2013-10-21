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
 * Tests for @ref mesh.h.
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
#include <boost/array.hpp>
#include <boost/smart_ptr/scoped_array.hpp>
#include <CL/cl.hpp>
#include "testutil.h"
#include "test_clh.h"
#include "../src/mesh.h"

/**
 * Tests construction of a @ref DeviceKeyMesh.
 */
class TestDeviceKeyMesh : public CLH::Test::TestFixture
{
    CPPUNIT_TEST_SUITE(TestDeviceKeyMesh);
    CPPUNIT_TEST(testConstructor);
    CPPUNIT_TEST_SUITE_END();
public:
    void testConstructor();   ///< Test that the constructor creates the buffers
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestDeviceKeyMesh, TestSet::perBuild());

void TestDeviceKeyMesh::testConstructor()
{
    DeviceKeyMesh mesh(context, CL_MEM_READ_WRITE, MeshSizes(10, 20, 5));
    CPPUNIT_ASSERT(mesh.vertices());
    CPPUNIT_ASSERT(mesh.vertexKeys());
    CPPUNIT_ASSERT(mesh.triangles());
    CPPUNIT_ASSERT_EQUAL(std::size_t(10), mesh.numVertices());
    CPPUNIT_ASSERT_EQUAL(std::size_t(5), mesh.numInternalVertices());
    CPPUNIT_ASSERT_EQUAL(std::size_t(20), mesh.numTriangles());
    CPPUNIT_ASSERT_EQUAL(10 * 3 * sizeof(cl_float), mesh.vertices.getInfo<CL_MEM_SIZE>());
    CPPUNIT_ASSERT_EQUAL(10 * sizeof(cl_ulong), mesh.vertexKeys.getInfo<CL_MEM_SIZE>());
    CPPUNIT_ASSERT_EQUAL(20 * 3 * sizeof(cl_uint), mesh.triangles.getInfo<CL_MEM_SIZE>());
}

/**
 * Test @ref enqueueReadMesh.
 */
class TestEnqueueReadMesh : public CLH::Test::TestFixture
{
    CPPUNIT_TEST_SUITE(TestEnqueueReadMesh);
    CPPUNIT_TEST(testSimple);
    CPPUNIT_TEST(testZeroInternal);
    CPPUNIT_TEST(testSkipVertices);
    CPPUNIT_TEST(testSkipVertexKeys);
    CPPUNIT_TEST(testSkipTriangles);
    CPPUNIT_TEST_SUITE_END();
private:
    /**
     * Device mesh with some data.
     */
    DeviceKeyMesh dMesh;

    /**
     * Host mesh with some data different to @ref dMesh, to test that it is
     * overwritten correctly.
     */
    HostKeyMesh hMesh;

    /**
     * Backing store for @ref hMesh.
     */
    boost::scoped_array<char> hMeshBuffer;

    std::vector<boost::array<cl_float, 3> > expectedVertices;
    std::vector<cl_ulong> expectedVertexKeys;
    std::vector<boost::array<cl_uint, 3> > expectedTriangles;

    void validateVertices(const HostKeyMesh &hMesh);
    void validateVertexKeys(const HostKeyMesh &hMesh, std::size_t numInternalVertices);
    void validateTriangles(const HostKeyMesh &hMesh);

public:
    virtual void setUp();

    void testSimple();          ///< Test normal operation with all properties
    void testZeroInternal();    ///< Test case where <code>numInternalVertices == 0</code>.
    void testSkipVertices();    ///< Test skipping transfer of vertices.
    void testSkipVertexKeys();  ///< Test skipping transfer of keys.
    void testSkipTriangles();   ///< Test skipping transfer of triangles.
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestEnqueueReadMesh, TestSet::perBuild());

void TestEnqueueReadMesh::setUp()
{
    CLH::Test::TestFixture::setUp();

    expectedVertices.resize(5);
    expectedVertexKeys.resize(5);
    expectedTriangles.resize(7);

    // some arbitrary values
    for (unsigned int i = 0; i < 5; i++)
    {
        expectedVertices[i][0] = i;
        expectedVertices[i][1] = 2.0f * i + 7.5f;
        expectedVertices[i][2] = i * i - 10.0f;
        expectedVertexKeys[i] = cl_ulong(0xDEADBEEF) * 0x1000 + i;
    }
    for (unsigned int i = 0; i < 7; i++)
    {
        expectedTriangles[i][0] = i % 5;
        expectedTriangles[i][1] = (2 * i + 1) % 5;
        expectedTriangles[i][2] = (3 * i * i + 2) % 5;
    }

    dMesh.assign(expectedVertices.size(), expectedTriangles.size(), 2);

    dMesh.vertices = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                dMesh.numVertices() * (3 * sizeof(cl_float)),
                                &expectedVertices[0][0]);
    dMesh.vertexKeys = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  dMesh.numVertices() * sizeof(cl_ulong),
                                  &expectedVertexKeys[0]);
    dMesh.triangles = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 dMesh.numTriangles() * (3 * sizeof(cl_uint)),
                                 &expectedTriangles[0][0]);

    hMeshBuffer.reset(new char[dMesh.getHostBytes()]);
    hMesh = HostKeyMesh(hMeshBuffer.get(), dMesh);
}

void TestEnqueueReadMesh::validateVertices(const HostKeyMesh &hMesh)
{
    CPPUNIT_ASSERT_EQUAL(expectedVertices.size(), hMesh.numVertices());
    for (std::size_t i = 0; i < expectedVertices.size(); i++)
    {
        CPPUNIT_ASSERT_EQUAL(expectedVertices[i][0], hMesh.vertices[i][0]);
        CPPUNIT_ASSERT_EQUAL(expectedVertices[i][1], hMesh.vertices[i][1]);
        CPPUNIT_ASSERT_EQUAL(expectedVertices[i][2], hMesh.vertices[i][2]);
    }
}

void TestEnqueueReadMesh::validateVertexKeys(const HostKeyMesh &hMesh, std::size_t numInternalVertices)
{
    CPPUNIT_ASSERT_EQUAL(numInternalVertices, hMesh.numInternalVertices());
    for (std::size_t i = numInternalVertices; i < expectedVertexKeys.size(); i++)
    {
        CPPUNIT_ASSERT_EQUAL(expectedVertexKeys[i], hMesh.vertexKeys[i - dMesh.numInternalVertices()]);
    }
}

void TestEnqueueReadMesh::validateTriangles(const HostKeyMesh &hMesh)
{
    CPPUNIT_ASSERT_EQUAL(expectedTriangles.size(), hMesh.numTriangles());
    for (std::size_t i = 0; i < expectedTriangles.size(); i++)
    {
        CPPUNIT_ASSERT_EQUAL(expectedTriangles[i][0], hMesh.triangles[i][0]);
        CPPUNIT_ASSERT_EQUAL(expectedTriangles[i][1], hMesh.triangles[i][1]);
        CPPUNIT_ASSERT_EQUAL(expectedTriangles[i][2], hMesh.triangles[i][2]);
    }
}

void TestEnqueueReadMesh::testSimple()
{
    cl::Event verticesEvent, vertexKeysEvent, trianglesEvent;
    enqueueReadMesh(queue, dMesh, hMesh, NULL, &verticesEvent, &vertexKeysEvent, &trianglesEvent);

    verticesEvent.wait();
    validateVertices(hMesh);
    vertexKeysEvent.wait();
    validateVertexKeys(hMesh, dMesh.numInternalVertices());
    trianglesEvent.wait();
    validateTriangles(hMesh);
}

void TestEnqueueReadMesh::testZeroInternal()
{
    cl::Event verticesEvent, vertexKeysEvent, trianglesEvent;
    dMesh.assign(dMesh.numVertices(), dMesh.numTriangles(), 0); // set internal vertices to 0

    // Need more memory to back this up
    hMeshBuffer.reset(new char[dMesh.getHostBytes()]);
    hMesh = HostKeyMesh(hMeshBuffer.get(), dMesh);

    enqueueReadMesh(queue, dMesh, hMesh, NULL, &verticesEvent, &vertexKeysEvent, &trianglesEvent);

    verticesEvent.wait();
    validateVertices(hMesh);
    vertexKeysEvent.wait();
    validateVertexKeys(hMesh, dMesh.numInternalVertices());
    trianglesEvent.wait();
    validateTriangles(hMesh);
}

void TestEnqueueReadMesh::testSkipVertices()
{
    cl::Event vertexKeysEvent, trianglesEvent;
    enqueueReadMesh(queue, dMesh, hMesh, NULL, NULL, &vertexKeysEvent, &trianglesEvent);

    vertexKeysEvent.wait();
    validateVertexKeys(hMesh, dMesh.numInternalVertices());
    trianglesEvent.wait();
    validateTriangles(hMesh);
}

void TestEnqueueReadMesh::testSkipVertexKeys()
{
    cl::Event verticesEvent, trianglesEvent;
    enqueueReadMesh(queue, dMesh, hMesh, NULL, &verticesEvent, NULL, &trianglesEvent);

    verticesEvent.wait();
    validateVertices(hMesh);
    trianglesEvent.wait();
    validateTriangles(hMesh);
}

void TestEnqueueReadMesh::testSkipTriangles()
{
    cl::Event verticesEvent, vertexKeysEvent;
    enqueueReadMesh(queue, dMesh, hMesh, NULL, &verticesEvent, &vertexKeysEvent, NULL);

    verticesEvent.wait();
    validateVertices(hMesh);
    vertexKeysEvent.wait();
    validateVertexKeys(hMesh, dMesh.numInternalVertices());
}
