/**
 * @file
 *
 * Test code for @ref mesh.cpp.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif

#ifndef __CL_ENABLE_EXCEPTIONS
# define __CL_ENABLE_EXCEPTIONS
#endif

#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/extensions/HelperMacros.h>
#include <string>
#include <vector>
#include <utility>
#include <stdexcept>
#include <cstring>
#include <tr1/cstdint>
#include <map>
#include <algorithm>
#include <cstddef>
#include <boost/array.hpp>
#include <boost/smart_ptr/scoped_ptr.hpp>
#include <boost/foreach.hpp>
#include <CL/cl.hpp>
#include "testmain.h"
#include "../src/fast_ply.h"
#include "../src/mesh.h"
#include "../src/errors.h"
#include "test_clh.h"

using namespace std;

/**
 * An implementation of the @ref FastPly::WriterBase
 * interface that does not actually write to file, but merely saves
 * a copy of the data in memory. It is aimed specifically at testing.
 */
class MemoryWriter : public FastPly::WriterBase
{
public:
    /// Constructor
    MemoryWriter();

    virtual void open(const std::string &filename);
    virtual std::pair<char *, size_type> open();
    virtual void close();
    virtual void writeVertices(size_type first, size_type count, const float *data);
    virtual void writeTriangles(size_type first, size_type count, const std::tr1::uint32_t *data);
    virtual bool supportsOutOfOrder() const;

    const vector<boost::array<float, 3> > &getVertices() const;
    const vector<boost::array<std::tr1::uint32_t, 3> > &getTriangles() const;

private:
    vector<boost::array<float, 3> > vertices;
    vector<boost::array<std::tr1::uint32_t, 3> > triangles;
};

void MemoryWriter::open(const std::string &filename)
{
    MLSGPU_ASSERT(!isOpen(), std::runtime_error);

    // Ignore the filename
    (void) filename;

    vertices.resize(getNumVertices());
    triangles.resize(getNumTriangles());
    setOpen(true);
}

std::pair<char *, MemoryWriter::size_type> MemoryWriter::open()
{
    MLSGPU_ASSERT(!isOpen(), std::runtime_error);

    vertices.resize(getNumVertices());
    triangles.resize(getNumTriangles());
    setOpen(true);

    return make_pair((char *) NULL, size_type(0));
}

void MemoryWriter::close()
{
    setOpen(false);
}

void MemoryWriter::writeVertices(size_type first, size_type count, const float *data)
{
    MLSGPU_ASSERT(isOpen(), std::runtime_error);
    MLSGPU_ASSERT(first + count <= getNumVertices() && first <= std::numeric_limits<size_type>::max() - count, std::out_of_range);

    std::memcpy(&vertices[first], data, count * 3 * sizeof(float));
}

void MemoryWriter::writeTriangles(size_type first, size_type count, const std::tr1::uint32_t *data)
{
    MLSGPU_ASSERT(isOpen(), std::runtime_error);
    MLSGPU_ASSERT(first + count <= getNumTriangles() && first <= std::numeric_limits<size_type>::max() - count, std::out_of_range);

    std::memcpy(&triangles[first], data, count * 3 * sizeof(std::tr1::uint32_t));
}

bool MemoryWriter::supportsOutOfOrder() const
{
    return true;
}


/**
 * Tests that are shared across all the @ref MeshBase subclasses, including those
 * that don't do welding.
 */
class TestMeshBase : public CLH::Test::TestFixture
{
    CPPUNIT_TEST_SUITE(TestMeshBase);
    CPPUNIT_TEST(testSimple);
    CPPUNIT_TEST(testNoInternal);
    CPPUNIT_TEST(testNoExternal);
    CPPUNIT_TEST_SUITE_END();
private:
    /**
     * Returns a rotation of the triangle to a canonical form.
     */
    boost::array<std::tr1::uint32_t, 3> canonicalTriangle(
        const boost::array<std::tr1::uint32_t, 3> &triangle) const;

protected:
    virtual MeshBase *meshFactory();

    /**
     * Call the output functor with the data provided. This is a convenience
     * function which takes care of loading the data into OpenCL buffers.
     */
    void add(const Marching::OutputFunctor &functor,
             const vector<cl_float3> &vertices,
             const vector<cl_ulong> &vertexKeys,
             const vector<boost::array<cl_uint, 3> > &triangles,
             std::size_t numInternalVertices);

    /**
     * Assert that the mesh produced is isomorphic to the data provided.
     * Is it permitted for the vertices and triangles to have been permuted and
     * for the order of indices in a triangle to have been rotated (but not
     * reflected).
     *
     * @pre The vertices are all unique.
     */
    void checkIsomorphic(const vector<cl_float3> &expectedVertices,
                         const vector<boost::array<cl_uint, 3> > &expectedTriangles,
                         const MemoryWriter &actual) const;

public:
    void testSimple();
    void testNoInternal();
    void testNoExternal();
};

boost::array<std::tr1::uint32_t, 3> TestMeshBase::canonicalTriangle(
    const boost::array<std::tr1::uint32_t, 3> &triangle) const
{
    boost::array<std::tr1::uint32_t, 3> rot[3] =
    {
        triangle,
        {{ triangle[1], triangle[2], triangle[0] }},
        {{ triangle[2], triangle[0], triangle[1] }}
    };
    return *min_element(rot, rot + 3);
}

void TestMeshBase::add(
    const Marching::OutputFunctor &functor,
    const vector<cl_float3> &vertices,
    const vector<cl_ulong> &vertexKeys,
    const vector<boost::array<cl_uint, 3> > &triangles,
    std::size_t numInternalVertices)
{
    cl::Buffer dVertices(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                         vertices.size() * sizeof(vertices[0]), (void *) &vertices[0]);
    cl::Buffer dVertexKeys(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                           vertexKeys.size() * sizeof(vertexKeys[0]), (void *) &vertexKeys[0]);
    cl::Buffer dIndices(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                        triangles.size() * sizeof(triangles[0]), (void *) &triangles[0][0]);

    functor(queue, dVertices, dVertexKeys, dIndices,
            vertices.size(), numInternalVertices, triangles.size() * 3, NULL);
}

void TestMeshBase::checkIsomorphic(
    const vector<cl_float3> &expectedVertices,
    const vector<boost::array<cl_uint, 3> > &expectedTriangles,
    const MemoryWriter &actual) const
{
    const vector<boost::array<float, 3> > &actualVertices = actual.getVertices();
    const vector<boost::array<std::tr1::uint32_t, 3> > &actualTriangles = actual.getTriangles();
    size_t nVertices = expectedVertices.size();
    size_t nTriangles = expectedTriangles.size();
    CPPUNIT_ASSERT_EQUAL(nVertices, actualVertices.size());
    CPPUNIT_ASSERT_EQUAL(nTriangles, actualTriangles.size());

    // Maps vertex data to its position in the expectedVertices list
    map<boost::array<float, 3>, size_t> vertexMap;
    // Maps triangle in canonical form to number of occurrences in expectedTriangles list
    map<boost::array<std::tr1::uint32_t, 3>, size_t> triangleMap;
    for (size_t i = 0; i < nVertices; i++)
    {
        boost::array<float, 3> v;
        for (int j = 0; j < 3; j++)
            v[j] = expectedVertices[i].s[j];
        bool added = vertexMap.insert(make_pair(v, i)).second;
        CPPUNIT_ASSERT_MESSAGE("Vertices must be unique", added);
    }

    for (size_t i = 0; i < nTriangles; i++)
    {
        const boost::array<std::tr1::uint32_t, 3> canon
            = canonicalTriangle(expectedTriangles[i]);
        ++triangleMap[canon];
    }

    // Check that each vertex has a match. It is not necessary to check for
    // duplicate vertices because we've already checked for equal counts.
    for (size_t i = 0; i < nVertices; i++)
    {
        CPPUNIT_ASSERT(vertexMap.count(actualVertices[i]));
    }

    // Match up the actual triangles against the expected ones
    for (size_t i = 0; i < nTriangles; i++)
    {
        boost::array<std::tr1::uint32_t, 3> triangle = actualTriangles[i];
        for (int j = 0; j < 3; j++)
        {
            CPPUNIT_ASSERT(triangle[j] < nVertices);
            triangle[j] = vertexMap[actualVertices[triangle[j]]];
        }
        triangle = canonicalTriangle(triangle);
        --triangleMap[triangle];
    }

    pair<boost::array<std::tr1::uint32_t, 3>, size_t> i;
    BOOST_FOREACH(i, triangleMap)
    {
        CPPUNIT_ASSERT_MESSAGE("Triangle mismatch", i.second == 0);
    }
}

void TestMeshBase::testSimple()
{
    boost::scoped_ptr<MeshBase> mesh(meshFactory());
    unsigned int passes = mesh->numPasses();
    for (unsigned int i = 0; i < passes; i++)
    {
        Marching::OutputFunctor functor = mesh->outputFunctor(i);
    }
    mesh->finalize();
    MemoryWriter writer;
    mesh->write(writer, "");
}
