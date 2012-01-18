/**
 * @file
 *
 * Data structures for storing the output of @ref Marching.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif

#ifndef __CL_ENABLE_EXCEPTIONS
# define __CL_ENABLE_EXCEPTIONS
#endif

#include <CL/cl.h>
#include <vector>
#include <boost/array.hpp>
#include <boost/function.hpp>
#include <boost/bind/bind.hpp>
#include <tr1/unordered_map>
#include <cassert>
#include <cstdlib>
#include <utility>
#include "mesh.h"
#include "fast_ply.h"
#include "logging.h"
#include "errors.h"

#if UNIT_TESTS
# include <map>
# include <set>
# include <algorithm>

bool MeshBase::isManifold(std::size_t numVertices, const std::vector<boost::array<cl_uint, 3> > &triangles)
{
    // List of edges opposite each vertex
    std::vector<std::vector<std::pair<cl_uint, cl_uint> > > edges(numVertices);
    for (std::size_t i = 0; i < triangles.size(); i++)
    {
        cl_uint indices[3] = {triangles[i][0], triangles[i][1], triangles[i][2]};
        for (unsigned int j = 0; j < 3; j++)
        {
            assert(indices[0] < numVertices);
            if (indices[0] == indices[1])
            {
                Log::log[Log::debug] << "Triangle " << i << " contains vertex " << indices[0] << " twice\n";
                return false;
            }
            edges[indices[0]].push_back(std::make_pair(indices[1], indices[2]));
            std::rotate(indices, indices + 1, indices + 3);
        }
    }

    // Now check that the neighborhood of each vertex is a line or ring
    for (std::size_t i = 0; i < numVertices; i++)
    {
        const std::vector<std::pair<cl_uint, cl_uint> > &neigh = edges[i];
        if (neigh.empty())
        {
            // disallow isolated vertices
            Log::log[Log::debug] << "Vertex " << i << " is isolated\n";
            return false;
        }
        std::map<cl_uint, cl_uint> arrow; // maps .first to .second
        std::set<cl_uint> seen; // .second that have been observed
        for (std::size_t j = 0; j < neigh.size(); j++)
        {
            cl_uint x = neigh[j].first;
            cl_uint y = neigh[j].second;
            if (arrow.count(x))
            {
                Log::log[Log::debug] << "Edge " << i << " - " << x << " occurs twice with same winding\n";
                return false;
            }
            arrow[x] = y;
            if (seen.count(y))
            {
                Log::log[Log::debug] << "Edge " << y << " - " << i << " occurs twice with same winding\n";
                return false;
            }
            seen.insert(y);
        }

        /* At this point, we have in-degree and out-degree of at most 1 for
         * each vertex, so we have a collection of lines and rings.
         */

        // Look for a starting point for a line
        cl_uint start = neigh[0].first;
        for (std::size_t j = 0; j < neigh.size(); j++)
        {
            if (!seen.count(neigh[j].first))
            {
                start = neigh[j].first;
                break;
            }
        }
        std::size_t len = 0;
        cl_uint cur = start;
        do
        {
            cur = arrow[cur];
            len++;
        } while (arrow.count(cur) && cur != start);
        if (len != neigh.size())
        {
            Log::log[Log::debug] << "Vertex " << i << " contains multiple boundaries\n";
            return false;
        }
    }
    return true;
}

#endif /* UNIT_TESTS */

void SimpleMesh::add(const cl::CommandQueue &queue,
                     const cl::Buffer &vertices,
                     const cl::Buffer &vertexKeys,
                     const cl::Buffer &indices,
                     std::size_t numVertices,
                     std::size_t numInternalVertices,
                     std::size_t numIndices,
                     cl::Event *event)
{
    /* Unused parameters */
    (void) numInternalVertices;
    (void) vertexKeys;

    std::size_t oldVertices = this->vertices.size();
    std::size_t oldTriangles = this->triangles.size();
    std::size_t numTriangles = numIndices / 3;
    this->vertices.resize(oldVertices + numVertices);
    this->triangles.resize(oldTriangles + numTriangles);

    std::vector<cl::Event> wait(1);
    cl::Event last;
    queue.enqueueReadBuffer(vertices, CL_FALSE,
                            0, numVertices * (3 * sizeof(cl_float)),
                            &this->vertices[oldVertices][0],
                            NULL, &last);
    wait[0] = last;
    queue.enqueueReadBuffer(indices, CL_TRUE,
                            0, numTriangles * (3 * sizeof(cl_uint)),
                            &this->triangles[oldTriangles][0],
                            &wait, &last);
    queue.flush();

    /* Adjust the indices to be global */
    for (std::size_t i = 0; i < numTriangles; i++)
        for (unsigned int j = 0; j < 3; j++)
            triangles[i][j] += oldVertices;

    if (event != NULL)
        *event = last;
}

#if UNIT_TESTS
bool SimpleMesh::isManifold() const
{
    return MeshBase::isManifold(vertices.size(), triangles);
}
#endif /* UNIT_TESTS */

void SimpleMesh::write(FastPly::WriterBase &writer, const std::string &filename) const
{
    writer.setNumVertices(vertices.size());
    writer.setNumTriangles(triangles.size());
    writer.open(filename);
    writer.writeVertices(0, vertices.size(), &vertices[0][0]);
    writer.writeTriangles(0, triangles.size(), &triangles[0][0]);
}

Marching::OutputFunctor SimpleMesh::outputFunctor(unsigned int pass)
{
    /* only one pass, so ignore it */
    (void) pass;
    assert(pass == 0);

    return Marching::OutputFunctor(boost::bind(&SimpleMesh::add, this, _1, _2, _3, _4, _5, _6, _7, _8));
}


void WeldMesh::add(const cl::CommandQueue &queue,
                   const cl::Buffer &vertices,
                   const cl::Buffer &vertexKeys,
                   const cl::Buffer &indices,
                   std::size_t numVertices,
                   std::size_t numInternal,
                   std::size_t numIndices,
                   cl::Event *event)
{
    std::size_t oldInternal = internalVertices.size();
    std::size_t oldExternal = externalVertices.size();
    std::size_t oldTriangles = triangles.size();
    std::size_t numExternal = numVertices - numInternal;
    std::size_t numTriangles = numIndices / 3;

    internalVertices.resize(oldInternal + numInternal);
    externalVertices.resize(oldExternal + numExternal);
    externalKeys.resize(externalVertices.size());
    triangles.resize(oldTriangles + numTriangles);

    cl::Event indicesEvent, last;
    std::vector<cl::Event> wait(1);

    queue.enqueueReadBuffer(indices, CL_FALSE, 0, numTriangles * (3 * sizeof(cl_uint)),
                            &triangles[oldTriangles][0], NULL, &indicesEvent);
    queue.flush(); // Kick off this read-back in the background while we queue more.

    /* Read back the vertex and key data. We don't need it now, so we just return
     * an event for it.
     * TODO: allow them to proceed in parallel.
     */
    if (numInternal > 0)
    {
        queue.enqueueReadBuffer(vertices, CL_FALSE,
                                0,
                                numInternal * (3 * sizeof(cl_float)),
                                &internalVertices[oldInternal][0],
                                NULL, &last);
        wait[0] = last;
    }
    if (numExternal > 0)
    {
        queue.enqueueReadBuffer(vertices, CL_FALSE,
                                numInternal * (3 * sizeof(cl_float)),
                                numExternal * (3 * sizeof(cl_float)),
                                &externalVertices[oldExternal][0],
                                &wait, &last);
        wait[0] = last;
        queue.enqueueReadBuffer(vertexKeys, CL_FALSE,
                                numInternal * sizeof(cl_ulong),
                                numExternal * sizeof(cl_ulong),
                                &externalKeys[oldExternal],
                                &wait, &last);
        wait[0] = last;
    }

    /* Rewrite indices to refer to the two separate arrays, at the same time
     * applying ~ to the external indices to disambiguate them. Note that
     * these offsets may wrap around, but that is well-defined for unsigned
     * values.
     */
    indicesEvent.wait();
    cl_uint offsetInternal = oldInternal;
    cl_uint offsetExternal = oldExternal - numInternal;
    for (std::size_t i = oldTriangles; i < oldTriangles + numTriangles; i++)
        for (unsigned int j = 0; j < 3; j++)
        {
            cl_uint &index = triangles[i][j];
            if (index < numInternal)
                index = (index + offsetInternal);
            else
                index = ~(index + offsetExternal);
        }
    if (event != NULL)
        *event = last; /* Waits for vertices to be transferred */
}

void WeldMesh::finalize()
{
    std::size_t welded = 0;
    std::tr1::unordered_map<cl_ulong, cl_uint> place; // maps keys to new positions

    /* Maps original external indices to new ones. It includes a bias of
     * |internalVertices| so that we can index the concatenation of
     * internal and external vertices.
     */
    std::vector<cl_uint> remap(externalVertices.size());

    /* Weld the external vertices in place */
    for (size_t i = 0; i < externalVertices.size(); i++)
    {
        cl_ulong key = externalKeys[i];
        std::tr1::unordered_map<cl_ulong, cl_uint>::const_iterator pos = place.find(key);
        if (pos == place.end())
        {
            // New key, not seen before
            place[key] = welded;
            remap[i] = welded + internalVertices.size();
            // Shuffle down the vertex data in-place
            externalVertices[welded] = externalVertices[i];
            welded++;
        }
        else
        {
            remap[i] = pos->second + internalVertices.size();
        }
    }

    /* Rewrite the indices that refer to external vertices
     * (TODO: is it possible to partition these as well, to
     * reduce the work?)
     */
    for (std::size_t i = 0; i < triangles.size(); i++)
        for (unsigned int j = 0; j < 3; j++)
        {
            cl_uint &index = triangles[i][j];
            if (index >= internalVertices.size())
            {
                assert(~index < externalVertices.size());
                index = remap[~index];
            }
        }

    /* Throw away unneeded data. */
    std::vector<cl_ulong>().swap(externalKeys);
    externalVertices.resize(welded);
}

#if UNIT_TESTS
bool WeldMesh::isManifold() const
{
    return MeshBase::isManifold(internalVertices.size() + externalVertices.size(), triangles);
}
#endif

void WeldMesh::write(FastPly::WriterBase &writer, const std::string &filename) const
{
    writer.setNumVertices(internalVertices.size() + externalVertices.size());
    writer.setNumTriangles(triangles.size());
    writer.open(filename);
    writer.writeVertices(0, internalVertices.size(), &internalVertices[0][0]);
    writer.writeVertices(internalVertices.size(), externalVertices.size(), &externalVertices[0][0]);
    writer.writeTriangles(0, triangles.size(), &triangles[0][0]);
}

Marching::OutputFunctor WeldMesh::outputFunctor(unsigned int pass)
{
    /* only one pass, so ignore it */
    (void) pass;
    assert(pass == 0);

    return Marching::OutputFunctor(boost::bind(&WeldMesh::add, this, _1, _2, _3, _4, _5, _6, _7, _8));
}


BigMesh::BigMesh(FastPly::WriterBase &writer, const std::string &filename)
    : writer(writer), filename(filename), nVertices(0), nTriangles(0),
    nextVertex(0), nextTriangle(0)
{
    MLSGPU_ASSERT(writer.supportsOutOfOrder(), std::invalid_argument);
}

void BigMesh::count(const cl::CommandQueue &queue,
                    const cl::Buffer &vertices,
                    const cl::Buffer &vertexKeys,
                    const cl::Buffer &indices,
                    std::size_t numVertices,
                    std::size_t numInternalVertices,
                    std::size_t numIndices,
                    cl::Event *event)
{
    /* Unused parameters */
    (void) vertices;
    (void) indices;
    (void) numIndices;

    std::size_t numExternalVertices = numVertices - numInternalVertices;
    nTriangles += numIndices / 3;

    tmpKeys.resize(numExternalVertices);
    queue.enqueueReadBuffer(vertexKeys, CL_TRUE,
                            numInternalVertices * sizeof(cl_ulong),
                            numExternalVertices * sizeof(cl_ulong),
                            &tmpKeys[0],
                            NULL, event);

    nVertices += numInternalVertices;

    /* Build keyMap, counting how many external vertices are really new */
    std::size_t newKeys = 0;
    for (std::size_t i = 0; i < numExternalVertices; i++)
    {
        if (keyMap.insert(std::make_pair(tmpKeys[i], nVertices + newKeys)).second)
            newKeys++;
    }
    nVertices += newKeys;
}

void BigMesh::add(const cl::CommandQueue &queue,
                  const cl::Buffer &vertices,
                  const cl::Buffer &vertexKeys,
                  const cl::Buffer &indices,
                  std::size_t numVertices,
                  std::size_t numInternalVertices,
                  std::size_t numIndices,
                  cl::Event *event)
{
    cl::Event verticesEvent, indicesEvent;
    std::size_t numExternalVertices = numVertices - numInternalVertices;
    std::size_t numTriangles = numIndices / 3;

    tmpVertices.resize(numVertices);
    tmpKeys.resize(numExternalVertices);
    tmpTriangles.resize(numTriangles);
    queue.enqueueReadBuffer(vertices, CL_FALSE, 0, numVertices * (3 * sizeof(cl_float)),
                            &tmpVertices[0][0], NULL, &verticesEvent);
    queue.enqueueReadBuffer(vertexKeys, CL_TRUE,
                            numInternalVertices * sizeof(cl_ulong),
                            numExternalVertices * sizeof(cl_ulong),
                            &tmpKeys[0],
                            NULL, event);
    queue.enqueueReadBuffer(indices, CL_FALSE,
                            0, numIndices * sizeof(cl_uint),
                            &tmpTriangles[0][0],
                            NULL, &indicesEvent);
    queue.flush();

    verticesEvent.wait();
    std::size_t newKeys = 0;
    for (std::size_t i = 0; i < numExternalVertices; i++)
    {
        const cl_uint pos = keyMap[tmpKeys[i]];
        if (pos >= nextVertex)
        {
            assert(pos - nextVertex >= numInternalVertices
                   && pos - nextVertex <= numInternalVertices + i);
            tmpVertices[pos - nextVertex] = tmpVertices[numInternalVertices + i];
            newKeys++;
        }
    }

    /* TODO: store a vector for directly remapping indices to new indices,
     * to avoid hash table lookups on the key.
     */
    indicesEvent.wait();
    for (std::size_t i = 0; i < numTriangles; i++)
        for (unsigned int j = 0; j < 3; j++)
        {
            cl_uint &index = tmpTriangles[i][j];
            assert(index < numVertices);
            if (index < numInternalVertices)
                index = nextVertex + index;
            else
                index = keyMap[tmpKeys[index - numInternalVertices]];
        }

    writer.writeVertices(nextVertex, numInternalVertices + newKeys, &tmpVertices[0][0]);
    writer.writeTriangles(nextTriangle, numTriangles, &tmpTriangles[0][0]);
    nextVertex += numInternalVertices + newKeys;
    nextTriangle += numTriangles;
}

Marching::OutputFunctor BigMesh::outputFunctor(unsigned int pass)
{
    switch (pass)
    {
    case 0:
        return Marching::OutputFunctor(boost::bind(&BigMesh::count, this, _1, _2, _3, _4, _5, _6, _7, _8));
    case 1:
        nextVertex = 0;
        writer.setNumVertices(nVertices);
        writer.setNumTriangles(nTriangles);
        writer.open(filename);
        return Marching::OutputFunctor(boost::bind(&BigMesh::add, this, _1, _2, _3, _4, _5, _6, _7, _8));
    default:
        abort();
    }
}

void BigMesh::finalize()
{
}

void BigMesh::write(FastPly::WriterBase &writer, const std::string &filename) const
{
    assert(&writer == &this->writer);
    assert(filename == this->filename);
}
