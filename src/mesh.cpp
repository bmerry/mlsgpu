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
#include <boost/ref.hpp>
#include <boost/thread/locks.hpp>
#include <boost/thread/mutex.hpp>
#include <tr1/unordered_map>
#include <cassert>
#include <cstdlib>
#include <utility>
#include <iterator>
#include <map>
#include <string>
#include "mesh.h"
#include "fast_ply.h"
#include "logging.h"
#include "errors.h"

std::map<std::string, MeshType> MeshTypeWrapper::getNameMap()
{
    std::map<std::string, MeshType> ans;
    ans["simple"] = SIMPLE_MESH;
    ans["weld"] = WELD_MESH;
    ans["big"] = BIG_MESH;
#if HAVE_STXXL
    ans["stxxl"] = STXXL_MESH;
#endif
    return ans;
}

/**
 * Utility class used by @ref serializeOutputFunctor.
 */
template<typename T>
class SerializeOutputFunctor
{
private:
    const T out;
    boost::mutex &mutex;

public:
    SerializeOutputFunctor(const T &out, boost::mutex &mutex)
        : out(out), mutex(mutex) {}

    void operator()(const cl::CommandQueue &queue,
                    const cl::Buffer &vertices,
                    const cl::Buffer &vertexKeys,
                    const cl::Buffer &indices,
                    std::size_t numVertices,
                    std::size_t numInternalVertices,
                    std::size_t numIndices,
                    cl::Event *event)
    {
        boost::lock_guard<boost::mutex> lock(mutex);
        out(queue, vertices, vertexKeys, indices, numVertices, numInternalVertices, numIndices, event);
    }
};

/**
 * Creates a wrapper around a function object that will take a lock before
 * forwarding to the wrapper. This allows an output functor that is not
 * thread-safe to be made thread-safe.
 */
template<typename T>
static Marching::OutputFunctor serializeOutputFunctor(const T &out, boost::mutex &mutex)
{
    return SerializeOutputFunctor<T>(out, mutex);
}

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
                            &triangles[oldTriangles][0],
                            &wait, &last);
    queue.flush();

    /* Adjust the indices to be global */
    for (std::size_t i = oldTriangles; i < oldTriangles + numTriangles; i++)
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

    return serializeOutputFunctor(boost::bind(&SimpleMesh::add, this, _1, _2, _3, _4, _5, _6, _7, _8), mutex);
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
    std::vector<cl::Event> wait;

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
        wait.resize(1);
        wait[0] = last;
    }
    if (numExternal > 0)
    {
        queue.enqueueReadBuffer(vertices, CL_FALSE,
                                numInternal * (3 * sizeof(cl_float)),
                                numExternal * (3 * sizeof(cl_float)),
                                &externalVertices[oldExternal][0],
                                &wait, &last);
        wait.resize(1);
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

    return serializeOutputFunctor(boost::bind(&WeldMesh::add, this, _1, _2, _3, _4, _5, _6, _7, _8), mutex);
}


namespace detail
{

void KeyMapMesh::loadData(
    const cl::CommandQueue &queue,
    const cl::Buffer &dVertices,
    const cl::Buffer &dVertexKeys,
    const cl::Buffer &dIndices,
    std::vector<boost::array<cl_float, 3> > &hVertices,
    std::vector<cl_ulong> &hVertexKeys,
    std::vector<boost::array<cl_uint, 3> > &hTriangles,
    std::size_t numVertices,
    std::size_t numInternalVertices,
    std::size_t numTriangles,
    cl::Event *verticesEvent,
    cl::Event *trianglesEvent) const
{
    cl::Event keysEvent;
    std::size_t numExternalVertices = numVertices - numInternalVertices;

    hVertices.resize(numVertices);
    hVertexKeys.resize(numExternalVertices);
    hTriangles.resize(numTriangles);
    // TODO: revisit the dependency graph, here and in BigMesh
    if (numExternalVertices > 0)
    {
        queue.enqueueReadBuffer(dVertexKeys, CL_FALSE,
                                numInternalVertices * sizeof(cl_ulong),
                                numExternalVertices * sizeof(cl_ulong),
                                &hVertexKeys[0],
                                NULL, &keysEvent);
        /* Start this transfer going while we queue up the following ones */
        queue.flush();
    }

    queue.enqueueReadBuffer(dVertices, CL_FALSE, 0, numVertices * (3 * sizeof(cl_float)),
                            &hVertices[0][0], NULL, verticesEvent);
    queue.enqueueReadBuffer(dIndices, CL_FALSE,
                            0, numTriangles * (3 * sizeof(cl_uint)),
                            &hTriangles[0][0],
                            NULL, trianglesEvent);
    queue.flush();
    if (numExternalVertices > 0)
        keysEvent.wait();
}

std::size_t KeyMapMesh::updateKeyMap(
    cl_uint priorVertices,
    std::size_t numInternalVertices,
    const std::vector<cl_ulong> &hKeys,
    std::vector<cl_uint> &indexTable)
{
    cl_uint base = priorVertices + numInternalVertices;
    const std::size_t numExternalVertices = hKeys.size();
    std::size_t newKeys = 0;

    indexTable.resize(numExternalVertices);
    for (std::size_t i = 0; i < numExternalVertices; i++)
    {
        std::pair<map_type::iterator, bool> added;
        added = keyMap.insert(std::make_pair(tmpVertexKeys[i], base + newKeys));
        if (added.second)
            newKeys++;
        indexTable[i] = added.first->second;
    }
    return newKeys;
}

void KeyMapMesh::rewriteTriangles(
    cl_uint priorVertices,
    std::size_t numInternalVertices,
    const std::vector<cl_uint> &indexTable,
    std::vector<boost::array<cl_uint, 3> > &triangles) const
{
    for (std::size_t i = 0; i < triangles.size(); i++)
        for (unsigned int j = 0; j < 3; j++)
        {
            cl_uint &index = triangles[i][j];
            assert(index < numInternalVertices + indexTable.size());
            if (index < numInternalVertices)
                index = priorVertices + index;
            else
                index = indexTable[index - numInternalVertices];
        }
}

} // namespace detail

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

    if (numExternalVertices > 0)
    {
        tmpVertexKeys.resize(numExternalVertices);
        queue.enqueueReadBuffer(vertexKeys, CL_TRUE,
                                numInternalVertices * sizeof(cl_ulong),
                                numExternalVertices * sizeof(cl_ulong),
                                &tmpVertexKeys[0],
                                NULL, event);
    }

    nVertices += numInternalVertices;

    /* Build keyMap, counting how many external vertices are really new.
     * The values in the keymap are irrelevant since they will be destroyed
     * for the second pass.
     *
     * TODO: see whether it is worth using a separate unordered_set for this.
     */
    std::size_t newKeys = 0;
    for (std::size_t i = 0; i < numExternalVertices; i++)
    {
        if (keyMap.insert(std::make_pair(tmpVertexKeys[i], 0)).second)
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
    const std::size_t numExternalVertices = numVertices - numInternalVertices;
    const std::size_t numTriangles = numIndices / 3;

    loadData(queue,
             vertices, vertexKeys, indices,
             tmpVertices, tmpVertexKeys, tmpTriangles,
             numVertices, numInternalVertices, numTriangles,
             &verticesEvent, &indicesEvent);

    std::size_t newKeys = updateKeyMap(
        nextVertex, numInternalVertices,
        tmpVertexKeys, tmpIndexTable);

    verticesEvent.wait();
    /* Compact the vertex list to keep only new external vertices */
    for (std::size_t i = 0; i < numExternalVertices; i++)
    {
        const cl_uint pos = tmpIndexTable[i];
        if (pos >= nextVertex)
        {
            assert(pos - nextVertex >= numInternalVertices
                   && pos - nextVertex <= numInternalVertices + i);
            tmpVertices[pos - nextVertex] = tmpVertices[numInternalVertices + i];
        }
    }

    indicesEvent.wait();
    rewriteTriangles(nextVertex, numInternalVertices, tmpIndexTable, tmpTriangles);

    writer.writeVertices(nextVertex, numInternalVertices + newKeys, &tmpVertices[0][0]);
    writer.writeTriangles(nextTriangle, numTriangles, &tmpTriangles[0][0]);
    nextVertex += numInternalVertices + newKeys;
    nextTriangle += numTriangles;

    if (event != NULL)
        *event = indicesEvent;
}

Marching::OutputFunctor BigMesh::outputFunctor(unsigned int pass)
{
    switch (pass)
    {
    case 0:
        return serializeOutputFunctor(boost::bind(&BigMesh::count, this, _1, _2, _3, _4, _5, _6, _7, _8), mutex);
    case 1:
        nextVertex = 0;
        nextTriangle = 0;
        keyMap.clear();
        writer.setNumVertices(nVertices);
        writer.setNumTriangles(nTriangles);
        writer.open(filename);
        return serializeOutputFunctor(boost::bind(&BigMesh::add, this, _1, _2, _3, _4, _5, _6, _7, _8), mutex);
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


#if HAVE_STXXL

StxxlMesh::VertexBuffer::VertexBuffer(FastPly::WriterBase &writer, size_type capacity)
    : writer(writer), nextVertex(0)
{
    buffer.reserve(capacity);
}

void StxxlMesh::VertexBuffer::operator()(const boost::array<float, 3> &vertex)
{
    buffer.push_back(vertex);
    if (buffer.size() == buffer.capacity())
        flush();
}

void StxxlMesh::VertexBuffer::flush()
{
    writer.writeVertices(nextVertex, buffer.size(), &buffer[0][0]);
    nextVertex += buffer.size();
    buffer.clear();
}

StxxlMesh::TriangleBuffer::TriangleBuffer(FastPly::WriterBase &writer, size_type capacity)
    : writer(writer), nextTriangle(0)
{
    nextTriangle = 0;
    buffer.reserve(capacity);
}

void StxxlMesh::TriangleBuffer::operator()(const boost::array<std::tr1::uint32_t, 3> &triangle)
{
    buffer.push_back(triangle);
    if (buffer.size() == buffer.capacity())
        flush();
}

void StxxlMesh::add(
    const cl::CommandQueue &queue,
    const cl::Buffer &vertices,
    const cl::Buffer &vertexKeys,
    const cl::Buffer &indices,
    std::size_t numVertices,
    std::size_t numInternalVertices,
    std::size_t numIndices,
    cl::Event *event)
{
    cl::Event verticesEvent, indicesEvent;
    const std::size_t numExternalVertices = numVertices - numInternalVertices;
    const std::size_t numTriangles = numIndices / 3;
    const size_type priorVertices = this->vertices.size();

    loadData(queue, vertices, vertexKeys, indices,
             tmpVertices, tmpVertexKeys, tmpTriangles,
             numVertices, numInternalVertices, numTriangles,
             &verticesEvent, &indicesEvent);

    std::size_t newKeys = updateKeyMap(
        priorVertices, numInternalVertices,
        tmpVertexKeys, tmpIndexTable);

    /* Copy the vertices into storage */
    this->vertices.reserve(priorVertices + numInternalVertices + newKeys);
    verticesEvent.wait();
    std::copy(tmpVertices.begin(), tmpVertices.begin() + numInternalVertices,
              std::back_inserter(this->vertices));
    for (std::size_t i = 0; i < numExternalVertices; i++)
    {
        const cl_uint pos = tmpIndexTable[i];
        if (pos == this->vertices.size())
        {
            this->vertices.push_back(tmpVertices[numInternalVertices + i]);
        }
    }

    indicesEvent.wait();
    rewriteTriangles(priorVertices, numInternalVertices, tmpIndexTable, tmpTriangles);

    // Store the output triangles
    this->triangles.reserve(this->triangles.size() + tmpTriangles.size());
    copy(tmpTriangles.begin(), tmpTriangles.end(), std::back_inserter(this->triangles));

    if (event != NULL)
        *event = indicesEvent;
}

Marching::OutputFunctor StxxlMesh::outputFunctor(unsigned int pass)
{
    /* only one pass, so ignore it */
    (void) pass;
    assert(pass == 0);

    return serializeOutputFunctor(boost::bind(&StxxlMesh::add, this, _1, _2, _3, _4, _5, _6, _7, _8), mutex);
}

void StxxlMesh::finalize()
{
}

void StxxlMesh::TriangleBuffer::flush()
{
    writer.writeTriangles(nextTriangle, buffer.size(), &buffer[0][0]);
    nextTriangle += buffer.size();
    buffer.clear();
}

void StxxlMesh::write(FastPly::WriterBase &writer, const std::string &filename) const
{
    writer.setNumVertices(vertices.size());
    writer.setNumTriangles(triangles.size());
    writer.open(filename);

    // We need to use boost::bind here because stxxl::for_each does not handle unwrapping
    // a reference to a function object to call it.
    {
        VertexBuffer vb(writer, vertices_type::block_size / sizeof(vertices_type::value_type));
        stxxl::for_each(vertices.begin(), vertices.end(), boost::bind(boost::ref(vb), _1), 4);
        vb.flush();
    }
    {
        TriangleBuffer tb(writer, triangles_type::block_size / sizeof(triangles_type::value_type));
        stxxl::for_each(triangles.begin(), triangles.end(), boost::bind(boost::ref(tb), _1), 4);
        tb.flush();
    }
}

#endif /* HAVE_STXXL */


MeshBase *createMesh(MeshType type, FastPly::WriterBase &writer, const std::string &filename)
{
    switch (type)
    {
    case SIMPLE_MESH: return new SimpleMesh();
    case WELD_MESH:   return new WeldMesh();
    case BIG_MESH:    return new BigMesh(writer, filename);
#if HAVE_STXXL
    case STXXL_MESH:  return new StxxlMesh();
#endif
    }
    return NULL; // should never be reached
}
