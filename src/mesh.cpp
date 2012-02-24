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
#include <boost/smart_ptr/scoped_ptr.hpp>
#include <boost/foreach.hpp>
#include <boost/type_traits/make_unsigned.hpp>
#include <tr1/unordered_map>
#include <cassert>
#include <cstdlib>
#include <utility>
#include <iterator>
#include <map>
#include <string>
#include <ostream>
#include "mesh.h"
#include "fast_ply.h"
#include "logging.h"
#include "errors.h"
#include "progress.h"
#include "union_find.h"
#include "statistics.h"

std::map<std::string, MeshType> MeshTypeWrapper::getNameMap()
{
    std::map<std::string, MeshType> ans;
    ans["simple"] = SIMPLE_MESH;
    ans["weld"] = WELD_MESH;
    ans["big"] = BIG_MESH;
    ans["stxxl"] = STXXL_MESH;
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

void SimpleMesh::write(FastPly::WriterBase &writer, const std::string &filename,
                       std::ostream *progressStream) const
{
    (void) progressStream;

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

void WeldMesh::finalize(std::ostream *progressStream)
{
    std::size_t welded = 0;
    std::tr1::unordered_map<cl_ulong, cl_uint> place; // maps keys to new positions

    /* Maps original external indices to new ones. It includes a bias of
     * |internalVertices| so that we can index the concatenation of
     * internal and external vertices.
     */
    std::vector<cl_uint> remap(externalVertices.size());

    /* Weld the external vertices in place */
    boost::scoped_ptr<ProgressDisplay> progress;
    if (progressStream != NULL)
    {
        *progressStream << "\nWelding vertices\n";
        progress.reset(new ProgressDisplay(externalVertices.size()));
    }
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
        if (progress)
            ++*progress;
    }

    /* Rewrite the indices that refer to external vertices.
     */
    if (progressStream != NULL)
    {
        *progressStream << "\nAdjusting indices\n";
        progress->restart(triangles.size());
    }
    for (std::size_t i = 0; i < triangles.size(); i++)
    {
        for (unsigned int j = 0; j < 3; j++)
        {
            cl_uint &index = triangles[i][j];
            if (index >= internalVertices.size())
            {
                assert(~index < externalVertices.size());
                index = remap[~index];
            }
        }
        if (progress)
            ++*progress;
    }

    /* Throw away unneeded data and concatenate vertices */
    std::vector<cl_ulong>().swap(externalKeys);
    externalVertices.resize(welded);
    internalVertices.insert(internalVertices.end(), externalVertices.begin(), externalVertices.end());
    std::vector<boost::array<cl_float, 3> >().swap(externalVertices);

    /* Now detect components and throw away undersized ones */
    std::vector<UnionFind::Node<cl_int> > nodes(internalVertices.size());
    typedef boost::array<cl_uint, 3> triangle_type;
    BOOST_FOREACH(const triangle_type &triangle, triangles)
    {
        for (int j = 0; j < 2; j++)
            UnionFind::merge(nodes, triangle[j], triangle[j + 1]);
    }

    remap.resize(internalVertices.size());
    cl_uint newVertices = 0;
    const cl_uint pruneThresholdVertices = (cl_uint) (internalVertices.size() * getPruneThreshold());
    for (cl_uint i = 0; i < internalVertices.size(); i++)
    {
        cl_int root = UnionFind::findRoot(nodes, i);
        if ((cl_uint) nodes[root].size() >= pruneThresholdVertices)
        {
            internalVertices[newVertices] = internalVertices[i];
            remap[i] = newVertices;
            newVertices++;
        }
        else
            remap[i] = 0xFFFFFFFFu;
    }
    internalVertices.resize(newVertices);

    std::size_t newTriangles = 0;
    for (std::size_t i = 0; i < triangles.size(); i++)
    {
        if (remap[triangles[i][0]] != 0xFFFFFFFFu)
        {
            for (int j = 0; j < 3; j++)
                triangles[newTriangles][j] = remap[triangles[i][j]];
            newTriangles++;
        }
    }
    triangles.resize(newTriangles);
}

#if UNIT_TESTS
bool WeldMesh::isManifold() const
{
    return MeshBase::isManifold(internalVertices.size() + externalVertices.size(), triangles);
}
#endif

void WeldMesh::write(FastPly::WriterBase &writer, const std::string &filename,
                     std::ostream *progressStream) const
{
    // Probably not worth trying to use this given the amount of data that can be
    // handled by WeldMesh
    (void) progressStream;

    writer.setNumVertices(internalVertices.size());
    writer.setNumTriangles(triangles.size());
    writer.open(filename);
    writer.writeVertices(0, internalVertices.size(), &internalVertices[0][0]);
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
    // TODO: revisit the dependency graph
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

    if (dVertices())
    {
        queue.enqueueReadBuffer(dVertices, CL_FALSE, 0, numVertices * (3 * sizeof(cl_float)),
                                &hVertices[0][0], NULL, verticesEvent);
    }
    queue.enqueueReadBuffer(dIndices, CL_FALSE,
                            0, numTriangles * (3 * sizeof(cl_uint)),
                            &hTriangles[0][0],
                            NULL, trianglesEvent);
    queue.flush();
    if (numExternalVertices > 0)
        keysEvent.wait();
}

void KeyMapMesh::computeLocalComponents(
    std::size_t numVertices,
    const std::vector<boost::array<cl_uint, 3> > &triangles,
    std::vector<clump_id> &clumpId)
{
    tmpNodes.clear();
    tmpNodes.resize(numVertices);
    typedef boost::array<cl_uint, 3> triangle_type;
    BOOST_FOREACH(const triangle_type &triangle, triangles)
    {
        // Only need to use two edges in the union-find tree, since the
        // third will be redundant.
        for (unsigned int j = 0; j < 2; j++)
            UnionFind::merge(tmpNodes, triangle[j], triangle[j + 1]);
    }

    // Allocate clumps for the local components
    clumpId.resize(numVertices);
    for (std::size_t i = 0; i < numVertices; i++)
    {
        if (tmpNodes[i].isRoot())
        {
            if (clumps.size() > boost::make_unsigned<clump_id>::type(std::numeric_limits<clump_id>::max()))
            {
                throw std::overflow_error("Too many clumps");
            }
            clumpId[i] = clumps.size();
            clumps.push_back(Clump(tmpNodes[i].size()));
        }
    }

    // Compute clump IDs for the non-root vertices
    for (std::size_t i = 0; i < numVertices; i++)
    {
        cl_int r = UnionFind::findRoot(tmpNodes, i);
        clumpId[i] = clumpId[r];
    }

    // Compute triangle counts for the clumps
    BOOST_FOREACH(const triangle_type &triangle, triangles)
    {
        Clump &clump = clumps[clumpId[triangle[0]]];
        clump.triangles++;
    }
}

std::size_t KeyMapMesh::updateKeyMap(
    cl_uint vertexOffset,
    const std::vector<cl_ulong> &hKeys,
    const std::vector<clump_id> &clumpId,
    std::vector<cl_uint> &indexTable)
{
    const std::size_t numExternalVertices = hKeys.size();
    const std::size_t numInternalVertices = clumpId.size() - numExternalVertices;
    std::size_t newKeys = 0;

    indexTable.resize(numExternalVertices);
    for (std::size_t i = 0; i < numExternalVertices; i++)
    {
        clump_id cid = clumpId[i + numInternalVertices];
        ExternalVertexData ed(vertexOffset + newKeys, cid);
        std::pair<map_type::iterator, bool> added;
        added = keyMap.insert(std::make_pair(tmpVertexKeys[i], ed));
        if (added.second)
            newKeys++;
        else
        {
            // Unified two external vertices. Also need to unify their clumps.
            clump_id cid2 = added.first->second.clumpId;
            UnionFind::merge(clumps, cid, cid2);
            // They will both have counted the common vertex, so we need to
            // subtract it.
            cid = UnionFind::findRoot(clumps, cid);
            clumps[cid].vertices--;
        }
        indexTable[i] = added.first->second.vertexId;
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
    : writer(writer), filename(filename),
    nextVertex(0), nextTriangle(0), pruneThresholdVertices(0)
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

    cl::Event trianglesEvent;
    const std::size_t numExternalVertices = numVertices - numInternalVertices;
    const std::size_t numTriangles = numIndices / 3;

    loadData(queue, cl::Buffer(), vertexKeys, indices,
             tmpVertices, tmpVertexKeys, tmpTriangles,
             numVertices, numInternalVertices, numTriangles,
             NULL, &trianglesEvent);

    trianglesEvent.wait();
    computeLocalComponents(numVertices, tmpTriangles, tmpClumpId);

    /* Build keyClump */
    for (std::size_t i = 0; i < numExternalVertices; i++)
    {
        std::pair<key_clump_type::iterator, bool> added;
        clump_id cid = tmpClumpId[i + numInternalVertices];
        added = keyClump.insert(std::make_pair(tmpVertexKeys[i], cid));
        if (!added.second)
        {
            // Merge the clumps that share the external vertex
            clump_id cid2 = added.first->second;
            UnionFind::merge(clumps, cid, cid2);
            // They will both have counted the common vertex, so we need to
            // subtract it.
            cid = UnionFind::findRoot(clumps, cid);
            clumps[cid].vertices--;
        }
    }

    if (event != NULL)
        *event = trianglesEvent;
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
    cl::Event verticesEvent, trianglesEvent;
    std::size_t numExternalVertices = numVertices - numInternalVertices;
    std::size_t numTriangles = numIndices / 3;

    loadData(queue,
             vertices, vertexKeys, indices,
             tmpVertices, tmpVertexKeys, tmpTriangles,
             numVertices, numInternalVertices, numTriangles,
             &verticesEvent, &trianglesEvent);

    trianglesEvent.wait();
    clumps.clear();
    computeLocalComponents(numVertices, tmpTriangles, tmpClumpId);

    /* Determine which components are valid. We no longer need the triangles
     * member of Clump, so we overwrite it with a validity boolean. A clump
     * is valid if either it has the requisite number of vertices on its
     * own, or if it contains an external vertex that has been marked as
     * valid.
     */
    for (std::size_t i = 0; i < clumps.size(); i++)
    {
        clumps[i].triangles = (clumps[i].vertices >= pruneThresholdVertices);
    }
    for (std::size_t i = 0; i < numExternalVertices; i++)
    {
        if (keyClump[tmpVertexKeys[i]])
            clumps[tmpClumpId[i + numInternalVertices]].triangles = true;
    }

    verticesEvent.wait();

    /* Apply clump validity to remove unwanted vertices */
    std::size_t vptr = 0;  // next output vertex in compaction
    std::size_t kptr = 0;  // next output key in compaction
    tmpIndexTable.resize(numVertices);
    for (std::size_t i = 0; i < numInternalVertices; i++)
    {
        if (clumps[tmpClumpId[i]].triangles)
        {
            tmpIndexTable[i] = vptr;
            tmpVertices[vptr++] = tmpVertices[i];
        }
        else
            tmpIndexTable[i] = 0xFFFFFFFFu;
        // TODO: need to compact tmpVertexKeys as well!!!
    }
    for (std::size_t i = numInternalVertices; i < numVertices; i++)
    {
        if (clumps[tmpClumpId[i]].triangles)
        {
            tmpIndexTable[i] = vptr;
            tmpVertices[vptr++] = tmpVertices[i];
            tmpVertexKeys[kptr++] = tmpVertexKeys[i - numInternalVertices];
        }
        else
            tmpIndexTable[i] = 0xFFFFFFFFu;
    }
    numVertices = vptr;
    numExternalVertices = kptr;
    numInternalVertices = vptr - kptr;
    tmpVertices.resize(numVertices);
    tmpVertexKeys.resize(numExternalVertices);

    /* Use clump validity to remove dead triangles and rewrite remaining ones */
    std::size_t tptr = 0;
    for (std::size_t i = 0; i < numTriangles; i++)
    {
        if (tmpIndexTable[tmpTriangles[i][0]] != 0xFFFFFFFFu)
        {
            for (unsigned int j = 0; j < 3; j++)
                tmpTriangles[tptr][j] = tmpIndexTable[tmpTriangles[i][j]];
            tptr++;
        }
    }
    numTriangles = tptr;
    tmpTriangles.resize(numTriangles);

    std::size_t newKeys = updateKeyMap(
        nextVertex + numInternalVertices,
        tmpVertexKeys, tmpClumpId, tmpIndexTable);

    /* Compact the vertex list (again) to keep only new external vertices */
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
    numVertices = numInternalVertices + newKeys;
    tmpVertices.resize(numVertices);

    /* Rewrite triangles (again) to global indices */
    rewriteTriangles(nextVertex, numInternalVertices, tmpIndexTable, tmpTriangles);

    writer.writeVertices(nextVertex, numVertices, &tmpVertices[0][0]);
    writer.writeTriangles(nextTriangle, numTriangles, &tmpTriangles[0][0]);
    nextVertex += numInternalVertices + newKeys;
    nextTriangle += numTriangles;

    if (event != NULL)
        *event = trianglesEvent;
}

void BigMesh::prepareAdd()
{
    Statistics::Registry &registry = Statistics::Registry::getInstance();

    size_type totalVertices = 0;
    /* Count the vertices */
    BOOST_FOREACH(const Clump &clump, clumps)
    {
        if (clump.isRoot())
        {
            totalVertices += clump.vertices;
        }
    }

    size_type numVertices = 0, numTriangles = 0;
    pruneThresholdVertices = (cl_uint) (totalVertices * getPruneThreshold());
    clump_id keptComponents = 0, totalComponents = 0;

    /* Determine the total number of vertices and triangles to retain */
    BOOST_FOREACH(const Clump &clump, clumps)
    {
        if (clump.isRoot())
        {
            totalComponents++;
            if (clump.vertices >= pruneThresholdVertices)
            {
                numVertices += clump.vertices;
                numTriangles += clump.triangles;
                keptComponents++;
            }
        }
    }

    registry.getStatistic<Statistics::Variable>("components.total").add(totalComponents);
    registry.getStatistic<Statistics::Variable>("components.kept").add(keptComponents);
    registry.getStatistic<Statistics::Variable>("externalvertices").add(keyClump.size());

    /* Determine which external vertices belong to valid clumps, replacing
     * the clump IDs (which will become useless anyway) with a boolean
     * indicating whether the external vertex should be retained.
     */
    for (key_clump_type::iterator i = keyClump.begin(); i != keyClump.end(); ++i)
    {
        clump_id cid = i->second;
        cid = UnionFind::findRoot(clumps, cid);
        i->second = clumps[cid].vertices >= pruneThresholdVertices;
    }

    nextVertex = 0;
    nextTriangle = 0;
    keyMap.clear();
    clumps.clear();
    writer.setNumVertices(numVertices);
    writer.setNumTriangles(numTriangles);
    writer.open(filename);
}

Marching::OutputFunctor BigMesh::outputFunctor(unsigned int pass)
{
    switch (pass)
    {
    case 0:
        return serializeOutputFunctor(boost::bind(&BigMesh::count, this, _1, _2, _3, _4, _5, _6, _7, _8), mutex);
    case 1:
        prepareAdd();
        return serializeOutputFunctor(boost::bind(&BigMesh::add, this, _1, _2, _3, _4, _5, _6, _7, _8), mutex);
    default:
        abort();
    }
}

void BigMesh::write(FastPly::WriterBase &writer, const std::string &filename,
                    std::ostream *progressStream) const
{
    (void) writer;
    (void) filename;
    (void) progressStream;
    assert(&writer == &this->writer);
    assert(filename == this->filename);
}


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

    indicesEvent.wait();
    computeLocalComponents(numVertices, tmpTriangles, tmpClumpId);

    std::size_t newKeys = updateKeyMap(
        priorVertices + numInternalVertices,
        tmpVertexKeys, tmpClumpId, tmpIndexTable);

    /* Copy the vertices into storage */
    this->vertices.reserve(priorVertices + numInternalVertices + newKeys);
    verticesEvent.wait();
    for (std::size_t i = 0; i < numInternalVertices; i++)
    {
        this->vertices.push_back(std::make_pair(tmpVertices[i], tmpClumpId[i]));
    }
    for (std::size_t i = 0; i < numExternalVertices; i++)
    {
        const cl_uint pos = tmpIndexTable[i];
        if (pos == this->vertices.size())
        {
            this->vertices.push_back(std::make_pair(
                    tmpVertices[numInternalVertices + i],
                    tmpClumpId[numInternalVertices + i]));
        }
    }

    rewriteTriangles(priorVertices, numInternalVertices, tmpIndexTable, tmpTriangles);

    // Store the output triangles
    this->triangles.reserve(this->triangles.size() + tmpTriangles.size());
    std::copy(tmpTriangles.begin(), tmpTriangles.end(), std::back_inserter(this->triangles));

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

void StxxlMesh::TriangleBuffer::flush()
{
    writer.writeTriangles(nextTriangle, buffer.size(), &buffer[0][0]);
    nextTriangle += buffer.size();
    buffer.clear();
}

void StxxlMesh::write(FastPly::WriterBase &writer, const std::string &filename,
                      std::ostream *progressStream) const
{
    Statistics::Registry &registry = Statistics::Registry::getInstance();

    // TODO: make a parameter
    const cl_uint thresholdVertices = (cl_uint) (vertices.size() * getPruneThreshold());
    FastPly::WriterBase::size_type numVertices = 0, numTriangles = 0;
    clump_id keptComponents = 0, totalComponents = 0;
    BOOST_FOREACH(const Clump &clump, clumps)
    {
        if (clump.isRoot())
        {
            totalComponents++;
            if (clump.vertices >= thresholdVertices)
            {
                numVertices += clump.vertices;
                numTriangles += clump.triangles;
                keptComponents++;
            }
        }
    }
    registry.getStatistic<Statistics::Variable>("components.vertices.total").add(vertices.size());
    registry.getStatistic<Statistics::Variable>("components.vertices.threshold").add(thresholdVertices);
    registry.getStatistic<Statistics::Variable>("components.total").add(totalComponents);
    registry.getStatistic<Statistics::Variable>("components.kept").add(keptComponents);
    registry.getStatistic<Statistics::Variable>("externalvertices").add(keyMap.size());

    // TODO: get from clumps
    writer.setNumVertices(numVertices);
    writer.setNumTriangles(numTriangles);
    writer.open(filename);

    boost::scoped_ptr<ProgressDisplay> progress;
    if (progressStream != NULL)
    {
        *progressStream << "\nWriting file\n";
        progress.reset(new ProgressDisplay(vertices.size() + triangles.size()));
    }

    stxxl::VECTOR_GENERATOR<cl_uint, 4, 16>::result vertexRemap;
    const stxxl::VECTOR_GENERATOR<cl_uint, 4, 16>::result & vertexRemapConst = vertexRemap;
    vertexRemap.reserve(vertices.size());
    cl_uint nextVertex = 0;
    const cl_uint badIndex = std::numeric_limits<cl_uint>::max();

    {
        stxxl::stream::streamify_traits<vertices_type::const_iterator>::stream_type
            vertex_stream = stxxl::stream::streamify(vertices.begin(), vertices.end());
        VertexBuffer vb(writer, vertices_type::block_size / sizeof(vertices_type::value_type));
        while (!vertex_stream.empty())
        {
            vertices_type::value_type vertex = *vertex_stream;
            ++vertex_stream;
            clump_id clumpId = UnionFind::findRoot(clumps, vertex.second);
            if (clumps[clumpId].vertices >= thresholdVertices)
            {
                vb(vertex.first);
                vertexRemap.push_back(nextVertex++);
            }
            else
            {
                vertexRemap.push_back(badIndex);
            }
            if (progress)
                ++*progress;
        }
        vb.flush();
    }
    assert(nextVertex == numVertices);

    {
        stxxl::stream::streamify_traits<triangles_type::const_iterator>::stream_type
            triangle_stream = stxxl::stream::streamify(triangles.begin(), triangles.end());
        TriangleBuffer tb(writer, triangles_type::block_size / sizeof(triangles_type::value_type));
        while (!triangle_stream.empty())
        {
            triangles_type::value_type triangle = *triangle_stream;
            ++triangle_stream;

            boost::array<cl_uint, 3> rewritten;
            for (unsigned int i = 0; i < 3; i++)
            {
                rewritten[i] = vertexRemapConst[triangle[i]] + 1;
            }
            if (rewritten[0] != badIndex)
            {
                assert(rewritten[1] != badIndex);
                assert(rewritten[2] != badIndex);
                tb(rewritten);
            }
            else
            {
                assert(rewritten[1] == badIndex);
                assert(rewritten[2] == badIndex);
            }
            if (progress)
                ++*progress;
        }
        tb.flush();
    }
}


MeshBase *createMesh(MeshType type, FastPly::WriterBase &writer, const std::string &filename)
{
    switch (type)
    {
    case SIMPLE_MESH: return new SimpleMesh();
    case WELD_MESH:   return new WeldMesh();
    case BIG_MESH:    return new BigMesh(writer, filename);
    case STXXL_MESH:  return new StxxlMesh();
    }
    return NULL; // should never be reached
}
