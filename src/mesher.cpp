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
#include "mesher.h"
#include "fast_ply.h"
#include "logging.h"
#include "errors.h"
#include "progress.h"
#include "union_find.h"
#include "statistics.h"
#include "clh.h"

std::map<std::string, MesherType> MesherTypeWrapper::getNameMap()
{
    std::map<std::string, MesherType> ans;
    ans["simple"] = SIMPLE_MESHER;
    ans["weld"] = WELD_MESHER;
    ans["big"] = BIG_MESHER;
    ans["stxxl"] = STXXL_MESHER;
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
                    const DeviceKeyMesh &mesh,
                    const std::vector<cl::Event> *events,
                    cl::Event *event)
    {
        boost::lock_guard<boost::mutex> lock(mutex);
        out(queue, mesh, events, event);
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

void SimpleMesher::add(const cl::CommandQueue &queue,
                       const DeviceKeyMesh &mesh,
                       const std::vector<cl::Event> *events,
                       cl::Event *event)
{
    std::size_t oldVertices = this->vertices.size();
    std::size_t oldTriangles = this->triangles.size();
    this->vertices.resize(oldVertices + mesh.numVertices);
    this->triangles.resize(oldTriangles + mesh.numTriangles);

    cl::Event verticesEvent;
    CLH::enqueueReadBuffer(queue,
                           mesh.vertices, CL_FALSE,
                           0, mesh.numVertices * (3 * sizeof(cl_float)),
                           &vertices[oldVertices][0],
                           events, &verticesEvent);
    // Kick off this enqueue in the backward while enqueuing more work
    queue.flush();
    CLH::enqueueReadBuffer(queue,
                           mesh.triangles, CL_TRUE,
                           0, mesh.numTriangles * (3 * sizeof(cl_uint)),
                           &triangles[oldTriangles][0],
                           events, NULL);

    /* Adjust the indices to be global */
    for (std::size_t i = oldTriangles; i < oldTriangles + mesh.numTriangles; i++)
        for (unsigned int j = 0; j < 3; j++)
            triangles[i][j] += oldVertices;

    if (event != NULL)
        *event = verticesEvent;
}

void SimpleMesher::write(FastPly::WriterBase &writer, const std::string &filename,
                         std::ostream *progressStream) const
{
    (void) progressStream;

    writer.setNumVertices(vertices.size());
    writer.setNumTriangles(triangles.size());
    writer.open(filename);
    writer.writeVertices(0, vertices.size(), &vertices[0][0]);
    writer.writeTriangles(0, triangles.size(), &triangles[0][0]);
}

Marching::OutputFunctor SimpleMesher::outputFunctor(unsigned int pass)
{
    /* only one pass, so ignore it */
    (void) pass;
    assert(pass == 0);

    return serializeOutputFunctor(boost::bind(&SimpleMesher::add, this, _1, _2, _3, _4), mutex);
}


void WeldMesher::add(const cl::CommandQueue &queue,
                     const DeviceKeyMesh &mesh,
                     const std::vector<cl::Event> *events,
                     cl::Event *event)
{
    const std::size_t oldInternal = internalVertices.size();
    const std::size_t oldExternal = externalVertices.size();
    const std::size_t oldTriangles = triangles.size();
    const std::size_t numInternal = mesh.numInternalVertices;
    const std::size_t numExternal = mesh.numVertices - numInternal;

    internalVertices.resize(oldInternal + numInternal);
    externalVertices.resize(oldExternal + numExternal);
    externalKeys.resize(externalVertices.size());
    triangles.resize(oldTriangles + mesh.numTriangles);

    cl::Event indicesEvent, internalVerticesEvent, externalVerticesEvent, vertexKeysEvent;
    std::vector<cl::Event> wait;

    CLH::enqueueReadBuffer(queue,
                           mesh.triangles, CL_FALSE, 0, mesh.numTriangles * (3 * sizeof(cl_uint)),
                           &triangles[oldTriangles][0], events, &indicesEvent);
    queue.flush(); // Kick off this read-back in the background while we queue more.

    /* Read back the vertex and key data. We don't need it now, so we just return
     * an event for it.
     * TODO: allow them to proceed in parallel.
     */
    CLH::enqueueReadBuffer(queue,
                           mesh.vertices, CL_FALSE,
                           0,
                           numInternal * (3 * sizeof(cl_float)),
                           &internalVertices[oldInternal][0],
                           events, &internalVerticesEvent);
    wait.push_back(internalVerticesEvent);

    CLH::enqueueReadBuffer(queue,
                           mesh.vertices, CL_FALSE,
                           numInternal * (3 * sizeof(cl_float)),
                           numExternal * (3 * sizeof(cl_float)),
                           &externalVertices[oldExternal][0],
                           events, &externalVerticesEvent);
    CLH::enqueueReadBuffer(queue,
                           mesh.vertexKeys, CL_FALSE,
                           numInternal * sizeof(cl_ulong),
                           numExternal * sizeof(cl_ulong),
                           &externalKeys[oldExternal],
                           events, &vertexKeysEvent);
    wait.push_back(externalVerticesEvent);
    wait.push_back(vertexKeysEvent);

    /* Rewrite indices to refer to the two separate arrays, at the same time
     * applying ~ to the external indices to disambiguate them. Note that
     * these offsets may wrap around, but that is well-defined for unsigned
     * values.
     */
    queue.flush();
    indicesEvent.wait();
    cl_uint offsetInternal = oldInternal;
    cl_uint offsetExternal = oldExternal - numInternal;
    for (std::size_t i = oldTriangles; i < oldTriangles + mesh.numTriangles; i++)
        for (unsigned int j = 0; j < 3; j++)
        {
            cl_uint &index = triangles[i][j];
            if (index < numInternal)
                index = (index + offsetInternal);
            else
                index = ~(index + offsetExternal);
        }

    if (event != NULL)
    {
        CLH::enqueueMarkerWithWaitList(queue, &wait, event);
        queue.flush();
    }
}

void WeldMesher::finalize(std::ostream *progressStream)
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
        progress.reset(new ProgressDisplay(externalVertices.size(), *progressStream));
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
            // Verify that vertex generation is invariant
            if (externalVertices[i] != externalVertices[pos->second])
            {
                boost::array<cl_float, 3> v1 = externalVertices[i];
                boost::array<cl_float, 3> v2 = externalVertices[pos->second];
                Log::log[Log::warn] << "Vertex mismatch at vertex " << i << ":\n"
                    << "(" << v1[0] << ", " << v1[1] << ", " << v1[2] << ") vs ("
                    << v2[0] << ", " << v2[1] << ", " << v2[2] << ")\n";
            }
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

void WeldMesher::write(FastPly::WriterBase &writer, const std::string &filename,
                     std::ostream *progressStream) const
{
    // Probably not worth trying to use this given the amount of data that can be
    // handled by WeldMesher
    (void) progressStream;

    writer.setNumVertices(internalVertices.size());
    writer.setNumTriangles(triangles.size());
    writer.open(filename);
    writer.writeVertices(0, internalVertices.size(), &internalVertices[0][0]);
    writer.writeTriangles(0, triangles.size(), &triangles[0][0]);
}

Marching::OutputFunctor WeldMesher::outputFunctor(unsigned int pass)
{
    /* only one pass, so ignore it */
    (void) pass;
    assert(pass == 0);

    return serializeOutputFunctor(boost::bind(&WeldMesher::add, this, _1, _2, _3, _4), mutex);
}


namespace detail
{

void KeyMapMesher::computeLocalComponents(
    std::size_t numVertices, const std::vector<boost::array<cl_uint, 3> > &triangles,
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

std::size_t KeyMapMesher::updateKeyMap(
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
        added = keyMap.insert(std::make_pair(tmpMesh.vertexKeys[i], ed));
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

void KeyMapMesher::rewriteTriangles(
    cl_uint priorVertices,
    const std::vector<cl_uint> &indexTable,
    HostKeyMesh &mesh) const
{
    const std::size_t numInternalVertices = mesh.vertices.size() - mesh.vertexKeys.size();
    for (std::size_t i = 0; i < mesh.triangles.size(); i++)
        for (unsigned int j = 0; j < 3; j++)
        {
            cl_uint &index = mesh.triangles[i][j];
            assert(index < mesh.vertices.size());
            if (index < numInternalVertices)
                index = priorVertices + index;
            else
                index = indexTable[index - numInternalVertices];
        }
}

} // namespace detail

BigMesher::BigMesher(FastPly::WriterBase &writer, const std::string &filename)
    : writer(writer), filename(filename),
    nextVertex(0), nextTriangle(0), pruneThresholdVertices(0)
{
    MLSGPU_ASSERT(writer.supportsOutOfOrder(), std::invalid_argument);
}

void BigMesher::count(const cl::CommandQueue &queue,
                      const DeviceKeyMesh &mesh,
                      const std::vector<cl::Event> *events,
                      cl::Event *event)
{
    cl::Event vertexKeysEvent, trianglesEvent;
    const std::size_t numExternalVertices = mesh.numVertices - mesh.numInternalVertices;

    enqueueReadMesh(queue, mesh, tmpMesh, events, NULL, &vertexKeysEvent, &trianglesEvent);

    trianglesEvent.wait();
    computeLocalComponents(mesh.numVertices, tmpMesh.triangles, tmpClumpId);

    /* Build keyClump */
    vertexKeysEvent.wait();
    for (std::size_t i = 0; i < numExternalVertices; i++)
    {
        std::pair<key_clump_type::iterator, bool> added;
        clump_id cid = tmpClumpId[i + mesh.numInternalVertices];
        added = keyClump.insert(std::make_pair(tmpMesh.vertexKeys[i], cid));
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
        *event = vertexKeysEvent;
}

void BigMesher::add(const cl::CommandQueue &queue,
                    const DeviceKeyMesh &mesh,
                    const std::vector<cl::Event> *events,
                    cl::Event *event)
{
    cl::Event verticesEvent, vertexKeysEvent, trianglesEvent;
    const std::size_t numExternalVertices = mesh.numVertices - mesh.numInternalVertices;

    enqueueReadMesh(queue, mesh, tmpMesh, events, &verticesEvent, &vertexKeysEvent, &trianglesEvent);

    trianglesEvent.wait();
    clumps.clear();
    computeLocalComponents(mesh.numVertices, tmpMesh.triangles, tmpClumpId);

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
    vertexKeysEvent.wait();
    for (std::size_t i = 0; i < numExternalVertices; i++)
    {
        if (keyClump[tmpMesh.vertexKeys[i]])
            clumps[tmpClumpId[i + mesh.numInternalVertices]].triangles = true;
    }

    verticesEvent.wait();

    /* Apply clump validity to remove unwanted vertices */
    std::size_t vptr = 0;  // next output vertex in compaction
    std::size_t kptr = 0;  // next output key in compaction
    tmpIndexTable.resize(mesh.numVertices);
    for (std::size_t i = 0; i < mesh.numInternalVertices; i++)
    {
        if (clumps[tmpClumpId[i]].triangles)
        {
            tmpIndexTable[i] = vptr;
            tmpMesh.vertices[vptr++] = tmpMesh.vertices[i];
        }
        else
            tmpIndexTable[i] = 0xFFFFFFFFu;
    }
    for (std::size_t i = mesh.numInternalVertices; i < mesh.numVertices; i++)
    {
        if (clumps[tmpClumpId[i]].triangles)
        {
            tmpIndexTable[i] = vptr;
            tmpMesh.vertices[vptr++] = tmpMesh.vertices[i];
            tmpMesh.vertexKeys[kptr++] = tmpMesh.vertexKeys[i - mesh.numInternalVertices];
        }
        else
            tmpIndexTable[i] = 0xFFFFFFFFu;
    }
    tmpMesh.vertices.resize(vptr);
    tmpMesh.vertexKeys.resize(kptr);
    const std::size_t newInternalVertices = vptr - kptr;

    /* Use clump validity to remove dead triangles and rewrite remaining ones */
    std::size_t tptr = 0;
    for (std::size_t i = 0; i < mesh.numTriangles; i++)
    {
        if (tmpIndexTable[tmpMesh.triangles[i][0]] != 0xFFFFFFFFu)
        {
            for (unsigned int j = 0; j < 3; j++)
                tmpMesh.triangles[tptr][j] = tmpIndexTable[tmpMesh.triangles[i][j]];
            tptr++;
        }
    }
    tmpMesh.triangles.resize(tptr);

    std::size_t newKeys = updateKeyMap(
        nextVertex + newInternalVertices,
        tmpMesh.vertexKeys, tmpClumpId, tmpIndexTable);

    /* Compact the vertex list (again) to keep only new external vertices */
    for (std::size_t i = 0; i < tmpMesh.vertexKeys.size(); i++)
    {
        const cl_uint pos = tmpIndexTable[i];
        if (pos >= nextVertex)
        {
            assert(pos - nextVertex >= newInternalVertices
                   && pos - nextVertex <= newInternalVertices + i);
            tmpMesh.vertices[pos - nextVertex] = tmpMesh.vertices[newInternalVertices + i];
        }
    }

    /* Rewrite triangles (again) to global indices */
    rewriteTriangles(nextVertex, tmpIndexTable, tmpMesh);
    tmpMesh.vertices.resize(newInternalVertices + newKeys);

    writer.writeVertices(nextVertex, tmpMesh.vertices.size(), &tmpMesh.vertices[0][0]);
    writer.writeTriangles(nextTriangle, tmpMesh.triangles.size(), &tmpMesh.triangles[0][0]);
    nextVertex += tmpMesh.vertices.size();
    nextTriangle += tmpMesh.triangles.size();

    if (event != NULL)
        *event = trianglesEvent;
}

void BigMesher::prepareAdd()
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

Marching::OutputFunctor BigMesher::outputFunctor(unsigned int pass)
{
    switch (pass)
    {
    case 0:
        return serializeOutputFunctor(boost::bind(&BigMesher::count, this, _1, _2, _3, _4), mutex);
    case 1:
        prepareAdd();
        return serializeOutputFunctor(boost::bind(&BigMesher::add, this, _1, _2, _3, _4), mutex);
    default:
        abort();
    }
}

void BigMesher::write(FastPly::WriterBase &writer, const std::string &filename,
                      std::ostream *progressStream) const
{
    (void) writer;
    (void) filename;
    (void) progressStream;
    assert(&writer == &this->writer);
    assert(filename == this->filename);
}


StxxlMesher::VertexBuffer::VertexBuffer(FastPly::WriterBase &writer, size_type capacity)
    : writer(writer), nextVertex(0)
{
    buffer.reserve(capacity);
}

void StxxlMesher::VertexBuffer::operator()(const boost::array<float, 3> &vertex)
{
    buffer.push_back(vertex);
    if (buffer.size() == buffer.capacity())
        flush();
}

void StxxlMesher::VertexBuffer::flush()
{
    writer.writeVertices(nextVertex, buffer.size(), &buffer[0][0]);
    nextVertex += buffer.size();
    buffer.clear();
}

StxxlMesher::TriangleBuffer::TriangleBuffer(FastPly::WriterBase &writer, size_type capacity)
    : writer(writer), nextTriangle(0)
{
    nextTriangle = 0;
    buffer.reserve(capacity);
}

void StxxlMesher::TriangleBuffer::operator()(const boost::array<std::tr1::uint32_t, 3> &triangle)
{
    buffer.push_back(triangle);
    if (buffer.size() == buffer.capacity())
        flush();
}

void StxxlMesher::add(
    const cl::CommandQueue &queue,
    const DeviceKeyMesh &mesh,
    const std::vector<cl::Event> *events,
    cl::Event *event)
{
    cl::Event verticesEvent, vertexKeysEvent, trianglesEvent;
    const std::size_t numExternalVertices = mesh.numVertices - mesh.numInternalVertices;
    const size_type priorVertices = this->vertices.size();

    enqueueReadMesh(queue, mesh, tmpMesh, events,
                    &verticesEvent, &vertexKeysEvent, &trianglesEvent);

    trianglesEvent.wait();
    computeLocalComponents(mesh.numVertices, tmpMesh.triangles, tmpClumpId);

    vertexKeysEvent.wait();
    std::size_t newKeys = updateKeyMap(
        priorVertices + mesh.numInternalVertices,
        tmpMesh.vertexKeys, tmpClumpId, tmpIndexTable);

    /* Copy the vertices into storage */
    vertices.reserve(priorVertices + mesh.numInternalVertices + newKeys);
    verticesEvent.wait();
    for (std::size_t i = 0; i < mesh.numInternalVertices; i++)
    {
        vertices.push_back(std::make_pair(tmpMesh.vertices[i], tmpClumpId[i]));
    }
    for (std::size_t i = 0; i < numExternalVertices; i++)
    {
        const cl_uint pos = tmpIndexTable[i];
        if (pos == vertices.size())
        {
            vertices.push_back(std::make_pair(
                    tmpMesh.vertices[mesh.numInternalVertices + i],
                    tmpClumpId[mesh.numInternalVertices + i]));
        }
    }

    rewriteTriangles(priorVertices, tmpIndexTable, tmpMesh);

    // Store the output triangles
    this->triangles.reserve(this->triangles.size() + tmpMesh.triangles.size());
    std::copy(tmpMesh.triangles.begin(), tmpMesh.triangles.end(), std::back_inserter(triangles));

    if (event != NULL)
        *event = trianglesEvent;
}

Marching::OutputFunctor StxxlMesher::outputFunctor(unsigned int pass)
{
    /* only one pass, so ignore it */
    (void) pass;
    assert(pass == 0);

    return serializeOutputFunctor(boost::bind(&StxxlMesher::add, this, _1, _2, _3, _4), mutex);
}

void StxxlMesher::TriangleBuffer::flush()
{
    writer.writeTriangles(nextTriangle, buffer.size(), &buffer[0][0]);
    nextTriangle += buffer.size();
    buffer.clear();
}

void StxxlMesher::write(FastPly::WriterBase &writer, const std::string &filename,
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
        progress.reset(new ProgressDisplay(vertices.size() + triangles.size(), *progressStream));
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
                rewritten[i] = vertexRemapConst[triangle[i]];
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


MesherBase *createMesher(MesherType type, FastPly::WriterBase &writer, const std::string &filename)
{
    switch (type)
    {
    case SIMPLE_MESHER: return new SimpleMesher();
    case WELD_MESHER:   return new WeldMesher();
    case BIG_MESHER:    return new BigMesher(writer, filename);
    case STXXL_MESHER:  return new StxxlMesher();
    }
    return NULL; // should never be reached
}
