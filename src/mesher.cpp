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
#include <boost/smart_ptr/shared_ptr.hpp>
#include <boost/smart_ptr/make_shared.hpp>
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
#include <iomanip>
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
    ans["big"] = BIG_MESHER;
    ans["stxxl"] = STXXL_MESHER;
    return ans;
}

std::string ChunkNamer::operator()(const ChunkId &chunkId) const
{
    std::ostringstream nameStream;
    nameStream << baseName;
    for (unsigned int i = 0; i < 3; i++)
        nameStream << '_' << std::setw(4) << std::setfill('0') << chunkId.coords[i];
    nameStream << ".ply";
    return nameStream.str();
}


namespace detail
{

void KeyMapMesher::computeLocalComponents(
    std::size_t numVertices,
    const std::vector<boost::array<cl_uint, 3> > &triangles,
    std::vector<UnionFind::Node<cl_int> > &nodes)
{
    nodes.clear();
    nodes.resize(numVertices);
    typedef boost::array<cl_uint, 3> triangle_type;
    BOOST_FOREACH(const triangle_type &triangle, triangles)
    {
        // Only need to use two edges in the union-find tree, since the
        // third will be redundant.
        for (unsigned int j = 0; j < 2; j++)
            UnionFind::merge(nodes, triangle[j], triangle[j + 1]);
    }
}

void KeyMapMesher::updateClumps(
    unsigned int chunkGen,
    const std::vector<UnionFind::Node<cl_int> > &nodes,
    const std::vector<boost::array<cl_uint, 3> > &triangles,
    std::vector<clump_id> &clumpId)
{
    std::size_t numVertices = nodes.size();

    // Allocate clumps for the local components
    clumpId.resize(numVertices);
    for (std::size_t i = 0; i < numVertices; i++)
    {
        if (nodes[i].isRoot())
        {
            if (clumps.size() > boost::make_unsigned<clump_id>::type(std::numeric_limits<clump_id>::max()))
            {
                throw std::overflow_error("Too many clumps");
            }
            clumpId[i] = clumps.size();
            clumps.push_back(Clump(chunkGen, nodes[i].size()));
        }
    }

    // Compute clump IDs for the non-root vertices
    for (std::size_t i = 0; i < numVertices; i++)
    {
        cl_int r = UnionFind::findRoot(nodes, i);
        clumpId[i] = clumpId[r];
    }

    // Compute triangle counts for the clumps
    // TODO: could be more efficient to first collect triangle count for
    // each clump, then add into clump data structure
    typedef boost::array<cl_uint, 3> triangle_type;
    BOOST_FOREACH(const triangle_type &triangle, triangles)
    {
        Clump &clump = clumps[clumpId[triangle[0]]];
        clump.counts.triangles++;
        clump.chunkCounts[chunkGen].triangles++;
    }
}

std::size_t KeyMapMesher::updateKeyMaps(
    unsigned int chunkGen,
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
        cl_ulong key = hKeys[i];
        clump_id cid = clumpId[i + numInternalVertices];

        {
            std::pair<std::tr1::unordered_map<cl_ulong, clump_id>::iterator, bool> added;
            added = clumpIdMap.insert(std::make_pair(key, cid));
            if (!added.second)
            {
                // Unified two external vertices. Also need to unify their clumps.
                clump_id cid2 = added.first->second;
                UnionFind::merge(clumps, cid, cid2);
                // They will both have counted the common vertex, so we need to
                // subtract it.
                cid = UnionFind::findRoot(clumps, cid);
                clumps[cid].counts.vertices--;
            }
        }

        {
            std::pair<std::tr1::unordered_map<cl_ulong, cl_uint>::iterator, bool> added;
            added = vertexIdMap.insert(std::make_pair(key, vertexOffset + newKeys));
            if (added.second)
                newKeys++; // key has not been set yet in this chunk
            else
            {
                // When we merged the clumps above, we adjusted the global vertex count,
                // but we also need to adjust the per-chunk vertex count because the
                // vertex has already appeared in this chunk.
                assert(clumps[cid].isRoot());
                assert(clumps[cid].chunkCounts.count(chunkGen));
                clumps[cid].chunkCounts[chunkGen].vertices--;
            }
            indexTable[i] = added.first->second;
        }
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

BigMesher::BigMesher(FastPly::WriterBase &writer, const Namer &namer)
    : writer(writer), namer(namer),
    nextVertex(0), nextTriangle(0), pruneThresholdVertices(0)
{
    MLSGPU_ASSERT(writer.supportsOutOfOrder(), std::invalid_argument);
}

void BigMesher::count(const ChunkId &chunkId, MesherWork &work)
{
    if (!curChunkGen || chunkId.gen != *curChunkGen)
    {
        assert(!curChunkGen || *curChunkGen < chunkId.gen);
        curChunkGen = chunkId.gen;
        vertexIdMap.clear(); // don't merge vertex IDs with other chunks
    }

    HostKeyMesh &mesh = work.mesh;

    work.trianglesEvent.wait();
    computeLocalComponents(mesh.vertices.size(), mesh.triangles, tmpNodes);
    updateClumps(chunkId.gen, tmpNodes, mesh.triangles, tmpClumpId);

    /* Update clumps. We don't actually care about vertex indices at this point,
     * so we pass 0 as the offset.
     */
    work.vertexKeysEvent.wait();
    updateKeyMaps(chunkId.gen, 0, mesh.vertexKeys, tmpClumpId, tmpIndexTable);
}

void BigMesher::add(const ChunkId &chunkId, MesherWork &work)
{
    if (!curChunkGen || chunkId.gen != *curChunkGen)
    {
        assert(!curChunkGen || *curChunkGen < chunkId.gen);

        // Completely skip empty chunks
        const Clump::Counts &counts = chunkCounts[chunkId.gen];
        if (counts.triangles == 0)
            return;

        if (writer.isOpen())
            writer.close();
        curChunkGen = chunkId.gen;
        writer.setNumVertices(counts.vertices);
        writer.setNumTriangles(counts.triangles);
        writer.open(namer(chunkId));

        nextVertex = 0;
        nextTriangle = 0;
        vertexIdMap.clear(); // don't merge vertex IDs with other chunks
    }

    HostKeyMesh &mesh = work.mesh;
    const std::size_t numExternalVertices = mesh.vertexKeys.size();
    const std::size_t numInternalVertices = mesh.vertices.size() - numExternalVertices;

    work.trianglesEvent.wait();
    clumps.clear();
    computeLocalComponents(mesh.vertices.size(), mesh.triangles, tmpNodes);

    /* Determine which components are valid. A clump is valid if either it has
     * the requisite number of vertices on its own, or if it contains an
     * external vertex that has been marked as valid.
     */
    tmpClumpValid.clear();
    tmpClumpValid.resize(tmpNodes.size());
    for (std::size_t i = 0; i < tmpNodes.size(); i++)
        if (tmpNodes[i].isRoot() && std::tr1::uint64_t(tmpNodes[i].size()) >= pruneThresholdVertices)
            tmpClumpValid[i] = true;

    work.vertexKeysEvent.wait();
    for (std::size_t i = 0; i < numExternalVertices; i++)
    {
        cl_int r = UnionFind::findRoot(tmpNodes, i + numInternalVertices);
        if (!tmpClumpValid[r] && retainedExternal.count(mesh.vertexKeys[i]))
            tmpClumpValid[r] = true;
    }

    work.verticesEvent.wait();
    /* Apply clump validity to remove unwanted vertices, and build an
     * index remap table at the same time.
     */
    std::size_t vptr = 0;  // next output vertex in compaction
    tmpIndexTable.resize(mesh.vertices.size());
    for (std::size_t i = 0; i < numInternalVertices; i++)
    {
        cl_int r = UnionFind::findRoot(tmpNodes, i);
        if (tmpClumpValid[r])
        {
            tmpIndexTable[i] = nextVertex + vptr;
            mesh.vertices[vptr++] = mesh.vertices[i];
        }
        else
            tmpIndexTable[i] = 0xFFFFFFFFu;
    }
    for (std::size_t i = numInternalVertices; i < mesh.vertices.size(); i++)
    {
        cl_int r = UnionFind::findRoot(tmpNodes, i);
        if (tmpClumpValid[r])
        {
            cl_ulong key = mesh.vertexKeys[i - numInternalVertices];
            std::pair<vertex_id_map_type::iterator, bool> added;
            added = vertexIdMap.insert(std::make_pair(key, nextVertex + vptr));
            if (added.second)
                mesh.vertices[vptr++] = mesh.vertices[i];
            tmpIndexTable[i] = added.first->second;
        }
        else
            tmpIndexTable[i] = 0xFFFFFFFFu;
    }
    mesh.vertices.resize(vptr);

    /* Use clump validity to remove dead triangles and rewrite remaining ones */
    std::size_t tptr = 0;
    for (std::size_t i = 0; i < mesh.triangles.size(); i++)
    {
        if (tmpIndexTable[mesh.triangles[i][0]] != 0xFFFFFFFFu)
        {
            for (unsigned int j = 0; j < 3; j++)
                mesh.triangles[tptr][j] = tmpIndexTable[mesh.triangles[i][j]];
            tptr++;
        }
    }
    mesh.triangles.resize(tptr);

    writer.writeVertices(nextVertex, mesh.vertices.size(), &mesh.vertices[0][0]);
    writer.writeTriangles(nextTriangle, mesh.triangles.size(), &mesh.triangles[0][0]);
    nextVertex += mesh.vertices.size();
    nextTriangle += mesh.triangles.size();
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
            totalVertices += clump.counts.vertices;
        }
    }

    pruneThresholdVertices = (cl_uint) (totalVertices * getPruneThreshold());
    clump_id keptComponents = 0, totalComponents = 0;

    /* Determine the total number of vertices and triangles to retain */
    BOOST_FOREACH(const Clump &clump, clumps)
    {
        if (clump.isRoot())
        {
            totalComponents++;
            if (clump.counts.vertices >= pruneThresholdVertices)
            {
                typedef std::pair<unsigned int, Clump::Counts> item_type;
                BOOST_FOREACH(const item_type &item, clump.chunkCounts)
                {
                    chunkCounts[item.first] += item.second;
                }
                keptComponents++;
            }
        }
    }

    registry.getStatistic<Statistics::Variable>("components.total").add(totalComponents);
    registry.getStatistic<Statistics::Variable>("components.kept").add(keptComponents);
    registry.getStatistic<Statistics::Variable>("externalvertices").add(clumpIdMap.size());

    /* Determine which external vertices belong to valid clumps. */
    retainedExternal.clear();
    for (clump_id_map_type::const_iterator i = clumpIdMap.begin(); i != clumpIdMap.end(); ++i)
    {
        clump_id cid = i->second;
        cid = UnionFind::findRoot(clumps, cid);
        if (clumps[cid].counts.vertices >= pruneThresholdVertices)
            retainedExternal.insert(i->first);
    }

    nextVertex = 0;
    nextTriangle = 0;

    // Throw away data that is not needed anymore, or that will be rewritten in next pass
    vertexIdMap.clear();
    clumpIdMap.clear();
    clumps.clear();
    curChunkGen = boost::none;
}

MesherBase::InputFunctor BigMesher::functor(unsigned int pass)
{
    switch (pass)
    {
    case 0:
        return boost::bind(&BigMesher::count, this, _1, _2);
    case 1:
        prepareAdd();
        return boost::bind(&BigMesher::add, this, _1, _2);
    default:
        abort();
    }
}

void BigMesher::write(FastPly::WriterBase &writer, const Namer &namer,
                      std::ostream *progressStream) const
{
    (void) namer;
    (void) progressStream;
    assert(&writer == &this->writer);
    if (writer.isOpen())
        writer.close();
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

void StxxlMesher::add(const ChunkId &chunkId, MesherWork &work)
{
    if (chunks.empty() || chunkId.gen != chunks.back().chunkId.gen)
    {
        assert(chunks.empty() || chunks.back().chunkId.gen < chunkId.gen);
        Chunk chunk;
        chunk.chunkId = chunkId;
        chunk.firstVertex = vertices.size();
        chunk.firstTriangle = triangles.size();
        chunks.push_back(chunk);

        vertexIdMap.clear(); // don't merge vertex IDs with other chunks
    }

    HostKeyMesh &mesh = work.mesh;
    const std::size_t numExternalVertices = mesh.vertexKeys.size();
    const std::size_t numInternalVertices = mesh.vertices.size() - numExternalVertices;
    const std::tr1::uint64_t priorVertices = vertices.size() - chunks.back().firstVertex;

    work.trianglesEvent.wait();
    computeLocalComponents(mesh.vertices.size(), mesh.triangles, tmpNodes);
    updateClumps(chunkId.gen, tmpNodes, mesh.triangles, tmpClumpId);

    work.vertexKeysEvent.wait();
    updateKeyMaps(
        chunkId.gen,
        priorVertices + numInternalVertices,
        mesh.vertexKeys, tmpClumpId, tmpIndexTable);

    /* Copy the vertices into storage */
    work.verticesEvent.wait();
    for (std::size_t i = 0; i < numInternalVertices; i++)
    {
        vertices.push_back(std::make_pair(mesh.vertices[i], tmpClumpId[i]));
    }
    for (std::size_t i = 0; i < numExternalVertices; i++)
    {
        const cl_uint pos = tmpIndexTable[i];
        if (pos >= priorVertices)
        {
            vertices.push_back(std::make_pair(
                    mesh.vertices[numInternalVertices + i],
                    tmpClumpId[numInternalVertices + i]));
        }
    }

    rewriteTriangles(priorVertices, tmpIndexTable, mesh);

    // Store the output triangles
    std::copy(mesh.triangles.begin(), mesh.triangles.end(), std::back_inserter(triangles));
}

MesherBase::InputFunctor StxxlMesher::functor(unsigned int pass)
{
    /* only one pass, so ignore it */
    (void) pass;
    assert(pass == 0);

    return boost::bind(&StxxlMesher::add, this, _1, _2);
}

void StxxlMesher::TriangleBuffer::flush()
{
    writer.writeTriangles(nextTriangle, buffer.size(), &buffer[0][0]);
    nextTriangle += buffer.size();
    buffer.clear();
}

void StxxlMesher::write(FastPly::WriterBase &writer, const Namer &namer,
                        std::ostream *progressStream) const
{
    Statistics::Registry &registry = Statistics::Registry::getInstance();

    const cl_uint thresholdVertices = (cl_uint) (vertices.size() * getPruneThreshold());
    FastPly::WriterBase::size_type numVertices = 0, numTriangles = 0;
    clump_id keptComponents = 0, totalComponents = 0;
    BOOST_FOREACH(const Clump &clump, clumps)
    {
        if (clump.isRoot())
        {
            totalComponents++;
            if (clump.counts.vertices >= thresholdVertices)
            {
                numVertices += clump.counts.vertices;
                numTriangles += clump.counts.triangles;
                keptComponents++;
            }
        }
    }
    registry.getStatistic<Statistics::Variable>("components.vertices.total").add(vertices.size());
    registry.getStatistic<Statistics::Variable>("components.vertices.threshold").add(thresholdVertices);
    registry.getStatistic<Statistics::Variable>("components.total").add(totalComponents);
    registry.getStatistic<Statistics::Variable>("components.kept").add(keptComponents);
    registry.getStatistic<Statistics::Variable>("externalvertices").add(clumpIdMap.size());

    stxxl::VECTOR_GENERATOR<cl_uint, 4, 16>::result vertexRemap;
    const stxxl::VECTOR_GENERATOR<cl_uint, 4, 16>::result & vertexRemapConst = vertexRemap;
    const cl_uint badIndex = std::numeric_limits<cl_uint>::max();

    boost::scoped_ptr<ProgressDisplay> progress;
    if (progressStream != NULL)
    {
        // TODO: count is wrong because of duplication at chunk boundaries
        *progressStream << "\nWriting file\n";
        progress.reset(new ProgressDisplay(vertices.size() + triangles.size(), *progressStream));
    }

    for (std::size_t i = 0; i < chunks.size(); i++)
    {
        vertexRemap.clear();

        const Chunk &chunk = chunks[i];
        std::tr1::uint64_t lastVertex, lastTriangle;
        if (i + 1 == chunks.size())
        {
            lastVertex = vertices.size();
            lastTriangle = triangles.size();
        }
        else
        {
            lastVertex = chunks[i + 1].firstVertex;
            lastTriangle = chunks[i + 1].firstTriangle;
        }
        unsigned int gen = chunk.chunkId.gen;
        std::tr1::uint64_t chunkVertices = 0;
        std::tr1::uint64_t chunkTriangles = 0;
        BOOST_FOREACH(const Clump &clump, clumps)
        {
            // TODO: save a list of these clumps
            if (clump.isRoot() && clump.counts.vertices >= thresholdVertices)
            {
                std::tr1::unordered_map<unsigned int, Clump::Counts>::const_iterator pos;
                pos = clump.chunkCounts.find(gen);
                if (pos != clump.chunkCounts.end())
                {
                    chunkVertices += pos->second.vertices;
                    chunkTriangles += pos->second.triangles;
                }
            }
        }
        // TODO: bail out if chunkVertices too large

        if (chunkTriangles > 0)
        {
            writer.setNumVertices(chunkVertices);
            writer.setNumTriangles(chunkTriangles);
            writer.open(namer(chunk.chunkId));

            cl_uint nextVertex = 0;
            {
                stxxl::stream::streamify_traits<vertices_type::const_iterator>::stream_type
                    vertex_stream = stxxl::stream::streamify(vertices.begin() + chunk.firstVertex,
                                                             vertices.begin() + lastVertex);
                VertexBuffer vb(writer, vertices_type::block_size / sizeof(vertices_type::value_type));
                while (!vertex_stream.empty())
                {
                    vertices_type::value_type vertex = *vertex_stream;
                    ++vertex_stream;
                    clump_id clumpId = UnionFind::findRoot(clumps, vertex.second);
                    if (clumps[clumpId].counts.vertices >= thresholdVertices)
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
            assert(nextVertex == chunkVertices);

            {
                stxxl::stream::streamify_traits<triangles_type::const_iterator>::stream_type
                    triangle_stream = stxxl::stream::streamify(triangles.begin() + chunk.firstTriangle,
                                                               triangles.begin() + lastTriangle);
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
            writer.close();
        }
    }
}

namespace
{

class DeviceMesher
{
private:
    const MesherBase::InputFunctor in;

public:
    typedef void result_type;

    void operator()(
        const cl::CommandQueue &queue,
        const DeviceKeyMesh &mesh,
        const std::vector<cl::Event> *events,
        cl::Event *event) const
    {
        MesherWork work;
        std::vector<cl::Event> wait(3);
        enqueueReadMesh(queue, mesh, work.mesh, events, &wait[0], &wait[1], &wait[2]);
        CLH::enqueueMarkerWithWaitList(queue, &wait, event);

        work.verticesEvent = wait[0];
        work.vertexKeysEvent = wait[1];
        work.trianglesEvent = wait[2];
        in(ChunkId(), work); // TODO: should be able to pass through a ChunkId
    }

    DeviceMesher(const MesherBase::InputFunctor &in) : in(in) {}
};

} // anonymous namespace

Marching::OutputFunctor deviceMesher(const MesherBase::InputFunctor &in)
{
    return DeviceMesher(in);
}

MesherBase *createMesher(MesherType type, FastPly::WriterBase &writer, const MesherBase::Namer &namer)
{
    switch (type)
    {
    case BIG_MESHER:    return new BigMesher(writer, namer);
    case STXXL_MESHER:  return new StxxlMesher();
    }
    return NULL; // should never be reached
}
