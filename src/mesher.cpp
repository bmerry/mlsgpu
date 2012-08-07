/**
 * @file
 *
 * Data structures for storing the output of @ref Marching.
 *
 * @todo Audit for overflows when >2^32 vertices emitted in total but chunked.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif

#ifndef __CL_ENABLE_EXCEPTIONS
# define __CL_ENABLE_EXCEPTIONS
#endif

#include "tr1_cstdint.h"
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
#include "tr1_unordered_map.h"
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
    const Statistics::Container::vector<boost::array<cl_uint, 3> > &triangles,
    Statistics::Container::vector<UnionFind::Node<std::tr1::int32_t> > &nodes)
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
    const Statistics::Container::vector<UnionFind::Node<std::tr1::int32_t> > &nodes,
    const Statistics::Container::vector<boost::array<cl_uint, 3> > &triangles,
    Statistics::Container::vector<clump_id> &clumpId)
{
    std::size_t numVertices = nodes.size();

    // Allocate clumps for the local components
    clumpId.resize(numVertices);
    for (std::size_t i = 0; i < numVertices; i++)
    {
        if (nodes[i].isRoot())
        {
            if (clumps.size() >= boost::make_unsigned<clump_id>::type(std::numeric_limits<clump_id>::max()))
            {
                /* Ideally this would throw, but it's called from a worker
                 * thread and there is no easy way to immediately notify the
                 * master thread that it should shut everything down.
                 */
                std::cerr << "There were too many connected components.\n";
                std::exit(1);
            }
            clumpId[i] = clumps.size();
            clumps.push_back(Clump(chunkGen, nodes[i].size()));
        }
    }

    // Compute clump IDs for the non-root vertices
    for (std::size_t i = 0; i < numVertices; i++)
    {
        std::tr1::int32_t r = UnionFind::findRoot(nodes, i);
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

void KeyMapMesher::updateKeyMaps(
    ChunkId::gen_type chunkGen,
    std::tr1::uint32_t vertexOffset,
    const Statistics::Container::vector<cl_ulong> &keys,
    const Statistics::Container::vector<clump_id> &clumpId,
    Statistics::Container::vector<std::tr1::uint32_t> &indexTable)
{
    const std::size_t numExternalVertices = keys.size();
    const std::size_t numInternalVertices = clumpId.size() - numExternalVertices;
    std::size_t newKeys = 0; // Number of external keys we haven't seen yet in this chunk

    indexTable.resize(numExternalVertices);
    for (std::size_t i = 0; i < numExternalVertices; i++)
    {
        cl_ulong key = keys[i];
        clump_id cid = clumpId[i + numInternalVertices];

        {
            std::pair<Statistics::Container::unordered_map<cl_ulong, clump_id>::const_iterator, bool> added;
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
            std::pair<Statistics::Container::unordered_map<cl_ulong, std::tr1::uint32_t>::const_iterator, bool> added;
            added = vertexIdMap.insert(std::make_pair(key, std::tr1::uint32_t(vertexOffset + newKeys)));
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
}

void KeyMapMesher::rewriteTriangles(
    std::tr1::uint32_t priorVertices,
    const Statistics::Container::vector<std::tr1::uint32_t> &indexTable,
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
    : KeyMapMesher(writer, namer),
    chunkIds("mem.BigMesher::chunkIds"),
    nextVertex(0), nextTriangle(0), pruneThresholdVertices(0),
    retainedExternal("mem.BigMesher::retainedExternal"),
    chunkCounts("mem.BigMesher::chunkCounts"),
    tmpClumpValid("mem.BigMesher::tmpChunkValid")
{
    MLSGPU_ASSERT(writer.supportsOutOfOrder(), std::invalid_argument);
}

void BigMesher::count(const ChunkId &chunkId, MesherWork &work)
{
    if (!curChunkGen || chunkId.gen != *curChunkGen)
    {
        // This is a new chunk
        assert(!curChunkGen || *curChunkGen < chunkId.gen);
        curChunkGen = chunkId.gen;
        chunkIds[chunkId.gen] = chunkId;
        vertexIdMap.clear(); // don't merge vertex IDs with other chunks
    }

    HostKeyMesh &mesh = work.mesh;

    work.trianglesEvent.wait();
    computeLocalComponents(mesh.vertices.size(), mesh.triangles, tmpNodes);
    updateClumps(chunkId.gen, tmpNodes, mesh.triangles, tmpClumpId);

    /* Merge clumps. We don't actually care about vertex indices at this point,
     * so we just pass 0 as the offset. We can't entirely eliminate vertexIdMap
     * through, because the keys tell us which external vertex keys have already
     * occurred in this chunk.
     */
    work.vertexKeysEvent.wait();
    updateKeyMaps(chunkId.gen, 0, mesh.vertexKeys, tmpClumpId, tmpIndexTable);
}

void BigMesher::add(const ChunkId &chunkId, MesherWork &work)
{
    FastPly::WriterBase &writer = getWriter();
    if (!curChunkGen || chunkId.gen != *curChunkGen)
    {
        assert(!curChunkGen || *curChunkGen < chunkId.gen);
        if (writer.isOpen())
            writer.close();

        // Completely skip empty chunks
        const Clump::Counts &counts = chunkCounts[chunkId.gen];
        if (counts.triangles == 0)
            return;

        curChunkGen = chunkId.gen;
        writer.setNumVertices(counts.vertices);
        writer.setNumTriangles(counts.triangles);
        writer.open(getOutputName(chunkId));
        Statistics::getStatistic<Statistics::Counter>("output.files").add();

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
    tmpClumpValid.clear(); // ensures that the resize with zero-fill
    tmpClumpValid.resize(tmpNodes.size());
    for (std::size_t i = 0; i < tmpNodes.size(); i++)
        if (tmpNodes[i].isRoot() && std::tr1::uint64_t(tmpNodes[i].size()) >= pruneThresholdVertices)
            tmpClumpValid[i] = true;

    work.vertexKeysEvent.wait();
    for (std::size_t i = 0; i < numExternalVertices; i++)
    {
        std::tr1::int32_t r = UnionFind::findRoot(tmpNodes, i + numInternalVertices);
        if (!tmpClumpValid[r] && retainedExternal.count(mesh.vertexKeys[i]))
            tmpClumpValid[r] = true;
    }

    work.verticesEvent.wait();
    /* Apply clump validity to remove unwanted vertices, and build an
     * index remap table at the same time.
     */
    const std::tr1::uint32_t badIndex = UINT32_MAX;
    std::tr1::uint32_t vptr = 0;  // next output vertex in compaction
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
            tmpIndexTable[i] = badIndex;
    }
    for (std::size_t i = numInternalVertices; i < mesh.vertices.size(); i++)
    {
        std::tr1::int32_t r = UnionFind::findRoot(tmpNodes, i);
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
            tmpIndexTable[i] = badIndex;
    }
    mesh.vertices.resize(vptr);

    /* Use clump validity to remove dead triangles and rewrite remaining ones */
    std::size_t tptr = 0;
    for (std::size_t i = 0; i < mesh.triangles.size(); i++)
    {
        if (tmpIndexTable[mesh.triangles[i][0]] != badIndex)
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

    std::tr1::uint64_t totalVertices = 0;
    /* Count the vertices */
    BOOST_FOREACH(const Clump &clump, clumps)
    {
        if (clump.isRoot())
        {
            totalVertices += clump.counts.vertices;
        }
    }

    pruneThresholdVertices = std::tr1::uint64_t(totalVertices * getPruneThreshold());
    clump_id keptComponents = 0, totalComponents = 0;

    /* Determine the total number of vertices and triangles to retain */
    BOOST_FOREACH(const Clump &clump, clumps)
    {
        if (clump.isRoot())
        {
            totalComponents++;
            if (clump.counts.vertices >= pruneThresholdVertices)
            {
                typedef std::pair<ChunkId::gen_type, Clump::Counts> item_type;
                BOOST_FOREACH(const item_type &item, clump.chunkCounts)
                {
                    chunkCounts[item.first] += item.second;
                }
                keptComponents++;
            }
        }
    }

    // Overflow checks
    {
        typedef std::pair<ChunkId::gen_type, ChunkId> item_type;
        BOOST_FOREACH(const item_type &item, chunkIds)
        {
            if (chunkCounts[item.first].vertices >= UINT32_MAX)
            {
                std::string name = getOutputName(item.second);
                throw std::overflow_error("Too many vertices for " + name);
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
        std::abort();
        return MesherBase::InputFunctor(); // should never be reached
    }
}

void BigMesher::write(std::ostream *progressStream)
{
    (void) progressStream;
    if (getWriter().isOpen())
        getWriter().close();
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

void StxxlMesher::computeLocalComponents(
    std::size_t numVertices,
    const Statistics::Container::vector<boost::array<cl_uint, 3> > &triangles,
    Statistics::Container::vector<UnionFind::Node<std::tr1::int32_t> > &nodes)
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

void StxxlMesher::updateGlobalClumps(
    const Statistics::Container::vector<UnionFind::Node<std::tr1::int32_t> > &nodes,
    const Statistics::Container::vector<boost::array<cl_uint, 3> > &triangles,
    Statistics::Container::vector<clump_id> &clumpId)
{
    std::size_t numVertices = nodes.size();

    // Allocate clumps for the local components
    clumpId.resize(numVertices);
    for (std::size_t i = 0; i < numVertices; i++)
    {
        if (nodes[i].isRoot())
        {
            if (clumps.size() >= boost::make_unsigned<clump_id>::type(std::numeric_limits<clump_id>::max()))
            {
                /* Ideally this would throw, but it's called from a worker
                 * thread and there is no easy way to immediately notify the
                 * master thread that it should shut everything down.
                 */
                std::cerr << "There were too many connected components.\n";
                std::exit(1);
            }
            clumpId[i] = clumps.size();
            clumps.push_back(Clump(nodes[i].size()));
        }
    }

    // Compute clump IDs for the non-root vertices
    for (std::size_t i = 0; i < numVertices; i++)
    {
        std::tr1::int32_t r = UnionFind::findRoot(nodes, i);
        clumpId[i] = clumpId[r];
    }

    // Compute triangle counts for the clumps
    // TODO: could be more efficient to do this after sorting?
    typedef boost::array<cl_uint, 3> triangle_type;
    BOOST_FOREACH(const triangle_type &triangle, triangles)
    {
        Clump &clump = clumps[clumpId[triangle[0]]];
        clump.counts.triangles++;
    }
}

void StxxlMesher::updateClumpKeyMap(
    const Statistics::Container::vector<cl_ulong> &keys,
    const Statistics::Container::vector<clump_id> &clumpId)
{
    const std::size_t numExternalVertices = keys.size();
    const std::size_t numInternalVertices = clumpId.size() - numExternalVertices;

    for (std::size_t i = 0; i < numExternalVertices; i++)
    {
        cl_ulong key = keys[i];
        clump_id cid = clumpId[i + numInternalVertices];

        {
            std::pair<Statistics::Container::unordered_map<cl_ulong, clump_id>::const_iterator, bool> added;
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
    }
}

void StxxlMesher::updateLocalClumps(
    Chunk &chunk,
    const Statistics::Container::vector<clump_id> &globalClumpId,
    HostKeyMesh &mesh)
{
    const std::size_t numVertices = mesh.vertices.size();
    const std::size_t numInternalVertices = numVertices - mesh.vertexKeys.size();

    // TODO: use tmp* variables for these
    std::vector<std::tr1::uint32_t> vertexLabel(numVertices);
    std::vector<std::tr1::uint32_t> vertexOrder;

    vertexOrder.reserve(numVertices);
    for (std::size_t i = 0; i < numVertices; i++)
        vertexOrder.push_back(i);
    std::sort(vertexOrder.begin(), vertexOrder.end(), VertexCompare(globalClumpId));
    std::sort(mesh.triangles.begin(), mesh.triangles.end(), TriangleCompare(globalClumpId));

    std::size_t nextVertex = 0;
    std::size_t nextTriangle = 0;
    while (nextVertex < numVertices)
    {
        clump_id cid = globalClumpId[vertexOrder[nextVertex]];
        std::size_t lastVertex = nextVertex;
        std::size_t lastTriangle = nextTriangle;
        std::size_t clumpInternalVertices = 0;
        std::size_t clumpExternalVertices = 0;
        do
        {
            std::tr1::uint32_t vid = vertexOrder[nextVertex];
            bool elide = false;
            if (vid >= numInternalVertices)
            {
                std::pair<Chunk::vertex_id_map_type::iterator, bool> added;
                added = chunk.vertexIdMap.insert(
                    std::make_pair(mesh.vertexKeys[vid - numInternalVertices],
                                   ~chunk.numExternalVertices));
                if (added.second)
                {
                    chunk.numExternalVertices++;
                    clumpExternalVertices++;
                }
                else
                    elide = true;
                vertexLabel[vid] = added.first->second;
            }
            else
            {
                vertexLabel[vid] = nextVertex - lastVertex;
                clumpInternalVertices++;
            }

            if (!elide)
                vertices.push_back(mesh.vertices[vid]);

            nextVertex++;
        } while (nextVertex < numVertices && globalClumpId[vertexOrder[nextVertex]] == cid);

        while (nextTriangle < mesh.triangles.size()
               && globalClumpId[mesh.triangles[nextTriangle][0]] == cid)
        {
            boost::array<std::tr1::uint32_t, 3> out;
            for (int j = 0; j < 3; j++)
                out[j] = vertexLabel[mesh.triangles[nextTriangle][j]];
            triangles.push_back(out);
            nextTriangle++;
        }

        std::size_t clumpTriangles = nextTriangle - lastTriangle;
        chunk.clumps.push_back(Chunk::Clump(
                vertices.size() - clumpInternalVertices - clumpExternalVertices,
                clumpInternalVertices,
                clumpExternalVertices,
                triangles.size() - clumpTriangles,
                clumpTriangles,
                cid));
    }
}

void StxxlMesher::add(const ChunkId &chunkId, MesherWork &work)
{
    Chunk &chunk = chunks[chunkId.gen];
    chunk.chunkId = chunkId;

    HostKeyMesh &mesh = work.mesh;

    work.trianglesEvent.wait();
    computeLocalComponents(mesh.vertices.size(), mesh.triangles, tmpNodes);
    updateGlobalClumps(tmpNodes, mesh.triangles, tmpClumpId);

    work.vertexKeysEvent.wait();
    updateClumpKeyMap(mesh.vertexKeys, tmpClumpId);

    work.verticesEvent.wait();
    updateLocalClumps(chunk, tmpClumpId, mesh);
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

void StxxlMesher::write(std::ostream *progressStream)
{
    FastPly::WriterBase &writer = getWriter();

    Statistics::Registry &registry = Statistics::Registry::getInstance();

    // Number of vertices in the mesh, not double-counting chunk boundaries
    std::tr1::uint64_t totalVertices = 0;
    BOOST_FOREACH(const Clump &clump, clumps)
    {
        if (clump.isRoot())
        {
            totalVertices += clump.counts.vertices;
        }
    }
    const std::tr1::uint64_t thresholdVertices = std::tr1::uint64_t(totalVertices * getPruneThreshold());

    clump_id keptComponents = 0, totalComponents = 0;
    std::tr1::uint64_t keptTriangles = 0;
    BOOST_FOREACH(const Clump &clump, clumps)
    {
        if (clump.isRoot())
        {
            totalComponents++;
            if (clump.counts.vertices >= thresholdVertices)
            {
                keptComponents++;
                keptTriangles += clump.counts.triangles;
                typedef std::pair<ChunkId::gen_type, Clump::Counts> item_type;
            }
        }
    }

    registry.getStatistic<Statistics::Variable>("components.vertices.total").add(vertices.size());
    registry.getStatistic<Statistics::Variable>("components.vertices.threshold").add(thresholdVertices);
    registry.getStatistic<Statistics::Variable>("components.total").add(totalComponents);
    registry.getStatistic<Statistics::Variable>("components.kept").add(keptComponents);
    registry.getStatistic<Statistics::Variable>("externalvertices").add(clumpIdMap.size());

    boost::scoped_ptr<ProgressDisplay> progress;
    if (progressStream != NULL)
    {
        *progressStream << "\nWriting file(s)\n";
        progress.reset(new ProgressDisplay(2 * keptTriangles, *progressStream));
    }

    const std::tr1::uint32_t badIndex = std::numeric_limits<std::tr1::uint32_t>::max();

    // Maps from an linear enumeration of all external vertices of a chunk to the
    // final index in the file. It is badIndex for dropped vertices.
    Statistics::Container::vector<std::tr1::uint32_t> externalRemap("mem.mesher.externalRemap");
    // Offset to first vertex of each clump in output file
    Statistics::Container::vector<std::tr1::uint32_t> startVertex("mem.mesher.startVertex");
    for (std::size_t i = 0; i < chunks.size(); i++)
    {
        startVertex.clear();
        externalRemap.clear();

        const Chunk &chunk = chunks[i];
        std::tr1::uint64_t chunkVertices = 0;
        std::tr1::uint64_t chunkTriangles = 0;
        std::size_t chunkExternal = 0;
        for (std::size_t j = 0; j < chunk.clumps.size(); j++)
        {
            const Chunk::Clump &cc = chunk.clumps[j];
            clump_id cid = UnionFind::findRoot(clumps, cc.globalId);
            if (clumps[cid].counts.vertices >= thresholdVertices)
            {
                chunkVertices += cc.numInternalVertices + cc.numExternalVertices;
                chunkExternal += cc.numExternalVertices;
                chunkTriangles += cc.numTriangles;
            }
        }

        if (chunkVertices >= UINT32_MAX)
        {
            std::string name = getOutputName(chunk.chunkId);
            throw std::overflow_error("Too many vertices for " + name);
        }

        if (chunkTriangles > 0)
        {
            writer.setNumVertices(chunkVertices);
            writer.setNumTriangles(chunkTriangles);
            writer.open(getOutputName(chunk.chunkId));
            Statistics::getStatistic<Statistics::Counter>("output.files").add();

            std::tr1::uint32_t writtenVertices = 0;
            // Write out the valid vertices, simultaneously building externalRemap
            VertexBuffer vb(writer, vertices_type::block_size / sizeof(vertices_type::value_type));
            for (std::size_t j = 0; j < chunk.clumps.size(); j++)
            {
                const Chunk::Clump &cc = chunk.clumps[j];
                clump_id cid = UnionFind::findRoot(clumps, cc.globalId);
                startVertex.push_back(writtenVertices);
                if (clumps[cid].counts.vertices >= thresholdVertices)
                {
                    vertices_type::const_iterator v = vertices.cbegin() + cc.firstVertex;
                    for (std::size_t i = 0; i < cc.numInternalVertices; i++, ++v)
                    {
                        vb(*v);
                    }
                    writtenVertices += cc.numInternalVertices;

                    for (std::size_t i = 0; i < cc.numExternalVertices; i++, ++v)
                    {
                        externalRemap.push_back(writtenVertices);
                        vb(*v);
                        ++writtenVertices;
                    }

                    // Yes, numTriangles. That's easier to make add up to the total
                    // than vertices (which share), and still a good proxy.
                    if (progress != NULL)
                        *progress += cc.numTriangles;
                }
                else
                {
                    // appends n copies of badIndex
                    externalRemap.resize(externalRemap.size() + cc.numExternalVertices, badIndex);
                }
            }
            vb.flush();

            // Now write out the triangles
            TriangleBuffer tb(writer, triangles_type::block_size / sizeof(triangles_type::value_type));
            for (std::size_t j = 0; j < chunk.clumps.size(); j++)
            {
                const Chunk::Clump &cc = chunk.clumps[j];
                clump_id cid = UnionFind::findRoot(clumps, cc.globalId);
                if (clumps[cid].counts.vertices >= thresholdVertices)
                {
                    triangles_type::const_iterator tp = triangles.cbegin() + cc.firstTriangle;
                    for (std::size_t i = 0; i < cc.numTriangles; i++, ++tp)
                    {
                        boost::array<std::tr1::uint32_t, 3> t = *tp;
                        for (int k = 0; k < 3; k++)
                        {
                            if (~t[k] < externalRemap.size())
                            {
                                t[k] = externalRemap[~t[k]];
                                assert(t[k] != badIndex);
                            }
                            else
                                t[k] += startVertex[j];
                        }
                        tb(t);
                    }
                    if (progress != NULL)
                        *progress += cc.numTriangles;
                }
            }
            tb.flush();

            writer.close();
        }
    }
}

namespace
{

/**
 * Class implementing @ref deviceMesher.
 */
class DeviceMesher
{
private:
    const MesherBase::InputFunctor in;
    const ChunkId chunkId;

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
        in(chunkId, work);
    }

    DeviceMesher(const MesherBase::InputFunctor &in, const ChunkId &chunkId)
        : in(in), chunkId(chunkId) {}
};

} // anonymous namespace

Marching::OutputFunctor deviceMesher(const MesherBase::InputFunctor &in, const ChunkId &chunkId)
{
    return DeviceMesher(in, chunkId);
}

MesherBase *createMesher(MesherType type, FastPly::WriterBase &writer, const MesherBase::Namer &namer)
{
    switch (type)
    {
    case BIG_MESHER:    return new BigMesher(writer, namer);
    case STXXL_MESHER:  return new StxxlMesher(writer, namer);
    }
    return NULL; // should never be reached
}
