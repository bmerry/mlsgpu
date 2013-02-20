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
#include <boost/smart_ptr/scoped_array.hpp>
#include <boost/smart_ptr/shared_ptr.hpp>
#include <boost/smart_ptr/make_shared.hpp>
#include <boost/system/error_code.hpp>
#include <boost/foreach.hpp>
#include <boost/type_traits/make_unsigned.hpp>
#include <boost/exception/all.hpp>
#include <boost/filesystem.hpp>
#include <boost/system/error_code.hpp>
#include <boost/iostreams/positioning.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include "tr1_unordered_map.h"
#include <cassert>
#include <cstdlib>
#include <utility>
#include <iterator>
#include <map>
#include <string>
#include <ostream>
#include <iomanip>
#include <cerrno>
#include "mesher.h"
#include "fast_ply.h"
#include "logging.h"
#include "errors.h"
#include "progress.h"
#include "union_find.h"
#include "statistics.h"
#include "clh.h"
#include "misc.h"
#include "circular_buffer.h"
#include "binary_io.h"

std::map<std::string, MesherType> MesherTypeWrapper::getNameMap()
{
    std::map<std::string, MesherType> ans;
    ans["ooc"] = OOC_MESHER;
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


OOCMesher::TmpWriterItem::TmpWriterItem()
    : vertices("mem.OOCMesher::TmpWriterItem::vertices"),
    triangles("mem.OOCMesher::TmpWriterItem::triangles"),
    vertexRanges("mem.OOCMesher::TmpWriterItem::vertexRanges"),
    triangleRanges("mem.OOCMesher::TmpWriterItem::triangleRanges")
{
}

void OOCMesher::TmpWriterWorker::operator()(TmpWriterItem &item)
{
    Timeplot::Action timer("compute", getTimeplotWorker(), owner.getComputeStat());
    typedef std::pair<std::size_t, std::size_t> range;
    BOOST_FOREACH(const range &r, item.vertexRanges)
    {
        verticesFile.write(reinterpret_cast<char *>(&item.vertices[r.first]),
                           (r.second - r.first) * sizeof(vertex_type));
    }
    BOOST_FOREACH(const range &r, item.triangleRanges)
    {
        trianglesFile.write(reinterpret_cast<char *>(&item.triangles[r.first]),
                            (r.second - r.first) * sizeof(triangle_type));
    }
    if (!verticesFile || !trianglesFile)
    {
        Log::log[Log::error] << "Failed while writing temporary files: "
            << boost::system::errc::make_error_code((boost::system::errc::errc_t) errno).message() << std::endl;
        std::exit(1);
    }
}

OOCMesher::TmpWriterWorkerGroup::TmpWriterWorkerGroup(std::size_t slots)
    : WorkerGroup<TmpWriterItem, TmpWriterWorker, TmpWriterWorkerGroup>("tmpwriter", 1),
    itemAllocator("mem.OOCMesher::TmpWriterWorkerGroup::itemAllocator", slots)
{
    addWorker(new TmpWriterWorker(*this, verticesFile, trianglesFile));
    for (std::size_t i = 0; i < itemAllocator.size(); i++)
        itemPool.push_back(boost::make_shared<TmpWriterItem>());
}

void OOCMesher::TmpWriterWorkerGroup::start()
{
    createTmpFile(verticesPath, verticesFile);
    createTmpFile(trianglesPath, trianglesFile);
    WorkerGroup<TmpWriterItem, TmpWriterWorker, TmpWriterWorkerGroup>::start();
}

void OOCMesher::TmpWriterWorkerGroup::stopPostJoin()
{
    verticesFile.close();
    trianglesFile.close();
    if (!verticesFile || !trianglesFile)
    {
        Log::log[Log::error] << "Failed while writing temporary files: "
            << boost::system::errc::make_error_code((boost::system::errc::errc_t) errno).message() << std::endl;
        std::exit(1);
    }
}

boost::shared_ptr<OOCMesher::TmpWriterItem> OOCMesher::TmpWriterWorkerGroup::get(Timeplot::Worker &tworker, std::size_t size)
{
    (void) size;
    CircularBufferBase::Allocation alloc = itemAllocator.allocate(tworker, 1, &getStat);
    boost::shared_ptr<TmpWriterItem> item = itemPool[alloc.get()];
    item->alloc = alloc;
    return item;
}

void OOCMesher::TmpWriterWorkerGroup::freeItem(boost::shared_ptr<TmpWriterItem> item)
{
    item->vertices.clear();
    item->triangles.clear();
    item->vertexRanges.clear();
    item->triangleRanges.clear();
    itemAllocator.free(item->alloc);
}

const int OOCMesher::reorderSlots = 3;

OOCMesher::OOCMesher(FastPly::Writer &writer, const Namer &namer)
    : MesherBase(writer, namer),
    tmpNodes("mem.OOCMesher::tmpNodes"),
    tmpClumpId("mem.OOCMesher::tmpClumpId"),
    tmpVertexLabel("mem.OOCMesher::tmpVertexLabel"),
    tmpFirstVertex("mem.OOCMesher::tmpFirstVertex"),
    tmpNextVertex("mem.OOCMesher::tmpNextVertex"),
    tmpFirstTriangle("mem.OOCMesher::tmpFirstTriangle"),
    tmpNextTriangle("mem.OOCMesher::tmpNextTriangle"),
    tmpWriter(reorderSlots),
    retainFiles(false),
    chunks("mem.OOCMesher::chunks"),
    clumps("mem.OOCMesher::clumps"),
    clumpIdMap("mem.OOCMesher::clumpIdMap")
{
}

OOCMesher::~OOCMesher()
{
    if (tmpWriter.running())
        tmpWriter.stop();

    if (!retainFiles)
    {
        boost::filesystem::path verticesTmpPath = tmpWriter.getVerticesPath();
        boost::filesystem::path trianglesTmpPath = tmpWriter.getTrianglesPath();
        if (!verticesTmpPath.empty())
        {
            boost::system::error_code ec;
            remove(verticesTmpPath, ec);
            if (ec)
                Log::log[Log::warn] << "Could not delete " << verticesTmpPath.string() << ": " << ec.message() << std::endl;
        }
        if (!trianglesTmpPath.empty())
        {
            boost::system::error_code ec;
            remove(trianglesTmpPath, ec);
            if (ec)
                Log::log[Log::warn] << "Could not delete " << trianglesTmpPath.string() << ": " << ec.message() << std::endl;
        }
    }
}

void OOCMesher::computeLocalComponents(
    std::size_t numVertices,
    std::size_t numTriangles,
    const triangle_type *triangles,
    Statistics::Container::vector<UnionFind::Node<std::tr1::int32_t> > &nodes)
{
    nodes.clear();
    nodes.resize(numVertices);
    for (std::size_t i = 0; i < numTriangles; i++)
    {
        // Only need to use two edges in the union-find tree, since the
        // third will be redundant.
        for (unsigned int j = 0; j < 2; j++)
            UnionFind::merge(nodes, triangles[i][j], triangles[i][j + 1]);
    }
}

void OOCMesher::updateGlobalClumps(
    std::size_t numTriangles,
    const Statistics::Container::vector<UnionFind::Node<std::tr1::int32_t> > &nodes,
    const triangle_type *triangles,
    Statistics::Container::PODBuffer<clump_id> &clumpId)
{
    std::size_t numVertices = nodes.size();

    // Allocate clumps for the local components
    clumpId.reserve(numVertices, false);
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
    for (std::size_t i = 0; i < numTriangles; i++)
    {
        Clump &clump = clumps[clumpId[triangles[i][0]]];
        clump.triangles++;
    }
}

void OOCMesher::updateClumpKeyMap(
    std::size_t numVertices,
    std::size_t numExternalVertices,
    const cl_ulong *keys,
    const Statistics::Container::PODBuffer<clump_id> &clumpId)
{
    const std::size_t numInternalVertices = numVertices - numExternalVertices;

    for (std::size_t i = 0; i < numExternalVertices; i++)
    {
        cl_ulong key = keys[i];
        clump_id cid = clumpId[i + numInternalVertices];

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
            clumps[cid].vertices--;
        }
    }
}

void OOCMesher::flushBuffer(Timeplot::Worker &tworker)
{
    if (!reorderBuffer)
        return;
    Statistics::Timer flushTimer("mesher.flush");
    BOOST_FOREACH(Chunk &chunk, chunks)
    {
        if (!chunk.bufferedClumps.empty())
        {
            BOOST_FOREACH(const Chunk::Clump &clump, chunk.bufferedClumps)
            {
                const std::size_t numVertices = clump.numInternalVertices + clump.numExternalVertices;
                const std::tr1::uint64_t firstVertex = writtenVerticesTmp;
                const std::tr1::uint64_t firstTriangle = writtenTrianglesTmp;
                reorderBuffer->vertexRanges.push_back(std::make_pair(
                        clump.firstVertex, clump.firstVertex + numVertices));
                reorderBuffer->triangleRanges.push_back(std::make_pair(
                        clump.firstTriangle, clump.firstTriangle + clump.numTriangles));
                writtenVerticesTmp += numVertices;
                writtenTrianglesTmp += clump.numTriangles;
                chunk.clumps.push_back(Chunk::Clump(
                    firstVertex,
                    clump.numInternalVertices,
                    clump.numExternalVertices,
                    firstTriangle,
                    clump.numTriangles,
                    clump.globalId));
            }
            chunk.bufferedClumps.clear();
        }
    }
    tmpWriter.push(tworker, reorderBuffer);
    reorderBuffer.reset();
}

void OOCMesher::updateLocalClumps(
    Chunk &chunk,
    const Statistics::Container::PODBuffer<clump_id> &globalClumpId,
    clump_id clumpIdFirst,
    clump_id clumpIdLast,
    HostKeyMesh &mesh,
    Timeplot::Worker &tworker)
{
    const std::size_t numVertices = mesh.numVertices();
    const std::size_t numInternalVertices = mesh.numInternalVertices();
    const clump_id numClumps = clumpIdLast - clumpIdFirst;

    tmpFirstVertex.reserve(numClumps, false);
    std::fill(tmpFirstVertex.data(), tmpFirstVertex.data() + numClumps, -1);
    tmpFirstTriangle.reserve(numClumps, false);
    std::fill(tmpFirstTriangle.data(), tmpFirstTriangle.data() + numClumps, -1);
    tmpNextVertex.reserve(numVertices, false);
    tmpNextTriangle.reserve(mesh.numTriangles(), false);

    for (std::tr1::int32_t i = (std::tr1::int32_t) numVertices - 1; i >= 0; i--)
    {
        clump_id cid = globalClumpId[i] - clumpIdFirst;
        tmpNextVertex[i] = tmpFirstVertex[cid];
        tmpFirstVertex[cid] = i;
    }

    for (std::tr1::int32_t i = (std::tr1::int32_t) mesh.numTriangles() - 1; i >= 0; i--)
    {
        clump_id cid = globalClumpId[mesh.triangles[i][0]] - clumpIdFirst;
        tmpNextTriangle[i] = tmpFirstTriangle[cid];
        tmpFirstTriangle[cid] = i;
    }

    tmpVertexLabel.reserve(numVertices, false);

    if (reorderBuffer)
    {
        if ((numVertices + reorderBuffer->vertices.size()) * sizeof(vertex_type)
            + (mesh.numTriangles() + reorderBuffer->triangles.size()) * sizeof(triangle_type)
            > getReorderCapacity() / reorderSlots)
            flushBuffer(tworker);
    }
    if (!reorderBuffer)
        reorderBuffer = tmpWriter.get(tworker, 1);

    for (clump_id gid = clumpIdFirst; gid < clumpIdLast; gid++)
    {
        clump_id cid = gid - clumpIdFirst;

        // These count *emitted* vertices - which for external vertices can be less than
        // incoming ones due to sharing within the chunk.
        std::size_t clumpInternalVertices = 0;
        std::size_t clumpExternalVertices = 0;
        std::size_t clumpTriangles = 0;
        for (std::tr1::int32_t vid = tmpFirstVertex[cid]; vid != -1; vid = tmpNextVertex[vid])
        {
            bool elide = false; // true if the vertex is elided due to sharing
            if (std::size_t(vid) >= numInternalVertices)
            {
                // external vertex
                std::pair<Chunk::vertex_id_map_type::iterator, bool> added;
                added = chunk.vertexIdMap.insert(
                    std::make_pair(mesh.vertexKeys[vid - numInternalVertices],
                                   (std::tr1::uint32_t) ~chunk.numExternalVertices));
                if (added.second)
                {
                    chunk.numExternalVertices++;
                    clumpExternalVertices++;
                }
                else
                    elide = true;
                tmpVertexLabel[vid] = added.first->second;
            }
            else
            {
                // internal vertex
                tmpVertexLabel[vid] = clumpInternalVertices++;
            }

            if (!elide)
                reorderBuffer->vertices.push_back(mesh.vertices[vid]);
        }

        // tmpVertexLabel now contains the intermediate encoded ID for each vertex.
        // Transform and emit the triangles using this mapping.
        for (std::tr1::int32_t tid = tmpFirstTriangle[cid]; tid != -1; tid = tmpNextTriangle[tid])
        {
            triangle_type out;
            for (int j = 0; j < 3; j++)
                out[j] = tmpVertexLabel[mesh.triangles[tid][j]];
            reorderBuffer->triangles.push_back(out);
            clumpTriangles++;
        }

        chunk.bufferedClumps.push_back(Chunk::Clump(
                reorderBuffer->vertices.size() - clumpInternalVertices - clumpExternalVertices,
                clumpInternalVertices,
                clumpExternalVertices,
                reorderBuffer->triangles.size() - clumpTriangles,
                clumpTriangles,
                gid));
    }
}

void OOCMesher::add(MesherWork &work, Timeplot::Worker &tworker)
{
    if (work.chunkId.gen >= chunks.size())
        chunks.resize(work.chunkId.gen + 1);
    Chunk &chunk = chunks[work.chunkId.gen];
    chunk.chunkId = work.chunkId;

    HostKeyMesh &mesh = work.mesh;

    if (work.hasEvents)
        work.trianglesEvent.wait();
    clump_id oldClumps = clumps.size();
    computeLocalComponents(mesh.numVertices(), mesh.numTriangles(), mesh.triangles, tmpNodes);
    updateGlobalClumps(mesh.numTriangles(), tmpNodes, mesh.triangles, tmpClumpId);

    if (work.hasEvents)
        work.vertexKeysEvent.wait();
    updateClumpKeyMap(mesh.numVertices(), mesh.numExternalVertices(), mesh.vertexKeys, tmpClumpId);

    if (work.hasEvents)
        work.verticesEvent.wait();
    updateLocalClumps(chunk, tmpClumpId, oldClumps, clumps.size(), mesh, tworker);
}

MesherBase::InputFunctor OOCMesher::functor(unsigned int pass)
{
    /* only one pass, so ignore it */
    (void) pass;
    assert(pass == 0);

    writtenVerticesTmp = 0;
    writtenTrianglesTmp = 0;
    tmpWriter.start();

    return boost::bind(&OOCMesher::add, this, _1, _2);
}

void OOCMesher::finalize(Timeplot::Worker &tworker)
{
    flushBuffer(tworker);
    if (tmpWriter.running())
        tmpWriter.stop();
}

void OOCMesher::write(Timeplot::Worker &tworker, std::ostream *progressStream)
{
    Timeplot::Action writeAction("write", tworker, "finalize.time");
    FastPly::Writer &writer = getWriter();
    Statistics::Registry &registry = Statistics::Registry::getInstance();

    finalize(tworker);

    boost::scoped_ptr<BinaryReader> verticesTmpRead(createReader(SYSCALL_READER));
    verticesTmpRead->open(tmpWriter.getVerticesPath());
    boost::scoped_ptr<BinaryReader> trianglesTmpRead(createReader(SYSCALL_READER));
    trianglesTmpRead->open(tmpWriter.getTrianglesPath());

    // Number of vertices in the mesh, not double-counting chunk boundaries
    std::tr1::uint64_t totalVertices = 0;
    BOOST_FOREACH(const Clump &clump, clumps)
    {
        if (clump.isRoot())
        {
            totalVertices += clump.vertices;
        }
    }
    const std::tr1::uint64_t thresholdVertices = std::tr1::uint64_t(totalVertices * getPruneThreshold());

    clump_id keptComponents = 0, totalComponents = 0;
    std::tr1::uint64_t keptTriangles = 0;
    std::tr1::uint64_t keptVertices = 0;
    BOOST_FOREACH(const Clump &clump, clumps)
    {
        if (clump.isRoot())
        {
            totalComponents++;
            if (clump.vertices >= thresholdVertices)
            {
                keptComponents++;
                keptVertices += clump.vertices;
                keptTriangles += clump.triangles;
            }
        }
    }

    registry.getStatistic<Statistics::Variable>("components.vertices.total").add(writtenVerticesTmp);
    registry.getStatistic<Statistics::Variable>("components.vertices.threshold").add(thresholdVertices);
    registry.getStatistic<Statistics::Variable>("components.vertices.kept").add(keptVertices);
    registry.getStatistic<Statistics::Variable>("components.triangles.kept").add(keptTriangles);
    registry.getStatistic<Statistics::Variable>("components.total").add(totalComponents);
    registry.getStatistic<Statistics::Variable>("components.kept").add(keptComponents);
    registry.getStatistic<Statistics::Variable>("externalvertices").add(clumpIdMap.size());
    Statistics::Variable &readVerticesStat = registry.getStatistic<Statistics::Variable>("write.readVertices.time");
    Statistics::Variable &readTrianglesStat = registry.getStatistic<Statistics::Variable>("write.readTriangles.time");

    boost::scoped_ptr<ProgressDisplay> progress;
    if (progressStream != NULL)
    {
        *progressStream << "\nWriting file(s)\n";
        progress.reset(new ProgressDisplay(2 * keptTriangles, *progressStream));
    }

    const std::tr1::uint32_t badIndex = std::numeric_limits<std::tr1::uint32_t>::max();

    /* Maps from an linear enumeration of all external vertices of a chunk to
     * the final index in the file. It is badIndex for dropped vertices,
     * although that is not actually used and could be skipped. This is declared
     * outside the loop purely to facilitate memory reuse.
     */
    Statistics::Container::vector<std::tr1::uint32_t> externalRemap("mem.OOCMesher::externalRemap");
    // Offset to first vertex of each clump in output file
    Statistics::Container::vector<std::tr1::uint32_t> startVertex("mem.OOCMesher::startVertex");

    Statistics::Container::PODBuffer<vertex_type> vertices("mem.OOCMesher::vertices");

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
            if (clumps[cid].vertices >= thresholdVertices)
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
            const std::string filename = getOutputName(chunk.chunkId);
            try
            {
                writer.setNumVertices(chunkVertices);
                writer.setNumTriangles(chunkTriangles);
                writer.open(filename);
                Statistics::getStatistic<Statistics::Counter>("output.files").add();

                {
                    Statistics::Timer verticesTimer("finalize.vertices.time");

                    std::tr1::uint32_t writtenVertices = 0;
                    // Write out the valid vertices, simultaneously building externalRemap
                    for (std::size_t j = 0; j < chunk.clumps.size(); j++)
                    {
                        const Chunk::Clump &cc = chunk.clumps[j];
                        clump_id cid = UnionFind::findRoot(clumps, cc.globalId);
                        startVertex.push_back(writtenVertices);
                        if (clumps[cid].vertices >= thresholdVertices)
                        {
                            std::size_t numVertices = cc.numInternalVertices + cc.numExternalVertices;
                            vertices.reserve(numVertices, false);
                            {
                                Statistics::Timer timer(readVerticesStat);
                                verticesTmpRead->read(
                                    vertices.data(),
                                    numVertices * sizeof(vertex_type),
                                    cc.firstVertex * sizeof(vertex_type));
                            }
                            writer.writeVertices(writtenVertices, numVertices, &vertices[0][0]);

                            writtenVertices += cc.numInternalVertices;
                            for (std::size_t i = 0; i < cc.numExternalVertices; i++)
                            {
                                externalRemap.push_back(writtenVertices);
                                ++writtenVertices;
                            }

                            // Yes, numTriangles. That's easier to make add up to the total
                            // than vertices (which share), and still a good indicator
                            // of progress.
                            if (progress != NULL)
                                *progress += cc.numTriangles;
                        }
                        else
                        {
                            // appends n copies of badIndex
                            externalRemap.resize(externalRemap.size() + cc.numExternalVertices, badIndex);
                        }
                    }
                }

                {
                    Statistics::Timer trianglesTimer("finalize.triangles.time");
                    std::tr1::uint32_t externalBoundary = ~externalRemap.size();

                    // Determine where to write the triangles
                    FastPly::Writer::size_type writtenTriangles = 0;
                    std::vector<FastPly::Writer::size_type> chunkPos;
                    chunkPos.reserve(chunk.clumps.size());
                    for (std::size_t j = 0; j < chunk.clumps.size(); j++)
                    {
                        const Chunk::Clump &cc = chunk.clumps[j];
                        clump_id cid = UnionFind::findRoot(clumps, cc.globalId);
                        chunkPos.push_back(writtenTriangles);
                        if (clumps[cid].vertices >= thresholdVertices)
                        {
                            writtenTriangles += cc.numTriangles;
                        }
                    }

                    // Now write out the triangles
#ifdef _OPENMP
#pragma omp parallel shared(trianglesTmpRead, chunk, readTrianglesStat, externalBoundary, externalRemap, chunkPos, startVertex, writer, progress) default(none)
#endif
                    {
                        Statistics::Container::PODBuffer<triangle_type> triangles("mem.OOCMesher::triangles");
                        Statistics::Container::PODBuffer<std::tr1::uint8_t> trianglesRaw("mem.OOCMesher::trianglesRaw");
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 1)
#endif
                        for (std::size_t j = 0; j < chunk.clumps.size(); j++)
                        {
                            const Chunk::Clump &cc = chunk.clumps[j];
                            clump_id cid = UnionFind::findRoot(clumps, cc.globalId);
                            std::tr1::uint32_t offset = startVertex[j];
                            if (clumps[cid].vertices >= thresholdVertices)
                            {
                                triangles.reserve(cc.numTriangles, false);
                                trianglesRaw.reserve(cc.numTriangles * FastPly::Writer::triangleSize, false);
                                {
                                    Statistics::Timer timer(readTrianglesStat);
                                    trianglesTmpRead->read(
                                        triangles.data(),
                                        cc.numTriangles * sizeof(triangle_type),
                                        cc.firstTriangle * sizeof(triangle_type));
                                }
                                for (std::size_t ti = 0; ti < cc.numTriangles; ti++)
                                {
                                    triangle_type t = triangles[ti];
                                    // Convert indices to account for compaction
                                    for (int k = 0; k < 3; k++)
                                    {
                                        if (t[k] > externalBoundary)
                                        {
                                            t[k] = externalRemap[~t[k]];
                                        }
                                        else
                                            t[k] += offset;
                                    }
                                    trianglesRaw[ti * FastPly::Writer::triangleSize] = 3;
                                    std::memcpy(trianglesRaw.data() + ti * FastPly::Writer::triangleSize + 1, &t, sizeof(t));
                                }
                                writer.writeTrianglesRaw(chunkPos[j], cc.numTriangles, trianglesRaw.data());
                                if (progress != NULL)
                                    *progress += cc.numTriangles;
                            }
                        }
                    }
                }

                writer.close();
            }
            catch (std::ios::failure &e)
            {
                throw boost::enable_error_info(e)
                    << boost::errinfo_file_name(filename)
                    << boost::errinfo_errno(errno);
            }
        }
    }
}

void OOCMesher::checkpoint(Timeplot::Worker &tworker, const boost::filesystem::path &path)
{
    retainFiles = true;
    finalize(tworker);

    try
    {
        boost::filesystem::ofstream dump(path);
        if (!dump)
            throw std::ios::failure("Could not open file");
        boost::archive::text_oarchive archive(dump);
        archive << *this;
        dump.close();
    }
    catch (std::ios::failure &e)
    {
        throw boost::enable_error_info(e)
            << boost::errinfo_errno(errno)
            << boost::errinfo_file_name(path.string());
    }
}

void OOCMesher::resume(
    Timeplot::Worker &tworker,
    const boost::filesystem::path &path,
    std::ostream *progressStream)
{
    retainFiles = true; // to allow resume to be re-run
    try
    {
        boost::filesystem::ifstream dump(path);
        if (!dump)
            throw std::ios::failure("Could not open file");
        boost::archive::text_iarchive archive(dump);
        archive >> *this;
        dump.close();
    }
    catch (std::ios::failure &e)
    {
        throw boost::enable_error_info(e)
            << boost::errinfo_errno(errno)
            << boost::errinfo_file_name(path.string());
    }
    write(tworker, progressStream);
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
    Timeplot::Worker &tworker;

public:
    typedef void result_type;

    void operator()(
        const cl::CommandQueue &queue,
        const DeviceKeyMesh &mesh,
        const std::vector<cl::Event> *events,
        cl::Event *event) const
    {
        MesherWork work;

        boost::scoped_array<char> buffer(new char[mesh.getHostBytes()]);
        work.mesh = HostKeyMesh(buffer.get(), mesh);
        std::vector<cl::Event> wait(3);
        enqueueReadMesh(queue, mesh, work.mesh, events, &wait[0], &wait[1], &wait[2]);
        CLH::enqueueMarkerWithWaitList(queue, &wait, event);

        work.chunkId = chunkId;
        work.hasEvents = true;
        work.verticesEvent = wait[0];
        work.vertexKeysEvent = wait[1];
        work.trianglesEvent = wait[2];
        in(work, tworker);
    }

    DeviceMesher(const MesherBase::InputFunctor &in, const ChunkId &chunkId, Timeplot::Worker &tworker)
        : in(in), chunkId(chunkId), tworker(tworker) {}
};

} // anonymous namespace

Marching::OutputFunctor deviceMesher(const MesherBase::InputFunctor &in, const ChunkId &chunkId, Timeplot::Worker &tworker)
{
    return DeviceMesher(in, chunkId, tworker);
}

MesherBase *createMesher(MesherType type, FastPly::Writer &writer, const MesherBase::Namer &namer)
{
    switch (type)
    {
    case OOC_MESHER:  return new OOCMesher(writer, namer);
    }
    return NULL; // should never be reached
}
