/**
 * @file
 *
 * Mesher that uses MPI to parallelise the final writeback.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <mpi.h>
#include <boost/smart_ptr/scoped_ptr.hpp>
#include <boost/smart_ptr/scoped_array.hpp>
#include <boost/exception/all.hpp>
#include <boost/thread/thread.hpp>
#include "statistics.h"
#include "allocator.h"
#include "mesher.h"
#include "mesher_mpi.h"
#include "progress_mpi.h"
#include "fast_ply_mpi.h"
#include "serialize.h"

OOCMesherMPI::OOCMesherMPI(
    FastPly::WriterMPI &writer, const Namer &namer,
    MPI_Comm comm, int root)
    : OOCMesher(writer, namer), comm(comm), root(root)
{
    if (rank != root)
        retainFiles = true; // Only the master deletes files
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
}

std::size_t OOCMesherMPI::write(Timeplot::Worker &tworker, std::ostream *progressStream)
{
    Timeplot::Action writeAction("write", tworker, "finalize.time");
    FastPly::WriterMPI &writer = static_cast<FastPly::WriterMPI &>(getWriter());
    std::size_t outputFiles = 0;

    finalize(tworker);
    if (rank == root)
    {
        std::ostringstream dump;
        boost::archive::text_oarchive archive(dump);
        archive << *this;
        std::string serial = dump.str();

        std::size_t len = serial.size();
        MPI_Bcast(&len, 1, Serialize::mpi_type_traits<std::size_t>::type(), root, comm);
        MPI_Bcast(const_cast<char *>(serial.data()), len, MPI_CHAR, root, comm);
    }
    else
    {
        std::size_t len;
        MPI_Bcast(&len, 1, Serialize::mpi_type_traits<std::size_t>::type(), root, comm);
        boost::scoped_array<char> serial(new char[len]);
        MPI_Bcast(serial.get(), len, MPI_CHAR, root, comm);

        std::istringstream dump(std::string(serial.get(), len));
        serial.reset();
        boost::archive::text_iarchive archive(dump);
        archive >> *this;
    }

    boost::scoped_ptr<BinaryReader> verticesTmpRead(createReader(SYSCALL_READER));
    verticesTmpRead->open(tmpWriter.getVerticesPath());
    boost::scoped_ptr<BinaryReader> trianglesTmpRead(createReader(SYSCALL_READER));
    trianglesTmpRead->open(tmpWriter.getTrianglesPath());

    std::tr1::uint64_t thresholdVertices;
    clump_id keptComponents;
    std::tr1::uint64_t keptVertices, keptTriangles;
    getStatistics(thresholdVertices, keptComponents, keptVertices, keptTriangles, rank == root);

    std::size_t asyncMem = getAsyncMem(thresholdVertices);

    boost::scoped_ptr<ProgressDisplay> progressDisplay;
    boost::scoped_ptr<ProgressMPI> progress;
    boost::scoped_ptr<boost::thread> progressThread;
    if (progressStream != NULL)
    {
        if (rank == root)
        {
            *progressStream << "\nWriting file(s)\n";
            progressDisplay.reset(new ProgressDisplay(2 * keptTriangles, *progressStream));
        }
        progress.reset(new ProgressMPI(progressDisplay.get(), 2 * keptTriangles, comm, root));
        if (rank == root)
            progressThread.reset(new boost::thread(boost::ref(*progress)));
    }

    /* Maps from an linear enumeration of all external vertices of a chunk to
     * the final index in the file. It is badIndex for dropped vertices,
     * although that is not actually used and could be skipped. This is declared
     * outside the loop purely to facilitate memory reuse.
     */
    Statistics::Container::PODBuffer<std::tr1::uint32_t> externalRemap("mem.OOCMesher::externalRemap");
    // Offset to first vertex of each clump in output file
    Statistics::Container::PODBuffer<std::tr1::uint32_t> startVertex("mem.OOCMesher::startVertex");
    // Offset to first triangle of each clump in output file
    Statistics::Container::PODBuffer<FastPly::Writer::size_type> startTriangle("mem.OOCMesher::startTriangle");
    Statistics::Container::PODBuffer<triangle_type> triangles("mem.OOCMesher::triangles");

    AsyncWriter asyncWriter(1, asyncMem * 2); // * 2 to allow overlapping

    /* When there are many chunks, we can simplify partition over chunks, and avoid worrying
     * about interference between output files. When there aren't enough chunks we have to
     * partition over clumps within each chunk. A hybrid scheme would avoid the need for
     * two code-paths, but would itself get very complicated due to file opening and closing
     * being collective operations. Either all files would have to be opened up-front by
     * all processes (which could easily exceed OS limits on open files) or there would need
     * to be complex logic to create a communicator for each subset of processes that share
     * access to a file, and to sequence the operations to avoid stalls.
     */
    bool perChunk = (chunks.size() >= (std::size_t) size);
    std::size_t firstChunk, lastChunk;

    if (perChunk)
    {
        asyncWriter.start();
        firstChunk = mulDiv(chunks.size(), rank, size);
        lastChunk = mulDiv(chunks.size(), rank + 1, size);
    }
    else
    {
        firstChunk = 0;
        lastChunk = chunks.size();
    }

    for (std::size_t i = firstChunk; i < lastChunk; i++)
    {
        const Chunk &chunk = chunks[i];
        std::tr1::uint64_t chunkVertices, chunkTriangles, chunkExternal;
        // Note: chunkExternal includes discarded clumps, the others exclude them
        getChunkStatistics(thresholdVertices, chunk, chunkVertices, chunkTriangles, chunkExternal);

        if (chunkTriangles > 0)
        {
            const std::string filename = getOutputName(chunk.chunkId);
            try
            {
                if (!perChunk)
                    asyncWriter.start();
                writer.setNumVertices(chunkVertices);
                writer.setNumTriangles(chunkTriangles);
                if (perChunk)
                    writer.open(filename);
                else
                    writer.open(filename, comm, root);
                outputFiles++;

                writeChunkPrepare(
                    chunk, thresholdVertices, chunkExternal,
                    startVertex, startTriangle, externalRemap);

                std::size_t first, last;
                if (perChunk)
                {
                    first = 0;
                    last = chunk.clumps.size();
                }
                else
                {
                    first = mulDiv(chunk.clumps.size(), rank, size);
                    last = mulDiv(chunk.clumps.size(), rank + 1, size);
                }

                writeChunkVertices(
                    tworker, *verticesTmpRead, asyncWriter, chunk,
                    thresholdVertices, startVertex.data(), progress.get(),
                    first, last);

                writeChunkTriangles(
                    tworker, *trianglesTmpRead, asyncWriter, chunk,
                    thresholdVertices, chunkExternal,
                    startVertex.data(), startTriangle.data(), externalRemap.data(),
                    triangles, progress.get(),
                    first, last);

                writer.close();
                if (!perChunk)
                    asyncWriter.stop();
            }
            catch (std::ios::failure &e)
            {
                throw boost::enable_error_info(e)
                    << boost::errinfo_file_name(filename)
                    << boost::errinfo_errno(errno);
            }
        }
    }

    if (perChunk)
        asyncWriter.stop();
    if (progress)
        progress->sync();

    if (perChunk)
    {
        std::size_t totalOutputFiles = 0;
        MPI_Reduce(&outputFiles, &totalOutputFiles, 1,
                   Serialize::mpi_type_traits<std::size_t>::type(),
                   MPI_SUM, root, comm);
        outputFiles = totalOutputFiles;
    }
    if (rank == root)
    {
        progressThread->join();
        Statistics::getStatistic<Statistics::Counter>("output.files").add(outputFiles);
    }

    // To ensure that the timer gives useful information
    MPI_Barrier(comm);
    return outputFiles;
}
