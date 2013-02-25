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

void OOCMesherMPI::write(Timeplot::Worker &tworker, std::ostream *progressStream)
{
    Timeplot::Action writeAction("write", tworker, "finalize.time");
    FastPly::Writer &writer = getWriter();

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
    getStatistics(thresholdVertices, keptComponents, keptVertices, keptTriangles);

    std::size_t asyncMem = getAsyncMem(thresholdVertices);

    boost::scoped_ptr<ProgressDisplay> progressDisplay;
    boost::scoped_ptr<ProgressMeter> progress;
    if (progressStream != NULL)
    {
        if (rank == root)
        {
            *progressStream << "\nWriting file(s)\n";
            progressDisplay.reset(new ProgressDisplay(2 * keptTriangles, *progressStream));
        }
        progress.reset(new ProgressMPI(progressDisplay.get(), 2 * keptTriangles, comm, root));
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

    for (std::size_t i = 0; i < chunks.size(); i++)
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
                asyncWriter.start();
                writer.setNumVertices(chunkVertices);
                writer.setNumTriangles(chunkTriangles);
                writer.open(filename);
                Statistics::getStatistic<Statistics::Counter>("output.files").add();

                writeChunkPrepare(
                    chunk, thresholdVertices, chunkExternal,
                    startVertex, startTriangle, externalRemap);

                writeChunkVertices(
                    tworker, *verticesTmpRead, asyncWriter, chunk,
                    thresholdVertices, startVertex.data(), progress.get(),
                    rank, size);

                writeChunkTriangles(
                    tworker, *trianglesTmpRead, asyncWriter, chunk,
                    thresholdVertices, chunkExternal,
                    startVertex.data(), startTriangle.data(), externalRemap.data(),
                    triangles, progress.get(),
                    rank, size);

                writer.close();
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
}
