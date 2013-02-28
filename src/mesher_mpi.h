/**
 * @file
 *
 * Mesher that uses MPI to parallelise the final writeback.
 */

#ifndef MESHER_MPI_H
#define MESHER_MPI_H

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <mpi.h>
#include "fast_ply_mpi.h"
#include "mesher.h"

/**
 * Mesher that uses MPI to parallelise the final writeback. Only one rank should
 * actually collect data, but @ref write is a collective operation that
 * distributes the writing process across all ranks.
 */
class OOCMesherMPI : public OOCMesher
{
public:
    /**
     * Constructor.
     *
     * @param writer         Writer that will be used to emit output files.
     * @param namer          Callback function to assign names to output files.
     * @param comm           Intracommunicator for the collective group.
     * @param root           Rank which contains the data to write.
     */
    OOCMesherMPI(FastPly::WriterMPI &writer, const Namer &namer, MPI_Comm comm, int root);

    /**
     * @copydoc OOCMesher::write
     *
     * @note The @a progressStream will only display progress on the root rank.
     */
    virtual std::size_t write(Timeplot::Worker &tworker, std::ostream *progressStream = NULL);

private:
    MPI_Comm comm;  ///< Communicator for the group that will participate
    int root;       ///< Rank which contains the data to write
    int rank;       ///< Self rank
    int size;       ///< Size of the communicator
};

#endif /* !MESHER_MPI_H */
