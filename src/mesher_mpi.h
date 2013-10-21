/*
 * mlsgpu: surface reconstruction from point clouds
 * Copyright (C) 2013  University of Cape Town
 *
 * This file is part of mlsgpu.
 *
 * mlsgpu is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

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
