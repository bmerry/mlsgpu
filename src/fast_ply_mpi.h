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
 * PLY writer for collective MPI operation.
 */

#ifndef FAST_PLY_MPI_H
#define FAST_PLY_MPI_H

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <boost/smart_ptr/shared_ptr.hpp>
#include <string>
#include <mpi.h>
#include "fast_ply.h"

namespace FastPly
{

/**
 * Variation on @ref FastPly::Writer that correctly handles writing the header
 * in only one rank. Additionally, the comments and vertex/triangle counts are
 * taken from the root rank.
 *
 * The base class @ref open(const std::string&) will create a file on a single node
 * (i.e. using @c MPI_COMM_SELF). To have parallel writes to a single file you
 * must use @ref open(const std::string&, MPI_Comm, int).
 */
class WriterMPI : public Writer
{
public:
    /// Constructor
    WriterMPI();

    using Writer::open;

    /**
     * Collectively open a file for parallel writing.
     * @param filename        Filename.
     * @param comm            Intracommunicator for the collective operation.
     * @param root            Rank that will write the file header.
     */
    void open(const std::string &filename, MPI_Comm comm, int root);
};

} // namespace FastPly

#endif /* !FAST_PLY_MPI_H */
