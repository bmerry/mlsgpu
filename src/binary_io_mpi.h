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
 * Binary file output for MPI.
 */

#ifndef BINARY_IO_MPI_H
#define BINARY_IO_MPI_H

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <cstddef>
#include <boost/filesystem/path.hpp>
#include <mpi.h>
#include "binary_io.h"

/**
 * Binary writer for MPI. The @ref open, @ref close and @ref resize operations
 * are collective, while writes may be made independently. Atomic mode is not
 * used, so to guarantee consistency it is required that no writes overlap.
 */
class BinaryWriterMPI : public BinaryWriter
{
public:
    explicit BinaryWriterMPI(MPI_Comm comm);
    virtual ~BinaryWriterMPI();

private:
    MPI_Comm comm;    ///< Communicator that will be used to open the file
    MPI_File handle;  ///< File handle when it is open

    virtual void openImpl(const boost::filesystem::path &path);
    virtual void closeImpl();
    virtual std::size_t writeImpl(const void *buf, std::size_t count, offset_type offset) const;
    virtual void resizeImpl(offset_type size) const;
};

#endif /* BINARY_IO_MPI_H */
