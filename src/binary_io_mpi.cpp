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

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <cstddef>
#include <boost/filesystem/path.hpp>
#include <mpi.h>
#include "binary_io_mpi.h"

BinaryWriterMPI::BinaryWriterMPI(MPI_Comm comm)
    : comm(comm)
{
}

BinaryWriterMPI::~BinaryWriterMPI()
{
    if (isOpen())
        close();
}

void BinaryWriterMPI::openImpl(const boost::filesystem::path &path)
{
    MPI_File_open(comm, const_cast<char *>(path.string().c_str()),
                  MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &handle);
    MPI_File_set_atomicity(handle, false);
}

void BinaryWriterMPI::closeImpl()
{
    MPI_File_close(&handle);
}

std::size_t BinaryWriterMPI::writeImpl(const void *buf, std::size_t count, offset_type offset) const
{
    MPI_File_write_at(handle, offset, const_cast<void *>(buf), count, MPI_BYTE, MPI_STATUS_IGNORE);
    return count;
}

void BinaryWriterMPI::resizeImpl(offset_type size) const
{
    MPI_File_set_size(handle, size);
    MPI_File_sync(handle);
}
