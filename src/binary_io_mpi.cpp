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
    int rank;
    MPI_Comm_rank(comm, &rank);
    MPI_File_open(comm, const_cast<char *>(path.string().c_str()),
                  MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &handle);
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
}
