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
    MPI_Comm comm;
    MPI_File handle;

    virtual void openImpl(const boost::filesystem::path &path);
    virtual void closeImpl();
    virtual std::size_t writeImpl(const void *buf, std::size_t count, offset_type offset) const;
    virtual void resizeImpl(offset_type size) const;
};

#endif /* BINARY_IO_MPI_H */
