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
