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
 */
class WriterMPI : public Writer
{
public:
    WriterMPI(MPI_Comm comm, int root);

    void open(const std::string &filename);

private:
    MPI_Comm comm;
    int root;

    static boost::shared_ptr<BinaryWriter> makeHandle(MPI_Comm comm);
};

} // namespace FastPly

#endif /* !FAST_PLY_MPI_H */
