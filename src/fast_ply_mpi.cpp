/**
 * @file
 *
 * PLY writer for collective MPI operation.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <boost/smart_ptr/shared_ptr.hpp>
#include <boost/smart_ptr/make_shared.hpp>
#include <boost/bind.hpp>
#include <string>
#include <mpi.h>
#include "fast_ply.h"
#include "fast_ply_mpi.h"
#include "serialize.h"
#include "binary_io_mpi.h"
#include "errors.h"

namespace FastPly
{

static boost::shared_ptr<BinaryWriter> makeHandle()
{
    return boost::make_shared<BinaryWriterMPI>(MPI_COMM_SELF);
}

WriterMPI::WriterMPI() : Writer(makeHandle)
{
}

void WriterMPI::open(const std::string &filename, MPI_Comm comm, int root)
{
    MLSGPU_ASSERT(!isOpen(), state_error);

    std::string header;
    size_type sizes[3]; // header size, num vertices, num triangles
    int rank;
    MPI_Comm_rank(comm, &rank);
    if (rank == root)
    {
        header = makeHeader();
        sizes[0] = header.size();
        sizes[1] = getNumVertices();
        sizes[2] = getNumTriangles();
    }
    MPI_Bcast(sizes, 3, Serialize::mpi_type_traits<size_type>::type(), root, comm);
    setNumVertices(sizes[1]);
    setNumTriangles(sizes[2]);

    handle = boost::make_shared<BinaryWriterMPI>(comm);
    handle->open(filename);
    handle->resize(sizes[0] + getNumVertices() * vertexSize + getNumTriangles() * triangleSize);
    if (rank == root)
        handle->write(header.data(), header.size(), 0);
    vertexStart = sizes[0];
    triangleStart = vertexStart + getNumVertices() * vertexSize;
}

} // namespace FastPly
