/**
 * @file
 *
 * Transmission of assorted data structures through MPI.
 */

#ifndef SERIALIZE_H
#define SERIALIZE_H

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <mpi.h>

/* Forward declaration */
class Grid;
struct ChunkId;
struct MesherWork;
namespace Bucket { struct Recursion; }
namespace SplatSet { class VectorSet; }

/**
 * Transmission of assorted data structures through MPI.
 *
 * Each of the @c send functions sends one object to a single destination,
 * while the @c recv functions can receive from either a named destination or
 * @c MPI_ANY_SOURCE. The sends are all blocking standard-mode. All communications
 * use @ref MLSGPU_TAG_WORK.
 *
 * Before using any of the @c send or @c recv functions, one must first call
 * @ref init.
 */
namespace Serialize
{

void send(const Grid &grid, MPI_Comm comm, int dest);
void recv(Grid &grid, MPI_Comm comm, int source);

void send(const Bucket::Recursion &recursion, MPI_Comm comm, int dest);
void recv(Bucket::Recursion &recursion, MPI_Comm comm, int source);

void send(const ChunkId &chunkId, MPI_Comm comm, int dest);
void recv(ChunkId &chunkId, MPI_Comm comm, int source);

void send(const SplatSet::VectorSet &splats, MPI_Comm comm, int dest);
void recv(SplatSet::VectorSet &splats, MPI_Comm comm, int source);

void send(const MesherWork &work, MPI_Comm comm, int dest);
void recv(MesherWork &work, MPI_Comm comm, int source);

/**
 * Registers MPI data types. This must be called before any of the send or
 * receive functions.
 */
void init();

} // namespace Serialize

#endif /* !SERIALIZE_H */
