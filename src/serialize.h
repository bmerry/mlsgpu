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

class Grid;

namespace Serialize
{

void send(const Grid &grid, MPI_Comm comm, int dest);
void recv(const Grid &grid, MPI_Comm comm, int source);

/**
 * Registers MPI data types. This must be called before any of the send or
 * receive functions.
 */
void init();

} // namespace Serialize

#endif /* !SERIALIZE_H */
