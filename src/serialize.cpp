/**
 * @file
 *
 * Transmission of assorted data structures through MPI.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <mpi.h>
#include "grid.h"
#include "tags.h"

namespace
{

/**
 * Representation of the Grid data type that can safely be turned into an MPI data type.
 */
struct RawGrid
{
    float reference[3];
    float spacing;
    int extents[6]; // x-lo, x-hi, y-lo, y-hi, z-lo, z-hi
};

static MPI_Datatype gridType; /// MPI datatype representing @ref RawGrid

/// Create @ref gridType
static void registerGridType()
{
    int lengths[3] = {3, 1, 6};
    MPI_Aint displacements[3] =
    {
        offsetof(RawGrid, reference),
        offsetof(RawGrid, spacing),
        offsetof(RawGrid, extents)
    };
    MPI_Datatype types[3] = { MPI_FLOAT, MPI_FLOAT, MPI_INT };

    MPI_Type_create_struct(3, lengths, displacements, types, &gridType);
    MPI_Type_commit(&gridType);
}

} // anonymous namespace

namespace Serialize
{

void send(const Grid &grid, MPI_Comm comm, int dest)
{
    RawGrid raw;
    raw.spacing = grid.getSpacing();
    for (int i = 0; i < 3; i++)
    {
        raw.reference[i] = grid.getReference()[i];
        raw.extents[2 * i] = grid.getExtent(i).first;
        raw.extents[2 * i + 1] = grid.getExtent(i).second;
    }

    MPI_Send(&raw, 1, gridType, dest, MLSGPU_TAG_WORK, comm);
}

Grid recv(MPI_Comm comm, int source)
{
    RawGrid raw;
    MPI_Recv(&raw, 1, gridType, source, MLSGPU_TAG_WORK, comm, MPI_STATUS_IGNORE);

    return Grid(raw.reference, raw.spacing,
                raw.extents[0], raw.extents[1],
                raw.extents[2], raw.extents[3],
                raw.extents[4], raw.extents[5]);
}

void init()
{
    registerGridType();
}

} // namespace Serialize
