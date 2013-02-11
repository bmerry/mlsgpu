/**
 * @file
 *
 * Transmission of assorted data structures through MPI.
 */

#ifndef __CL_ENABLE_EXCEPTIONS
# define __CL_ENABLE_EXCEPTIONS
#endif

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <mpi.h>
#include <cassert>
#include "grid.h"
#include "bucket.h"
#include "tags.h"
#include "serialize.h"
#include "mesher.h"
#include "mesh.h"

namespace
{

/// Representation of @ref Grid that can safely be turned into an MPI data type.
struct RawGrid
{
    float reference[3];
    float spacing;
    Grid::difference_type extents[6]; // x-lo, x-hi, y-lo, y-hi, z-lo, z-hi
};

static MPI_Datatype gridType; ///< MPI datatype representing @ref RawGrid

/// Create @ref gridType
static void registerGridType()
{
    int lengths[5] = {1, 3, 1, 6, 1};
    MPI_Aint displacements[5] =
    {
        0,
        offsetof(RawGrid, reference),
        offsetof(RawGrid, spacing),
        offsetof(RawGrid, extents),
        sizeof(RawGrid)
    };
    MPI_Datatype types[5] = { MPI_LB, MPI_FLOAT, MPI_FLOAT, Serialize::mpi_type_traits<Grid::difference_type>::type(), MPI_UB };

    MPI_Type_create_struct(5, lengths, displacements, types, &gridType);
    MPI_Type_set_name(gridType, const_cast<char *>("RawGrid"));
    MPI_Type_commit(&gridType);
}

/**
 * Scalar fields from @ref SplatSet::SubsetBase.
 */
struct SubsetMetadata
{
    std::size_t size;
    SplatSet::splat_id first, last;
    SplatSet::splat_id prev;
    SplatSet::splat_id nSplats;
    SplatSet::splat_id nRanges;
};

static MPI_Datatype subsetMetadataType; ///< MPI datatype representing @ref SubsetMetadata

static void registerSubsetMetadataType()
{
    int lengths[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    MPI_Aint displacements[8] =
    {
        0,
        offsetof(SubsetMetadata, size),
        offsetof(SubsetMetadata, first),
        offsetof(SubsetMetadata, last),
        offsetof(SubsetMetadata, prev),
        offsetof(SubsetMetadata, nSplats),
        offsetof(SubsetMetadata, nRanges),
        sizeof(SubsetMetadata)
    };
    MPI_Datatype types[8] =
    {
        MPI_LB,
        Serialize::mpi_type_traits<std::size_t>::type(),
        Serialize::mpi_type_traits<SplatSet::splat_id>::type(),
        Serialize::mpi_type_traits<SplatSet::splat_id>::type(),
        Serialize::mpi_type_traits<SplatSet::splat_id>::type(),
        Serialize::mpi_type_traits<SplatSet::splat_id>::type(),
        Serialize::mpi_type_traits<SplatSet::splat_id>::type(),
        MPI_UB
    };
    MPI_Type_create_struct(8, lengths, displacements, types, &subsetMetadataType);
    MPI_Type_set_name(subsetMetadataType, const_cast<char *>("SubsetMetadata"));
    MPI_Type_commit(&subsetMetadataType);
}

static MPI_Datatype chunkIdType; ///< MPI datatype representing @ref ChunkId

static void registerChunkIdType()
{
    int lengths[4] = {1, 1, 3, 1};
    MPI_Aint displacements[4] =
    {
        0,
        offsetof(ChunkIdPod, gen),
        offsetof(ChunkIdPod, coords),
        sizeof(ChunkIdPod)
    };
    MPI_Datatype types[4] =
    {
        MPI_LB,
        Serialize::mpi_type_traits<ChunkIdPod::gen_type>::type(),
        Serialize::mpi_type_traits<Grid::size_type>::type(),
        MPI_UB
    };

    MPI_Type_create_struct(4, lengths, displacements, types, &chunkIdType);
    MPI_Type_set_name(chunkIdType, const_cast<char *>("ChunkId"));
    MPI_Type_commit(&chunkIdType);
}

static MPI_Datatype splatType; ///< MPI datatype representing @ref Splat

static void registerSplatType()
{
    int lengths[4] = {3, 1, 3, 1};
    int displacements[4] =
    {
        offsetof(Splat, position) / sizeof(float),
        offsetof(Splat, radius) / sizeof(float),
        offsetof(Splat, normal) / sizeof(float),
        offsetof(Splat, quality) / sizeof(float)
    };

    MPI_Type_indexed(4, lengths, displacements, MPI_FLOAT, &splatType);
    MPI_Type_set_name(splatType, const_cast<char *>("Splat"));
    MPI_Type_commit(&splatType);

    /* Check that we didn't miss any new fields */
    MPI_Aint lb, extent;
    MPI_Type_get_extent(splatType, &lb, &extent);
    assert(lb == 0);
    assert(extent == sizeof(Splat));
}

} // anonymous namespace

namespace Serialize
{

/**
 * Helper class that other classes can make a friend to allow access to the
 * internals, without the class having to be aware of MPI_Comm.
 */
class Access
{
public:
    static void send(const SplatSet::SubsetBase &subset, MPI_Comm comm, int dest);
    static void recv(SplatSet::SubsetBase &subset, MPI_Comm comm, int source);
};

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

void recv(Grid &grid, MPI_Comm comm, int source)
{
    RawGrid raw;
    MPI_Recv(&raw, 1, gridType, source, MLSGPU_TAG_WORK, comm, MPI_STATUS_IGNORE);

    grid = Grid(raw.reference, raw.spacing,
                raw.extents[0], raw.extents[1],
                raw.extents[2], raw.extents[3],
                raw.extents[4], raw.extents[5]);
}

void send(const ChunkIdPod &chunkId, MPI_Comm comm, int dest)
{
    MPI_Send(const_cast<ChunkIdPod *>(&chunkId), 1, chunkIdType, dest, MLSGPU_TAG_WORK, comm);
}

void recv(ChunkIdPod &chunkId, MPI_Comm comm, int source)
{
    MPI_Recv(&chunkId, 1, chunkIdType, source, MLSGPU_TAG_WORK, comm, MPI_STATUS_IGNORE);
}

void send(const SplatSet::SubsetBase &subset, MPI_Comm comm, int dest)
{
    Access::send(subset, comm, dest);
}

void recv(SplatSet::SubsetBase &subset, MPI_Comm comm, int source)
{
    Access::recv(subset, comm, source);
}

void Access::send(const SplatSet::SubsetBase &subset, MPI_Comm comm, int dest)
{
    SubsetMetadata metadata;
    metadata.size = subset.splatRanges.size();
    metadata.first = subset.first;
    metadata.last = subset.last;
    metadata.prev = subset.prev;
    metadata.nSplats = subset.nSplats;
    metadata.nRanges = subset.nRanges;
    MPI_Send(&metadata, 1, subsetMetadataType, dest, MLSGPU_TAG_WORK, comm);
    MPI_Send(const_cast<std::tr1::uint32_t *>(&subset.splatRanges[0]),
             subset.splatRanges.size(), mpi_type_traits<std::tr1::uint32_t>::type(),
             dest, MLSGPU_TAG_WORK, comm);
}

void Access::recv(SplatSet::SubsetBase &subset, MPI_Comm comm, int source)
{
    SubsetMetadata metadata;
    MPI_Recv(&metadata, 1, subsetMetadataType, source, MLSGPU_TAG_WORK, comm, MPI_STATUS_IGNORE);
    subset.splatRanges.resize(metadata.size);
    subset.first = metadata.first;
    subset.last = metadata.last;
    subset.prev = metadata.prev;
    subset.nSplats = metadata.nSplats;
    subset.nRanges = metadata.nRanges;
    MPI_Recv(&subset.splatRanges[0], metadata.size, mpi_type_traits<std::tr1::uint32_t>::type(),
             source, MLSGPU_TAG_WORK, comm, MPI_STATUS_IGNORE);
}

void send(const BucketCollector::Bin &bin, MPI_Comm comm, int dest)
{
    send(bin.ranges, comm, dest);
    send(bin.chunkId, comm, dest);
    send(bin.grid, comm, dest);
}

void recv(BucketCollector::Bin &bin, MPI_Comm comm, int source)
{
    recv(bin.ranges, comm, source);
    recv(bin.chunkId, comm, source);
    recv(bin.grid, comm, source);
}

void send(const MesherWork &work, MPI_Comm comm, int dest)
{
    std::size_t sizes[3] =
    {
        work.mesh.numVertices(),
        work.mesh.numTriangles(),
        work.mesh.numInternalVertices()
    };

    send(work.chunkId, comm, dest);
    MPI_Send(&sizes, 3, mpi_type_traits<std::size_t>::type(), dest, MLSGPU_TAG_WORK, comm);

    if (work.hasEvents)
        work.trianglesEvent.wait();
    MPI_Send(const_cast<cl_uint *>(&work.mesh.triangles[0][0]), 3 * work.mesh.numTriangles(),
             mpi_type_traits<cl_uint>::type(), dest, MLSGPU_TAG_WORK, comm);

    if (work.hasEvents)
        work.vertexKeysEvent.wait();
    MPI_Send(const_cast<cl_ulong *>(&work.mesh.vertexKeys[0]), work.mesh.numExternalVertices(),
             mpi_type_traits<cl_ulong>::type(), dest, MLSGPU_TAG_WORK, comm);

    if (work.hasEvents)
        work.verticesEvent.wait();
    MPI_Send(const_cast<cl_float *>(&work.mesh.vertices[0][0]), 3 * work.mesh.numVertices(),
             mpi_type_traits<cl_float>::type(), dest, MLSGPU_TAG_WORK, comm);
}

void recv(MesherWork &work, void *ptr, MPI_Comm comm, int source)
{
    work.hasEvents = false;
    // Make sure any old references get dropped
    work.verticesEvent = cl::Event();
    work.trianglesEvent = cl::Event();
    work.vertexKeysEvent = cl::Event();

    recv(work.chunkId, comm, source);
    std::size_t sizes[3];
    MPI_Recv(&sizes, 3, mpi_type_traits<std::size_t>::type(),
             source, MLSGPU_TAG_WORK, comm, MPI_STATUS_IGNORE);

    work.mesh = HostKeyMesh(ptr, MeshSizes(sizes[0], sizes[1], sizes[2]));
    MPI_Recv(&work.mesh.triangles[0][0], 3 * work.mesh.numTriangles(),
             mpi_type_traits<cl_uint>::type(), source, MLSGPU_TAG_WORK, comm, MPI_STATUS_IGNORE);
    MPI_Recv(&work.mesh.vertexKeys[0], work.mesh.numExternalVertices(),
             mpi_type_traits<cl_ulong>::type(), source, MLSGPU_TAG_WORK, comm, MPI_STATUS_IGNORE);
    MPI_Recv(&work.mesh.vertices[0][0], 3 * work.mesh.numVertices(),
             mpi_type_traits<cl_float>::type(), source, MLSGPU_TAG_WORK, comm, MPI_STATUS_IGNORE);
}

void init()
{
    registerGridType();
    registerSubsetMetadataType();
    registerChunkIdType();
    registerSplatType();
}

} // namespace Serialize
