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

template<typename T>
class mpi_type_traits
{
};

template<>
class mpi_type_traits<int>
{
public:
    static MPI_Datatype type() { return MPI_INT; }
};

template<>
class mpi_type_traits<unsigned int>
{
public:
    static MPI_Datatype type() { return MPI_UNSIGNED; }
};

template<>
class mpi_type_traits<long>
{
public:
    static MPI_Datatype type() { return MPI_LONG; }
};

template<>
class mpi_type_traits<unsigned long>
{
public:
    static MPI_Datatype type() { return MPI_UNSIGNED_LONG; }
};

template<>
class mpi_type_traits<long long>
{
public:
    static MPI_Datatype type() { return MPI_LONG_LONG; }
};

template<>
class mpi_type_traits<unsigned long long>
{
public:
    static MPI_Datatype type() { return MPI_UNSIGNED_LONG_LONG; }
};

template<>
class mpi_type_traits<short>
{
public:
    static MPI_Datatype type() { return MPI_SHORT; }
};

template<>
class mpi_type_traits<unsigned short>
{
public:
    static MPI_Datatype type() { return MPI_UNSIGNED_SHORT; }
};

template<>
class mpi_type_traits<float>
{
public:
    static MPI_Datatype type() { return MPI_FLOAT; }
};

template<>
class mpi_type_traits<double>
{
public:
    static MPI_Datatype type() { return MPI_DOUBLE; }
};

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
    MPI_Datatype types[5] = { MPI_LB, MPI_FLOAT, MPI_FLOAT, mpi_type_traits<Grid::difference_type>::type(), MPI_UB };

    MPI_Type_create_struct(5, lengths, displacements, types, &gridType);
    MPI_Type_set_name(gridType, const_cast<char *>("RawGrid"));
    MPI_Type_commit(&gridType);
}


static MPI_Datatype bucketRecursionType; ///< MPI datatype representing @ref Bucket::Recursion

/// Create @ref bucketRecursionType
static void registerBucketRecursionType()
{
    int lengths[5] = {1, 1, 1, 3, 1};
    MPI_Aint displacements[5] =
    {
        0,
        offsetof(Bucket::Recursion, depth),
        offsetof(Bucket::Recursion, totalRanges),
        offsetof(Bucket::Recursion, chunk),
        sizeof(Bucket::Recursion)
    };
    MPI_Datatype types[5] =
    {
        MPI_LB,
        MPI_UNSIGNED,
        mpi_type_traits<std::size_t>::type(),
        mpi_type_traits<Grid::size_type>::type(),
        MPI_UB
    };

    MPI_Type_create_struct(5, lengths, displacements, types, &bucketRecursionType);
    MPI_Type_set_name(bucketRecursionType, const_cast<char *>("BucketRecursion"));
    MPI_Type_commit(&bucketRecursionType);
}

static MPI_Datatype chunkIdType; ///< MPI datatype representing @ref ChunkId

static void registerChunkIdType()
{
    int lengths[4] = {1, 1, 3, 1};
    MPI_Aint displacements[4] = {
        0,
        offsetof(ChunkId, gen),
        offsetof(ChunkId, coords),
        sizeof(ChunkId)
    };
    MPI_Datatype types[4] =
    {
        MPI_LB,
        mpi_type_traits<ChunkId::gen_type>::type(),
        mpi_type_traits<Grid::size_type>::type(),
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

void send(const Bucket::Recursion &recursion, MPI_Comm comm, int dest)
{
    MPI_Send(const_cast<Bucket::Recursion *>(&recursion), 1, bucketRecursionType, dest, MLSGPU_TAG_WORK, comm);
}

void recv(Bucket::Recursion &recursion, MPI_Comm comm, int source)
{
    MPI_Recv(&recursion, 1, bucketRecursionType, source, MLSGPU_TAG_WORK, comm, MPI_STATUS_IGNORE);
}

void send(const ChunkId &chunkId, MPI_Comm comm, int dest)
{
    MPI_Send(const_cast<ChunkId *>(&chunkId), 1, chunkIdType, dest, MLSGPU_TAG_WORK, comm);
}

void recv(ChunkId &chunkId, MPI_Comm comm, int source)
{
    MPI_Recv(&chunkId, 1, chunkIdType, source, MLSGPU_TAG_WORK, comm, MPI_STATUS_IGNORE);
}

void send(const Splat *splats, std::size_t numSplats, MPI_Comm comm, int dest)
{
    MPI_Send(const_cast<Splat *>(splats), numSplats, splatType, dest, MLSGPU_TAG_WORK, comm);
}

void recv(Splat *splats, std::size_t numSplats, MPI_Comm comm, int source)
{
    MPI_Recv(splats, numSplats, splatType, source, MLSGPU_TAG_WORK, comm, MPI_STATUS_IGNORE);
}

void send(const MesherWork &work, MPI_Comm comm, int dest)
{
    std::size_t sizes[3] =
    {
        work.mesh.vertices.size(),
        work.mesh.triangles.size(),
        work.mesh.vertexKeys.size()
    };

    send(work.chunkId, comm, dest);
    MPI_Send(&sizes, 3, mpi_type_traits<std::size_t>::type(), dest, MLSGPU_TAG_WORK, comm);

    if (work.hasEvents)
        work.trianglesEvent.wait();
    MPI_Send(const_cast<cl_uint *>(&work.mesh.triangles[0][0]), 3 * work.mesh.triangles.size(),
             mpi_type_traits<cl_uint>::type(), dest, MLSGPU_TAG_WORK, comm);

    if (work.hasEvents)
        work.vertexKeysEvent.wait();
    MPI_Send(const_cast<cl_ulong *>(&work.mesh.vertexKeys[0]), work.mesh.vertexKeys.size(),
             mpi_type_traits<cl_ulong>::type(), dest, MLSGPU_TAG_WORK, comm);

    if (work.hasEvents)
        work.verticesEvent.wait();
    MPI_Send(const_cast<cl_float *>(&work.mesh.vertices[0][0]), 3 * work.mesh.vertices.size(),
             mpi_type_traits<cl_float>::type(), dest, MLSGPU_TAG_WORK, comm);
}

void recv(MesherWork &work, MPI_Comm comm, int source)
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
    work.mesh.vertices.resize(sizes[0]);
    work.mesh.triangles.resize(sizes[1]);
    work.mesh.vertexKeys.resize(sizes[2]);
    MPI_Recv(&work.mesh.triangles[0][0], 3 * work.mesh.triangles.size(),
             mpi_type_traits<cl_uint>::type(), source, MLSGPU_TAG_WORK, comm, MPI_STATUS_IGNORE);
    MPI_Recv(&work.mesh.vertexKeys[0], work.mesh.vertexKeys.size(),
             mpi_type_traits<cl_ulong>::type(), source, MLSGPU_TAG_WORK, comm, MPI_STATUS_IGNORE);
    MPI_Recv(&work.mesh.vertices[0][0], 3 * work.mesh.vertices.size(),
             mpi_type_traits<cl_float>::type(), source, MLSGPU_TAG_WORK, comm, MPI_STATUS_IGNORE);
}

void init()
{
    registerGridType();
    registerBucketRecursionType();
    registerChunkIdType();
    registerSplatType();
}

} // namespace Serialize
