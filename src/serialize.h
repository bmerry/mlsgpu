/*
 * mlsgpu: surface reconstruction from point clouds
 * Copyright (C) 2013  University of Cape Town
 *
 * This file is part of mlsgpu.
 *
 * mlsgpu is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

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
#include "bucket_collector.h"

/* Forward declaration */
class Grid;
struct ChunkId;
struct MesherWork;
struct Splat;
namespace Bucket { struct Recursion; }
namespace SplatSet { class SubsetBase; }

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

template<>
class mpi_type_traits<char>
{
public:
    static MPI_Datatype type() { return MPI_CHAR; }
};

template<>
class mpi_type_traits<wchar_t>
{
public:
    static MPI_Datatype type() { return MPI_WCHAR; }
};

void send(const Grid &grid, MPI_Comm comm, int dest);
void recv(Grid &grid, MPI_Comm comm, int source);

void send(const ChunkIdPod &chunkId, MPI_Comm comm, int dest);
void recv(ChunkIdPod &chunkId, MPI_Comm comm, int source);

void send(const Splat *splats, std::size_t numSplats, MPI_Comm comm, int dest);
void recv(Splat *splats, std::size_t numSplats, MPI_Comm comm, int source);

void send(const SplatSet::SubsetBase &subset, MPI_Comm comm, int dest);
void recv(SplatSet::SubsetBase &subset, MPI_Comm comm, int source);

void send(const BucketCollector::Bin &bin, MPI_Comm comm, int dest);
void recv(BucketCollector::Bin &bin, MPI_Comm comm, int source);

void send(const MesherWork &work, MPI_Comm comm, int dest);
/**
 * Receive @ref MesherWork. The number of bytes required must have already
 * been communicated and used to allocate a suitable large buffer to hold
 * the mesh data.
 */
void recv(MesherWork &work, void *ptr, MPI_Comm comm, int source);

/**
 * Broadcast a string to all ranks (like @c MPI_Bcast).
 */
void broadcast(std::string &str, MPI_Comm comm, int root);

/**
 * Broadcast a path to all ranks (like @c MPI_Bcast).
 */
void broadcast(boost::filesystem::path &path, MPI_Comm comm, int root);

/**
 * Registers MPI data types. This must be called before any of the send or
 * receive functions.
 */
void init();

} // namespace Serialize

#endif /* !SERIALIZE_H */
