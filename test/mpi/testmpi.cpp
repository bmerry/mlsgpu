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
 * Main program for running MPI unit tests.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <iostream>
#include <mpi.h>
#include "../testutil.h"
#include "../../src/serialize.h"

int main(int argc, char **argv)
{
    int rank, size;
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    if (provided < MPI_THREAD_SERIALIZED)
    {
        std::cerr << "MPI implementation does not provide the required level of thread support\n";
        MPI_Finalize();
        return 1;
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size <= 1)
    {
        std::cerr << "Must use at least two processes\n";
        MPI_Finalize();
        return 1;
    }

    Serialize::init();
    bool isMaster = (rank == 0);
    int ret = runTests(argc, (const char **) argv, isMaster);

    MPI_Finalize();
    return ret;
}
