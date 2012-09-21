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

    bool isMaster = (rank == 0);
    int ret = runTests(argc, (const char **) argv, isMaster);

    MPI_Finalize();
    return ret;
}
