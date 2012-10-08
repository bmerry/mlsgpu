/**
 * @file
 *
 * A distributed progress meter.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <mpi.h>
#include <boost/thread/locks.hpp>
#include <boost/thread/thread.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>
#include "tags.h"
#include "progress.h"
#include "progress_mpi.h"

ProgressMPI::ProgressMPI(ProgressMeter *parent, size_type total, MPI_Comm comm, int root)
    : parent(parent), root(root), total(total), unsent(0)
{
    MPI_Comm_dup(comm, &this->comm);
}

void ProgressMPI::operator+=(size_type inc)
{
    boost::lock_guard<boost::mutex> lock(mutex);
    unsent += inc;
}

void ProgressMPI::sync()
{
    boost::lock_guard<boost::mutex> lock(mutex);
    if (unsent != 0)
    {
        long long buf = unsent;
        MPI_Bsend(&buf, 1, MPI_LONG_LONG, root, MLSGPU_TAG_PROGRESS, comm);
        unsent = 0;
    }
}

void ProgressMPI::operator()() const
{
    size_type current = 0;
    const boost::posix_time::time_duration sleepTime = boost::posix_time::milliseconds(100);
    while (current < total)
    {
        long long update;
        MPI_Request request;
        MPI_Irecv(&update, 1, MPI_LONG_LONG, MPI_ANY_SOURCE, MLSGPU_TAG_PROGRESS, comm, &request);
        int flag;
        MPI_Test(&request, &flag, MPI_STATUS_IGNORE);
        while (!flag)
        {
            boost::this_thread::sleep(sleepTime);
            MPI_Test(&request, &flag, MPI_STATUS_IGNORE);
        }
        current += update;
        *parent += update;
    }
}
