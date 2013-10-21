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
    : parent(parent), comm(comm), root(root), total(total), thresh(total / 1000), unsent(0)
{
}

void ProgressMPI::operator+=(size_type inc)
{
    boost::lock_guard<boost::mutex> lock(mutex);
    unsent += inc;
    if (unsent > thresh)
        syncUnlocked();
}

void ProgressMPI::syncUnlocked()
{
    if (unsent != 0)
    {
        long long buf = unsent;
        MPI_Bsend(&buf, 1, MPI_LONG_LONG, root, MLSGPU_TAG_PROGRESS, comm);
        unsent = 0;
    }
}

void ProgressMPI::sync()
{
    boost::lock_guard<boost::mutex> lock(mutex);
    syncUnlocked();
}

void ProgressMPI::operator()() const
{
    size_type current = 0;
    const boost::posix_time::time_duration sleepTime = boost::posix_time::milliseconds(500);
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
