/**
 * @file
 *
 * A distributed progress meter.
 */

#ifndef PROGRESS_MPI_H
#define PROGRESS_MPI_H

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <mpi.h>
#include <boost/thread/locks.hpp>
#include "progress.h"

/**
 * A distributed MPI progress meter. The root process will forward the progress
 * updates to another instance of ProgressMeter e.g. to display the results.
 * All processes (including the master) may update the progress meter using
 * the @c += and @c ++ operators, and this information is sent over MPI to the
 * root. To save bandwidth, updates are sent only when @ref sync is called or
 * when the unsent updates amount to at least 0.1%.
 *
 * To reduce CPU load on the root when using a busy-wait implementation of MPI
 * (e.g. OpenMPI), the communicator is polled on an interval.
 *
 * The root process must also call @ref operator() to receive the updates. This
 * will typically be done in a separate thread.
 *
 * The constructor and destructor are local operations. In fact, there is no
 * requirement for a 1-to-1 mapping of instances to processes.
 */
class ProgressMPI : public ProgressMeter, public boost::noncopyable
{
public:
    /**
     * Constructor. It is legal for @a total to be different to the capacity of
     * the parent meter. The @a total is the total amount that this meter will
     * forward to the parent, but it is legal to have other sources of progress
     * within the parent.
     *
     * @param parent     Progress meter which receives all the updates. Only relevant on root.
     * @param total      Total amount of progress for this progress meter.
     * @param comm       Communicator for the group. It is @em not cloned, so
     *                   progress updates can potentially cause problems on the communicator
     *                   if it gets mixed up with other traffic. It does have its own tag,
     *                   however, so it is possible to keep things separate as long as all
     *                   receives specify a tag.
     * @param root       Root process in @a comm that will receive and forward updates.
     */
    ProgressMPI(ProgressMeter *parent, size_type total, MPI_Comm comm, int root);

    virtual void operator+=(size_type inc);

    /**
     * Send all unsent updates to the root.
     */
    void sync();

    /**
     * Run the receive code. This will return when the capacity has been reached.
     */
    void operator()() const;

private:
    /// Like @ref sync but the caller locks
    void syncUnlocked();

    ProgressMeter * const parent; ///< Parent progress meter (well-defined only on root)
    MPI_Comm comm;
    const int root;

    const size_type total;        ///< Expected total progress
    const size_type thresh;       ///< Minimum progress before sending updates

    size_type unsent;             ///< Unsent increment amount (on slaves)
    boost::mutex mutex;           ///< Mutex protecting @ref unsent
};

#endif /* !PROGRESS_MPI_H */
