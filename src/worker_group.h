/**
 * @file
 *
 * Thread pool classes.
 */

#ifndef WORKER_GROUP_H
#define WORKER_GROUP_H

#if HAVE_CONFIG_H
# include <config.h>
#endif

#include <boost/noncopyable.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/locks.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/ptr_container/ptr_map.hpp>
#include <boost/smart_ptr/shared_ptr.hpp>
#include <boost/smart_ptr/make_shared.hpp>
#include <boost/foreach.hpp>
#include <cstdlib>
#include <cstddef>
#include <stdexcept>
#include <iostream>
#include "work_queue.h"
#include "statistics.h"
#include "errors.h"
#include "thread_name.h"
#include "timeplot.h"

/**
 * Base class from which workers may derive. They are not required to do so,
 * but if not they must provide the same interface.
 */
class WorkerBase : public boost::noncopyable
{
private:
    Timeplot::Worker tworker;
public:
    /**
     * Constructor. The @ref Timeplot::Worker is given a name composed of @a
     * name and @a idx. It is thus sane to pass the same name for all workers
     * in a group.
     *
     * @param name     Name for the worker.
     * @param idx      Number of the worker within the pool (zero-based).
     */
    WorkerBase(const std::string &name, int idx)
        : tworker(name, idx) {}

    /**
     * Called when the group starts. Reimplement if special action is needed.
     * Note that a group can be started and stopped multiple times, so this is
     * not equivalent to a constructor.
     */
    void start() {}

    /**
     * Called when the group stops. Reimplement if special action is needed.
     * Note that a group can be started and stopped multiple times, so this is
     * not equivalent to a destructor.
     */
    void stop() {}

    /**
     * Retrieve the @ref Timeplot::Worker to use for recording actions associated
     * with this worker.
     */
    Timeplot::Worker &getTimeplotWorker() { return tworker; }
};

/**
 * A collection of threads operating on work-items, fed by a queue.
 *
 * @param WorkItem     A POD type describing an item of work.
 * @param Worker       Function object class that is called to process elements.
 * @param Derived      The class that is being derived from the template.
 *
 * The @a Worker class must have an @c operator() that accepts a reference to a
 * @a WorkItem. The operator does not need to be @c const.  The worker class
 * does not need to be copyable or default-constructable and may contain
 * significant state.
 *
 * The @a Worker class must also have @c start() and @c stop() methods, which will
 * be called when the whole workgroup is started or stopped. These may be empty,
 * but are provided to allow for additional setup for cleanup, particularly when
 * the whole group is reused in multiple passes.
 *
 * The workitems may also be large objects, and the design is based on
 * recycling a fixed pool rather than having the caller construct them. Users
 * of the class will call @ref get to retrieve an item from the pool, populate
 * it, then call @ref push to enqueue the item for processing.
 *
 * At construction, the worker group is given both a number of threads and a
 * capacity. The capacity determines the number of workitems that will be
 * live. If capacity equals the number of workers, then it will only be
 * possible to populate a new work item while one of the workers is idle. If
 * capacity exceeds the number of threads, then it will be possible to
 * populate spare work items while all worker threads are busy. The capacity
 * is specified as a delta to the number of workers.
 *
 * The @ref start and @ref stop functions are not thread-safe: they should
 * only be called by a single manager thread. The other functions are
 * thread-safe, allowing for multiple producers.
 */
template<typename WorkItem, typename Worker, typename Derived>
class WorkerGroup
{
public:
    bool running() const
    {
        return !threads.empty();
    }

    /**
     * Retrieve an unused item to be populated with work.
     *
     * @todo Eliminate this entirely: callers can just create the item themselves, or
     * the subclass can create it for them.
     */
    boost::shared_ptr<WorkItem> get(Timeplot::Worker &tworker, std::size_t size)
    {
        return boost::make_shared<WorkItem>();
    }

    /**
     * Enqueue an item of work.
     *
     * @pre @a item was obtained by @ref get.
     *
     * @todo Eliminate tworker and size - it can no longer block.
     */
    void push(boost::shared_ptr<WorkItem> item, Timeplot::Worker &tworker, std::size_t size)
    {
        workQueue.push(item);
    }

    /**
     * Start the worker threads running. It is not required to do this
     * before calling @ref get or @ref push, but they may block until another
     * thread calls @ref start.
     *
     * @pre The worker threads are not already running.
     */
    void start()
    {
        MLSGPU_ASSERT(!running(), state_error);
        workQueue.start();
        threads.reserve(workers.size());
        for (std::size_t i = 0; i < workers.size(); i++)
            workers[i].start();
        for (std::size_t i = 0; i < workers.size(); i++)
            threads.push_back(new boost::thread(Thread(*static_cast<Derived *>(this), getWorker(i))));
    }

    /**
     * Shut down the worker threads.
     *
     * @warning This method is not thread-safe relative to other calls such
     * as @ref push. All producers must be shut down while this method is
     * called.
     *
     * @pre The worker threads are currently running.
     */
    void stop()
    {
        MLSGPU_ASSERT(threads.size() == workers.size(), state_error);

        workQueue.stop();
        static_cast<Derived *>(this)->stopPreJoin();
        for (std::size_t i = 0; i < threads.size(); i++)
            threads[i].join();
        threads.clear();
        static_cast<Derived *>(this)->stopPostJoin();
    }

    /// Returns the number of workers.
    std::size_t numWorkers() const
    {
        return workers.size();
    }

    /// Obtain the statistic for reporting compute times
    Statistics::Variable &getComputeStat() const
    {
        return computeStat;
    }

protected:
    /**
     * Register a worker during construction.
     *
     * @see @ref WorkerGroup::WorkerGroup.
     */
    void addWorker(Worker *worker)
    {
        workers.push_back(worker);
    }

    /// Retrieve a reference to a worker.
    Worker &getWorker(std::size_t index)
    {
        return workers.at(index);
    }

    /**
     * Constructor. The derived class must chain to this, and then
     * make exactly @a numWorkers calls to @ref addWorker to provide the
     * constructed workers.
     *
     * @param name           Name for the threads in the pool.
     * @param numWorkers     Number of worker threads to use.
     * @param spare          Number of work items to have available in the pool when all workers are busy.
     *
     * @pre @a numWorkers &gt; 0.
     */
    WorkerGroup(const std::string &name,
                std::size_t numWorkers, std::size_t spare)
        : threadName(name),
        workQueue(),
        pushStat(Statistics::getStatistic<Statistics::Variable>(name + ".push")),
        firstPopStat(Statistics::getStatistic<Statistics::Variable>(name + ".pop.first")),
        popStat(Statistics::getStatistic<Statistics::Variable>(name + ".pop")),
        getStat(Statistics::getStatistic<Statistics::Variable>(name + ".get")),
        computeStat(Statistics::getStatistic<Statistics::Variable>(name + ".compute"))
    {
        MLSGPU_ASSERT(numWorkers > 0, std::invalid_argument);
        workers.reserve(numWorkers);
    }

private:
    /// Thread object that processes items from the queue.
    class Thread
    {
        Derived &owner;
        Worker &worker;

    public:
        Thread(Derived &owner, Worker &worker)
            : owner(owner), worker(worker) {}

        void operator()()
        {
            Timeplot::Worker &tworker = worker.getTimeplotWorker();
            try
            {
                thread_set_name(owner.threadName);
                bool firstPop = true;
                while (true)
                {
                    boost::shared_ptr<WorkItem> item;
                    {
                        Timeplot::Action timer("pop", tworker, firstPop ? owner.firstPopStat : owner.popStat);
                        item = owner.workQueue.pop();
                    }
                    if (!item)
                        break; // we have been asked to shut down
                    firstPop = false;

                    worker(*item);

                    owner.freeItem(*item);
                }
                worker.stop();
            }
            catch (std::runtime_error &e)
            {
                std::cerr << "\n" << e.what() << std::endl;
                std::exit(1);
            }
        }
    };

    /// Name to assign to threads
    const std::string threadName;

    /**
     * Threads. This is empty when no threads are running and contains the
     * thread objects when it is running.
     */
    boost::ptr_vector<boost::thread> threads;

    /**
     * Workers. This is populated during construction (by @ref addWorker) and
     * persists until object destruction.
     */
    boost::ptr_vector<Worker> workers;

    /**
     * Queue of items waiting to be processed.
     */
    WorkQueue<boost::shared_ptr<WorkItem> > workQueue;

    Statistics::Variable &pushStat;
    Statistics::Variable &firstPopStat;
    Statistics::Variable &popStat;
    Statistics::Variable &getStat;
    Statistics::Variable &computeStat;

    /**
     * Take shutdown actions prior to joining the worker threads. This is a hook
     * that subclasses may override.
     */
    void stopPreJoin()
    {
    }

    /**
     * Take shutdown actions after the worker threads have been joined. This is
     * a hook that subclasses may override.
     */
    void stopPostJoin()
    {
    }

    /**
     * Release transient resources stored in an item. This is a hook that
     * subclasses my override.
     */
    void freeItem(WorkItem &item)
    {
        (void) item;
    }
};

#endif /* !WORKER_GROUP_H */
