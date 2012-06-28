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
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/ptr_container/ptr_map.hpp>
#include <boost/smart_ptr/shared_ptr.hpp>
#include <cstdlib>
#include <cstddef>
#include <stdexcept>
#include <iostream>
#include "work_queue.h"
#include "statistics.h"
#include "errors.h"

/**
 * Base class for @ref WorkerGroup that handles only the threads, workers and pool,
 * but not a work queue. See @ref WorkerGroup for details.
 */
template<typename WorkItem, typename GenType, typename Worker, typename Derived>
class WorkerGroupPool : public boost::noncopyable
{
public:
    typedef GenType gen_type;

    bool running() const
    {
        return !threads.empty();
    }

    /**
     * Retrieve an unused item from the pool to be populated with work. This
     * may block if all the items are currently in use.
     *
     * @warning The returned item @em must be enqueued with @ref push, even if
     * it turns out there is nothing to do. Failure to do so will lead to the
     * item being lost to the pool, and possibly deadlock.
     */
    boost::shared_ptr<WorkItem> get()
    {
        Statistics::Timer timer(getStat);
        return itemPool.pop();
    }

    /**
     * Enqueue an item of work.
     *
     * @pre @a item was obtained by @ref get.
     */
    void push(const gen_type &gen, boost::shared_ptr<WorkItem> item)
    {
        Statistics::Timer timer(pushStat);
        static_cast<Derived *>(this)->pushImpl(gen, item);
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
        static_cast<Derived *>(this)->stopImpl();
        for (std::size_t i = 0; i < threads.size(); i++)
            threads[i].join();
        threads.clear();
    }

    /// Returns the number of workers.
    std::size_t numWorkers() const
    {
        return workers.size();
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

    /**
     * Register a work item during construction.
     *
     * @see @ref WorkerGroup::WorkerGroup.
     */
    void addPoolItem(boost::shared_ptr<WorkItem> item)
    {
        itemPool.push(item);
    }

    /// Retrieve a reference to a worker.
    Worker &getWorker(std::size_t index)
    {
        return workers.at(index);
    }

    /**
     * Constructor. The derived class must chain to this, and then
     * make exactly @a numWorkers calls to @ref addWorker and @a numWorkers + @a spare
     * calls to @ref addPoolItem to provide the constructed workers and
     * work items.
     *
     * @param numWorkers     Number of worker threads to use.
     * @param spare          Number of work items to have available in the pool when all workers are busy.
     * @param pushStat       Statistic for time blocked in @ref push.
     * @param firstPopStat   Statistic for time blocked in the first call to @ref WorkQueue::pop.
     * @param popStat        Statistic for time blocked in @ref WorkQueue::pop after the first time.
     * @param getStat        Statistic for time blocked in @ref get.
     *
     * @pre @a numWorkers &gt; 0.
     */
    WorkerGroupPool(std::size_t numWorkers, std::size_t spare,
                    Statistics::Variable &pushStat,
                    Statistics::Variable &firstPopStat,
                    Statistics::Variable &popStat,
                    Statistics::Variable &getStat)
        : itemPool(numWorkers + spare),
        pushStat(pushStat), firstPopStat(firstPopStat), popStat(popStat), getStat(getStat)
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
            try
            {
                bool firstPop = true;
                while (true)
                {
                    Timer timer;
                    gen_type gen;
                    boost::shared_ptr<WorkItem> item = owner.popImpl(worker, gen);
                    if (!item.get())
                        break; // we have been asked to shut down
                    if (firstPop)
                    {
                        firstPop = false;
                        owner.firstPopStat.add(timer.getElapsed());
                    }
                    else
                        owner.popStat.add(timer.getElapsed());
                    worker(gen, *item);
                    owner.itemPool.push(item);
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

    Pool<WorkItem> itemPool;

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

    Statistics::Variable &pushStat;
    Statistics::Variable &firstPopStat;
    Statistics::Variable &popStat;
    Statistics::Variable &getStat;
};

/**
 * A collection of threads operating on work-items, fed by a queue.
 *
 * @param WorkItem     A POD type describing an item of work.
 * @param GenType      Key type used to represent generations of work.
 * @param Worker       Function object class that is called to process elements.
 * @param Derived      The class that is being derived from the template.
 *
 * The @a Worker class must have an @c operator() that accepts a reference to a
 * @a WorkItem. The operator does not need to be @c const.  The worker class
 * does not need to be copyable or default-constructable and may contain
 * significant state.
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
template<typename WorkItem, typename GenType, typename Worker, typename Derived>
class WorkerGroup : public WorkerGroupPool<WorkItem, GenType, Worker, Derived>
{
    typedef WorkerGroupPool<WorkItem, GenType, Worker, Derived> BaseType;
    friend class WorkerGroupPool<WorkItem, GenType, Worker, Derived>;

public:
    typedef GenType gen_type;

    /**
     * Wraps @ref GenerationalWorkQueue::producerStart.
     *
     * @pre The worker threads are not running.
     */
    void producerStart(const gen_type &gen)
    {
        MLSGPU_ASSERT(!this->running(), state_error);
        workQueue.producerStart(gen);
    }

    /**
     * Wraps @ref GenerationalWorkQueue::producerNext.
     */
    void producerNext(const gen_type &oldGen, const gen_type &newGen)
    {
        workQueue.producerNext(oldGen, newGen);
    }

    /**
     * Wraps @ref GenerationalWorkQueue::producerStop.
     */
    void producerStop(const gen_type &gen)
    {
        workQueue.producerStop(gen);
    }

    /**
     * Constructor. The derived class must chain to this, and then
     * make exactly @a numWorkers calls to @ref addWorker and @a numWorkers + @a spare
     * calls to @ref addPoolItem to provide the constructed workers and
     * work items.
     *
     * @param numWorkers     Number of worker threads to use.
     * @param spare          Number of work items to have available in the pool when all workers are busy.
     * @param pushStat       Statistic for time blocked in @ref push.
     * @param firstPopStat   Statistic for time blocked in first call to @ref WorkQueue::pop.
     * @param popStat        Statistic for time blocked in subsequent calls to @ref WorkQueue::pop.
     * @param getStat        Statistic for time blocked in @ref get.
     *
     * @pre @a numWorkers &gt; 0.
     */
    WorkerGroup(std::size_t numWorkers, std::size_t spare,
                Statistics::Variable &pushStat,
                Statistics::Variable &firstPopStat,
                Statistics::Variable &popStat,
                Statistics::Variable &getStat)
        : BaseType(numWorkers, spare, pushStat, firstPopStat, popStat, getStat),
          workQueue(numWorkers + spare)
    {
    }

private:
    /**
     * Enqueue an item of work.
     *
     * @pre @a item was obtained by @ref get.
     */
    void pushImpl(const gen_type &gen, boost::shared_ptr<WorkItem> item)
    {
        workQueue.push(gen, item);
    }

    /**
     * Dequeue an item of work.
     */
    boost::shared_ptr<WorkItem> popImpl(const Worker &worker, gen_type &gen)
    {
        (void) &worker;
        return workQueue.pop(gen);
    }

    /**
     * Shut down the worker threads.
     */
    void stopImpl()
    {
        for (std::size_t i = 0; i < this->numWorkers(); i++)
            workQueue.pushNoGen(boost::shared_ptr<WorkItem>());
    }

    /// Work queue
    GenerationalWorkQueue<boost::shared_ptr<WorkItem>, GenType> workQueue;
};


/**
 * Variation on @ref WorkerGroup in which the workers and items are partitioned into
 * sets, where each work item can only be used together with the corresponding set of
 * workers. There is a single item pool but a separate queue for each set. Items are
 * dispatched to their matching queue.
 *
 * This is aimed to supporting non-uniform or non-shared memory systems where each
 * worker is tied to a device and each item has memory allocated in that
 * device's memory space.
 *
 * The sets are each identified by a key, which must be suitable for use with
 * an associative container.
 */
template<typename WorkItem, typename GenType, typename Worker, typename Derived, typename Key, typename Compare = std::less<Key> >
class WorkerGroupMulti : public WorkerGroupPool<WorkItem, GenType, Worker, Derived>
{
    typedef WorkerGroupPool<WorkItem, GenType, Worker, Derived> BaseType;
    friend class WorkerGroupPool<WorkItem, GenType, Worker, Derived>;

public:
    typedef GenType gen_type;
    typedef Key key_type;

    /**
     * Wraps @ref GenerationalWorkQueue::producerStart.
     *
     * @pre The worker threads are not running.
     */
    void producerStart(const gen_type &gen)
    {
        MLSGPU_ASSERT(!this->running(), state_error);
        BOOST_FOREACH(typename work_queues_type::reference i, workQueues)
        {
            i.second->producerStart(gen);
        }
    }

    /**
     * Wraps @ref GenerationalWorkQueue::producerNext.
     */
    void producerNext(const gen_type &oldGen, const gen_type &newGen)
    {
        BOOST_FOREACH(typename work_queues_type::reference i, workQueues)
        {
            i.second->producerNext(oldGen, newGen);
        }
    }

    /**
     * Wraps @ref GenerationalWorkQueue::producerStop.
     */
    void producerStop(const gen_type &gen)
    {
        BOOST_FOREACH(typename work_queues_type::reference i, workQueues)
        {
            i.second->producerStop(gen);
        }
    }

    /**
     * Constructor. The derived class must chain to this, and then
     * make exactly @a numWorkers * @a numSets calls to @ref addWorker
     * and (@a numWorkers + @a spare) * @a numSets calls to @ref addPoolItem to
     * provide the constructed workers and work items (with the appropriate number
     * per set).
     *
     * @param numSets        Number of sets.
     * @param numWorkers     Number of worker threads to use <em>per set</em>.
     * @param spare          Number of work items to have available in the pool when all workers are busy, <em>per set</em>.
     * @param pushStat       Statistic for time blocked in @ref push.
     * @param firstPopStat   Statistic for time blocked in first call to @ref WorkQueue::pop.
     * @param popStat        Statistic for time blocked in subsequent calls to @ref WorkQueue::pop.
     * @param getStat        Statistic for time blocked in @ref get.
     *
     * @pre @a numSets &gt; 0.
     * @pre @a numWorkers &gt; 0.
     */
    WorkerGroupMulti(
        std::size_t numSets, std::size_t numWorkers, std::size_t spare,
        Statistics::Variable &pushStat,
        Statistics::Variable &firstPopStat,
        Statistics::Variable &popStat,
        Statistics::Variable &getStat)
        : BaseType(numWorkers * numSets, spare * numSets, pushStat, firstPopStat, popStat, getStat),
        workQueueCapacity(numWorkers + spare)
    {
        MLSGPU_ASSERT(numSets > 0, std::invalid_argument);
    }

    void addWorker(Worker *worker)
    {
        BaseType::addWorker(worker);
        Key key = worker->getKey();
        if (!workQueues.count(key))
            workQueues.insert(key, new GenerationalWorkQueue<boost::shared_ptr<WorkItem>, GenType>(workQueueCapacity));
    }

private:
    /**
     * Enqueue an item of work.
     *
     * @pre @a item was obtained by @ref get.
     */
    void pushImpl(const gen_type &gen, boost::shared_ptr<WorkItem> item)
    {
        workQueues.at(item->getKey()).push(gen, item);
    }

    /**
     * Dequeue an item of work.
     */
    boost::shared_ptr<WorkItem> popImpl(const Worker &worker, gen_type &gen)
    {
        return workQueues.at(worker.getKey()).pop(gen);
    }

    /**
     * Shut down the worker threads.
     */
    void stopImpl()
    {
        const std::size_t workersPerSet = this->numWorkers() / workQueues.size();
        BOOST_FOREACH(typename work_queues_type::reference q, workQueues)
            for (std::size_t i = 0; i < workersPerSet; i++)
                q.second->pushNoGen(boost::shared_ptr<WorkItem>());
    }

    typedef boost::ptr_map<Key, GenerationalWorkQueue<boost::shared_ptr<WorkItem>, GenType>, Compare> work_queues_type;
    /// Work queues
    work_queues_type workQueues;

    /// Capacity that will be used per work queue
    std::size_t workQueueCapacity;
};

#endif /* !WORKER_GROUP_H */
