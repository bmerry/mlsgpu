/**
 * @file
 *
 * Tests for @ref WorkerGroup and @ref WorkerGroupMulti.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/extensions/HelperMacros.h>
#include <boost/thread.hpp>
#include <boost/smart_ptr/shared_ptr.hpp>
#include <boost/smart_ptr/make_shared.hpp>
#include <algorithm>
#include <vector>
#include "testutil.h"
#include "../src/worker_group.h"
#include "../src/timeplot.h"

namespace
{

/**
 * Output where workitems are sent.
 */
struct Sink
{
    boost::mutex mutex;
    std::vector<int> values;
};

/**
 * Item used by @ref Group and @ref GroupMulti.
 */
struct Item
{
    int value;     ///< Value to process
    int subset;    ///< Subset for this item (used by @ref GroupMulti)

    int getKey() { return subset; }
    explicit Item(int subset = 0) : value(-1), subset(subset) {}
};

/**
 * Worker used by @ref Group and @ref GroupMulti.
 */
class Worker : public WorkerBase
{
private:
    Sink &sink;
    int subset;    ///< Expected subset of the items passed in
    bool running;

public:
    void operator()(Item &item);
    void start() { CPPUNIT_ASSERT(!running); running = true; }
    void stop() { CPPUNIT_ASSERT(running); running = false; }
    int getKey() const { return subset; }
    explicit Worker(Sink &sink, int idx, int subset = 0)
        : WorkerBase("test", idx), sink(sink), subset(subset), running(false) {}
};

/**
 * Implementation of @ref WorkerGroup that accepts numbers, doubles them and
 * appends them (thread-safely) to a vector.
 */
class Group : public WorkerGroup<Item, Worker, Group>
{
public:
    Group(Sink &sink, std::size_t workers, std::size_t spare);
};

/**
 * Implementation of @ref WorkerGroupMulti. It behaves essentially the same
 * as @ref WorkerGroup, but artificially creates subsets.
 */
class GroupMulti : public WorkerGroupMulti<Item, Worker, GroupMulti, int>
{
public:
    GroupMulti(Sink &sink, std::size_t numSets, std::size_t workers, std::size_t spare);
};

/**
 * Thread class that generates an arithmetic progression.
 */
template<typename T>
class Producer
{
private:
    int first;
    int last;
    int step;
    T &outGroup;
    boost::shared_ptr<Timeplot::Worker> tworker;
public:
    void operator()();

    Producer(int first, int last, int step, T &outGroup, int idx)
        : first(first), last(last), step(step), outGroup(outGroup),
        tworker(boost::make_shared<Timeplot::Worker>("test.producer", idx))
    {
    }
};


void Worker::operator()(Item &item)
{
    CPPUNIT_ASSERT(running);
    int out = item.value * 2;
    CPPUNIT_ASSERT_EQUAL(subset, item.subset);
    boost::lock_guard<boost::mutex> lock(sink.mutex);
    sink.values.push_back(out);
}

Group::Group(Sink &sink, std::size_t workers, std::size_t spare)
    : WorkerGroup<Item, Worker, Group>("test", workers, spare)
{
    for (std::size_t i = 0; i < workers + spare; i++)
        addPoolItem(boost::make_shared<Item>());
    for (std::size_t i = 0; i < workers; i++)
        addWorker(new Worker(sink, i));
}

GroupMulti::GroupMulti(Sink &sink, std::size_t numSets, std::size_t workers, std::size_t spare)
    : WorkerGroupMulti<Item, Worker, GroupMulti, int>("test", numSets, workers, spare)
{
    for (std::size_t s = 0; s < numSets; s++)
    {
        for (std::size_t i = 0; i < workers + spare; i++)
            addPoolItem(boost::make_shared<Item>(s));
        for (std::size_t i = 0; i < workers; i++)
            addWorker(new Worker(sink, i, s));
    }
}

template<typename T>
void Producer<T>::operator()()
{
    for (int i = first; i < last; i += step)
    {
        boost::shared_ptr<Item> item = outGroup.get(*tworker);
        item->value = i;
        outGroup.push(item, *tworker);
    }
}

} // anonymous namespace

/// Tests for @ref WorkerGroup
class TestWorkerGroup : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestWorkerGroup);
    CPPUNIT_TEST(testStress);
    CPPUNIT_TEST_SUITE_END();

private:
    void testStress();
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestWorkerGroup, TestSet::perCommit());

void TestWorkerGroup::testStress()
{
    const int numProducers = 4;
    const int numWorkers = 4;
    const int numbers = 10000;
    Sink sink;
    Group group(sink, numWorkers, numProducers);
    group.start();
    boost::thread_group producers;
    for (int i = 0; i < numProducers; i++)
        producers.add_thread(new boost::thread(Producer<Group>(i, numbers, numProducers, group, i)));
    producers.join_all();
    group.stop();

    CPPUNIT_ASSERT_EQUAL(numbers, int(sink.values.size()));
    std::sort(sink.values.begin(), sink.values.end());
    for (int i = 0; i < numbers; i++)
    {
        CPPUNIT_ASSERT_EQUAL(2 * i, sink.values[i]);
    }
}

/// Tests for @ref WorkerGroupMulti
class TestWorkerGroupMulti : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestWorkerGroupMulti);
    CPPUNIT_TEST(testStress);
    CPPUNIT_TEST_SUITE_END();

private:
    void testStress();
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestWorkerGroupMulti, TestSet::perCommit());

void TestWorkerGroupMulti::testStress()
{
    const int numProducers = 6;
    const int numWorkers = 2;
    const int numSets = 3;
    const int numbers = 10000;
    Sink sink;
    GroupMulti group(sink, numSets, numWorkers, numProducers / numSets);
    group.start();
    boost::thread_group producers;
    for (int i = 0; i < numProducers; i++)
        producers.add_thread(new boost::thread(Producer<GroupMulti>(i, numbers, numProducers, group, i)));
    producers.join_all();
    group.stop();

    CPPUNIT_ASSERT_EQUAL(numbers, int(sink.values.size()));
    std::sort(sink.values.begin(), sink.values.end());
    for (int i = 0; i < numbers; i++)
    {
        CPPUNIT_ASSERT_EQUAL(2 * i, sink.values[i]);
    }
}
