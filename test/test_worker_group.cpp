/**
 * @file
 *
 * Tests for @ref WorkerGroup.
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
 * Item used by @ref Group.
 */
struct Item
{
    int value;     ///< Value to process

    Item() : value(-1) {}
};

/**
 * Worker used by @ref Group.
 */
class Worker : public WorkerBase
{
private:
    Sink &sink;
    bool running;

public:
    void operator()(Item &item);
    void start() { CPPUNIT_ASSERT(!running); running = true; }
    void stop() { CPPUNIT_ASSERT(running); running = false; }
    explicit Worker(Sink &sink, int idx)
        : WorkerBase("test", idx), sink(sink), running(false) {}
};

/**
 * Implementation of @ref WorkerGroup that accepts numbers, doubles them and
 * appends them (thread-safely) to a vector.
 */
class Group : public WorkerGroup<Item, Worker, Group>
{
public:
    Group(Sink &sink, std::size_t workers);
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
    boost::lock_guard<boost::mutex> lock(sink.mutex);
    sink.values.push_back(out);
}

Group::Group(Sink &sink, std::size_t workers)
    : WorkerGroup<Item, Worker, Group>("test", workers)
{
    for (std::size_t i = 0; i < workers; i++)
        addWorker(new Worker(sink, i));
}

template<typename T>
void Producer<T>::operator()()
{
    for (int i = first; i < last; i += step)
    {
        boost::shared_ptr<Item> item = outGroup.get(*tworker, 1);
        item->value = i;
        outGroup.push(item);
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
    Group group(sink, numWorkers);
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
