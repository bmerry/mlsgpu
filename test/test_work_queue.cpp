/**
 * @file
 *
 * Tests for @ref WorkQueue.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/extensions/HelperMacros.h>
#include <vector>
#include <algorithm>
#include <utility>
#include <boost/ref.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/locks.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/bind.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/smart_ptr/scoped_ptr.hpp>
#include "testmain.h"
#include "../src/work_queue.h"

using namespace std;

namespace
{

/**
 * Function object that wraps the class-specific push function.
 */
template<typename T>
class Push
{
};

template<typename ValueType>
class Push<WorkQueue<ValueType> >
{
public:
    typedef void result_type;

    void operator()(WorkQueue<ValueType> &queue, const ValueType &value) const
    {
        queue.push(value);
    }
};

template<typename ValueType>
class Push<GenerationalWorkQueue<ValueType> >
{
public:
    typedef void result_type;

    void operator()(GenerationalWorkQueue<ValueType> &queue, const ValueType &value) const
    {
        queue.pushNoGen(value);
    }
};

template<typename T>
class Pop
{
};

template<typename ValueType>
class Pop<WorkQueue<ValueType> >
{
public:
    typedef ValueType result_type;
    ValueType operator()(WorkQueue<ValueType> &queue) const
    {
        return queue.pop();
    }
};

template<typename ValueType>
class Pop<GenerationalWorkQueue<ValueType> >
{
public:
    typedef ValueType result_type;
    ValueType operator()(GenerationalWorkQueue<ValueType> &queue) const
    {
        unsigned int gen;
        return queue.pop(gen);
    }
};

} // anonymous namespace

/**
 * Base class used by @ref TestWorkQueue and @ref TestGenerationalWorkQueue.
 * @param T        A work queue class with a value type of @c int.
 */
template<typename T>
class TestWorkQueueBase : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestWorkQueueBase<T>);
    CPPUNIT_TEST(testCapacity);
    CPPUNIT_TEST(testSize);
    CPPUNIT_TEST(testStress);
    CPPUNIT_TEST_SUITE_END_ABSTRACT();
private:
    /**
     * Adds a sequence of consecutive integers to the work queue.
     */
    static void producerThread(T &queue, int start, int count);

    /**
     * Pulls integers from a work queue and appends them to a vector. The
     * vector is locked while adding to it. A negative value in the queue is
     * used to signal shutdown.
     */
    static void consumerThread(T &queue, vector<int> &out, boost::mutex &mutex);

public:
    void testCapacity();         ///< Test the capacity function and that it can hold that much
    void testSize();             ///< Test the size function
    void testStress();           ///< Stress test with multiple consumers and producers
};

/// Tests for @ref WorkQueue
class TestWorkQueue : public TestWorkQueueBase<WorkQueue<int> >
{
    CPPUNIT_TEST_SUB_SUITE(TestWorkQueue, TestWorkQueueBase<WorkQueue<int> >);
    CPPUNIT_TEST_SUITE_END();
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestWorkQueue, TestSet::perCommit());

/// Tests for @ref GenerationalWorkQueue
class TestGenerationalWorkQueue : public TestWorkQueueBase<GenerationalWorkQueue<int> >
{
    CPPUNIT_TEST_SUB_SUITE(TestGenerationalWorkQueue, TestWorkQueueBase<GenerationalWorkQueue<int> >);
    CPPUNIT_TEST(testProducer);
    CPPUNIT_TEST(testStressGens);
    CPPUNIT_TEST_SUITE_END();

private:
    unsigned int genScale;
    unsigned int nextItem;
    unsigned int numItems;
    boost::mutex itemMutex;
    boost::scoped_ptr<GenerationalWorkQueue<int> > queue;

    vector<pair<int, unsigned int> > output;

    void producerThread();
    void consumerThread();

public:
    void testProducer();            ///< Test the producer functions

    /**
     * Stress test using generations. The work items are simply non-negative
     * integers, where the generation of a work item is the item divided by a
     * scale factor. The producers pull the items off a virtual queue (by
     * atomically incrementing a counter) and push them to the generational
     * queue. A single consumer pulls the items off the queue and push them in
     * a vector for later verification.
     */
    void testStressGens();
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestGenerationalWorkQueue, TestSet::perCommit());

template<typename T>
void TestWorkQueueBase<T>::producerThread(T &queue, int start, int end)
{
    for (int i = start; i < end; i++)
        Push<T>()(queue, i);
}

template<typename T>
void TestWorkQueueBase<T>::consumerThread(T &queue, vector<int> &out, boost::mutex &mutex)
{
    int next;
    while ((next = Pop<T>()(queue)) >= 0)
    {
        boost::lock_guard<boost::mutex> lock(mutex);
        out.push_back(next);
    }
}

template<typename T>
void TestWorkQueueBase<T>::testCapacity()
{
    T queue(5);
    CPPUNIT_ASSERT_EQUAL(typename T::size_type(5), queue.capacity());
    for (int i = 0; i < 5; i++)
        Push<T>()(queue, i);
    CPPUNIT_ASSERT_EQUAL(typename T::size_type(5), queue.capacity());
}

template<typename T>
void TestWorkQueueBase<T>::testSize()
{
    T queue(2);
    Push<T> pusher;
    Pop<T> popper;

    CPPUNIT_ASSERT_EQUAL(typename T::size_type(0), queue.size());
    pusher(queue, 1);
    CPPUNIT_ASSERT_EQUAL(typename T::size_type(1), queue.size());
    pusher(queue, 2);
    CPPUNIT_ASSERT_EQUAL(typename T::size_type(2), queue.size());
    popper(queue);
    CPPUNIT_ASSERT_EQUAL(typename T::size_type(1), queue.size());
    popper(queue);
    CPPUNIT_ASSERT_EQUAL(typename T::size_type(0), queue.size());
}

template<typename T>
void TestWorkQueueBase<T>::testStress()
{
    const int numProducers = 8;
    const int numConsumers = 8;
    const int capacity = 4;
    const int elements = 1000000;
    boost::ptr_vector<boost::thread> producers;
    boost::ptr_vector<boost::thread> consumers;
    vector<int> out;
    boost::mutex outMutex;
    T queue(capacity);

    for (int i = 0; i < numProducers; i++)
    {
        int start = elements * i / numProducers;
        int end = elements * (i + 1) / numProducers;
        producers.push_back(new boost::thread(
                boost::bind(&TestWorkQueueBase<T>::producerThread, boost::ref(queue), start, end)));
    }
    for (int i = 0; i < numConsumers; i++)
    {
        consumers.push_back(new boost::thread(
                boost::bind(&TestWorkQueueBase<T>::consumerThread,
                            boost::ref(queue), boost::ref(out), boost::ref(outMutex))));
    }

    for (int i = 0; i < numProducers; i++)
        producers[i].join();
    for (int i = 0; i < numConsumers; i++)
        Push<T>()(queue, -1); // signals one consumer to shut down
    for (int i = 0; i < numConsumers; i++)
        consumers[i].join();

    CPPUNIT_ASSERT_EQUAL(elements, int(out.size()));
    sort(out.begin(), out.end());
    for (int i = 0; i < elements; i++)
        CPPUNIT_ASSERT_EQUAL(i, out[i]);
}

void TestGenerationalWorkQueue::testProducer()
{
    GenerationalWorkQueue<int> queue(5);
    CPPUNIT_ASSERT_THROW(queue.producerStop(2), std::logic_error);
    queue.producerStart(1);
    CPPUNIT_ASSERT_THROW(queue.producerStop(2), std::logic_error);
    CPPUNIT_ASSERT_THROW(queue.producerNext(1, 0), std::invalid_argument); // going backwards
    queue.producerNext(1, 1);
    queue.producerNext(1, 3);
    CPPUNIT_ASSERT_THROW(queue.producerStop(2), std::logic_error);
    queue.producerStop(3);
    CPPUNIT_ASSERT_THROW(queue.producerStop(3), std::logic_error);
}

void TestGenerationalWorkQueue::producerThread()
{
    unsigned int curGen = 0;
    while (true)
    {
        unsigned int item;

        {
            boost::unique_lock<boost::mutex> lock(itemMutex);
            item = nextItem++;
        }

        if (item >= numItems)
            break;

        unsigned int nextGen = item / genScale;
        if (nextGen != curGen)
            queue->producerNext(curGen, nextGen);
        curGen = nextGen;
        queue->push(curGen, item);
    }
    queue->producerStop(curGen);
}

void TestGenerationalWorkQueue::consumerThread()
{
    while (true)
    {
        unsigned int gen;
        int item = queue->pop(gen);
        if (item < 0)
            break;
        output.push_back(make_pair(item, gen));
    }
}

void TestGenerationalWorkQueue::testStressGens()
{
    const unsigned int numProducers = 8;
    const unsigned int capacity = 20;
    genScale = 50;
    nextItem = 0;
    numItems = 1000000;

    boost::ptr_vector<boost::thread> producers;

    queue.reset(new GenerationalWorkQueue<int>(capacity));
    for (unsigned int i = 0; i < numProducers; i++)
    {
        queue->producerStart(0);
        producers.push_back(new boost::thread(
                boost::bind(&TestGenerationalWorkQueue::producerThread, this)));
    }

    boost::thread consumer(boost::bind(&TestGenerationalWorkQueue::consumerThread, this));

    // Wait for producers to complete
    for (unsigned int i = 0; i < numProducers; i++)
        producers[i].join();
    // Shut down the consumer
    queue->pushNoGen(-1);
    consumer.join();

    CPPUNIT_ASSERT_EQUAL(numItems, (unsigned int) output.size());
    for (unsigned int i = 0; i < numItems; i++)
    {
        CPPUNIT_ASSERT_EQUAL(output[i].first / genScale, output[i].second);
        CPPUNIT_ASSERT(i == 0 || output[i].second >= output[i - 1].second); // generation ordering
    }
    sort(output.begin(), output.end());
    for (unsigned int i = 0; i < numItems; i++)
    {
        CPPUNIT_ASSERT_EQUAL(int(i), output[i].first);
    }
}
