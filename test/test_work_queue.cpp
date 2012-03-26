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
#include <boost/ref.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/locks.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/bind.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
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

    void operator()(WorkQueue<ValueType> &queue, const ValueType &value, unsigned int) const
    {
        queue.push(value);
    }
};

template<typename ValueType>
class Push<GenerationalWorkQueue<ValueType> >
{
public:
    typedef void result_type;

    void operator()(GenerationalWorkQueue<ValueType> &queue, const ValueType &value, unsigned int gen) const
    {
        queue.push(value, gen);
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

protected:
    ///< Many-threaded stress test
    void testStressHelper(int numProducers, int numConsumers, bool ordered);

public:
    void testCapacity();         ///< Test the capacity function and that it can hold that much
    void testSize();             ///< Test the size function
    void testStress();           ///< Stress test with multiple consumers and producers
};

class TestWorkQueue : public TestWorkQueueBase<WorkQueue<int> >
{
    CPPUNIT_TEST_SUB_SUITE(TestWorkQueue, TestWorkQueueBase<WorkQueue<int> >);
    CPPUNIT_TEST_SUITE_END();
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestWorkQueue, TestSet::perCommit());

template<typename T>
void TestWorkQueueBase<T>::producerThread(T &queue, int start, int end)
{
    for (int i = start; i < end; i++)
        Push<T>()(queue, i, i);
}

template<typename T>
void TestWorkQueueBase<T>::consumerThread(T &queue, vector<int> &out, boost::mutex &mutex)
{
    int next;
    while ((next = queue.pop()) >= 0)
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
        Push<T>()(queue, i, i);
    CPPUNIT_ASSERT_EQUAL(typename T::size_type(5), queue.capacity());
}

template<typename T>
void TestWorkQueueBase<T>::testSize()
{
    T queue(2);
    Push<T> pusher;

    CPPUNIT_ASSERT_EQUAL(typename T::size_type(0), queue.size());
    pusher(queue, 1, 0);
    CPPUNIT_ASSERT_EQUAL(typename T::size_type(1), queue.size());
    pusher(queue, 2, 1);
    CPPUNIT_ASSERT_EQUAL(typename T::size_type(2), queue.size());
    queue.pop();
    CPPUNIT_ASSERT_EQUAL(typename T::size_type(1), queue.size());
    queue.pop();
    CPPUNIT_ASSERT_EQUAL(typename T::size_type(0), queue.size());
}

template<typename T>
void TestWorkQueueBase<T>::testStressHelper(int numProducers, int numConsumers, bool ordered)
{
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
        Push<T>()(queue, -1, elements + i); // signals one consumer to shut down
    for (int i = 0; i < numConsumers; i++)
        consumers[i].join();

    CPPUNIT_ASSERT_EQUAL(elements, int(out.size()));
    if (!ordered)
        sort(out.begin(), out.end());
    for (int i = 0; i < elements; i++)
        CPPUNIT_ASSERT_EQUAL(i, out[i]);
}

template<typename T>
void TestWorkQueueBase<T>::testStress()
{
    testStressHelper(8, 8, false);
}
