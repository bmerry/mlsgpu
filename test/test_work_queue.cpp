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

class TestWorkQueue : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestWorkQueue);
    CPPUNIT_TEST(testCapacity);
    CPPUNIT_TEST(testSize);
    CPPUNIT_TEST(testStress);
    CPPUNIT_TEST_SUITE_END();
private:
    /**
     * Adds a sequence of consecutive integers to the work queue.
     */
    static void producerThread(WorkQueue<int> &queue, int start, int count);
    /**
     * Pulls integers from a work queue and appends them to a vector. The
     * vector is locked while adding to it. A negative valid in the queue is
     * used to signal shutdown.
     */
    static void consumerThread(WorkQueue<int> &queue, vector<int> &out, boost::mutex &mutex);

public:
    void testCapacity();         ///< Test the capacity function and that it can hold that much
    void testSize();             ///< Test the size function
    void testStress();           ///< Many-threaded stress test
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestWorkQueue, TestSet::perCommit());

void TestWorkQueue::producerThread(WorkQueue<int> &queue, int start, int end)
{
    for (int i = start; i < end; i++)
        queue.push(i);
}

void TestWorkQueue::consumerThread(WorkQueue<int> &queue, vector<int> &out, boost::mutex &mutex)
{
    int next;
    while ((next = queue.pop()) >= 0)
    {
        boost::lock_guard<boost::mutex> lock(mutex);
        out.push_back(next);
    }
}

void TestWorkQueue::testCapacity()
{
    WorkQueue<int> queue(5);
    CPPUNIT_ASSERT_EQUAL(WorkQueue<int>::size_type(5), queue.capacity());
    for (int i = 0; i < 5; i++)
        queue.push(i);
    CPPUNIT_ASSERT_EQUAL(WorkQueue<int>::size_type(5), queue.capacity());
}

void TestWorkQueue::testSize()
{
    WorkQueue<int> queue(2);
    CPPUNIT_ASSERT_EQUAL(WorkQueue<int>::size_type(0), queue.size());
    queue.push(1);
    CPPUNIT_ASSERT_EQUAL(WorkQueue<int>::size_type(1), queue.size());
    queue.push(2);
    CPPUNIT_ASSERT_EQUAL(WorkQueue<int>::size_type(2), queue.size());
    queue.pop();
    CPPUNIT_ASSERT_EQUAL(WorkQueue<int>::size_type(1), queue.size());
    queue.pop();
    CPPUNIT_ASSERT_EQUAL(WorkQueue<int>::size_type(0), queue.size());
}

void TestWorkQueue::testStress()
{
    const int numProducers = 8;
    const int numConsumers = 8;
    const int capacity = 4;
    const int elements = 1000000;
    boost::ptr_vector<boost::thread> producers;
    boost::ptr_vector<boost::thread> consumers;
    vector<int> out;
    boost::mutex outMutex;
    WorkQueue<int> queue(capacity);

    for (int i = 0; i < numProducers; i++)
    {
        int start = elements * i / numProducers;
        int end = elements * (i + 1) / numProducers;
        producers.push_back(new boost::thread(
                boost::bind(&TestWorkQueue::producerThread, boost::ref(queue), start, end)));
    }
    for (int i = 0; i < numConsumers; i++)
    {
        consumers.push_back(new boost::thread(
                boost::bind(&TestWorkQueue::consumerThread,
                            boost::ref(queue), boost::ref(out), boost::ref(outMutex))));
    }

    for (int i = 0; i < numProducers; i++)
        producers[i].join();
    for (int i = 0; i < numConsumers; i++)
        queue.push(-1); // signals one consumer to shut down
    for (int i = 0; i < numConsumers; i++)
        consumers[i].join();

    CPPUNIT_ASSERT_EQUAL(elements, int(out.size()));
    sort(out.begin(), out.end());
    for (int i = 0; i < elements; i++)
        CPPUNIT_ASSERT_EQUAL(i, out[i]);
}
