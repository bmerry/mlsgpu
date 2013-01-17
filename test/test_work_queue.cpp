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
#include "testutil.h"
#include "../src/work_queue.h"

using namespace std;

/// Tests for @ref WorkQueue
class TestWorkQueue : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestWorkQueue);
    CPPUNIT_TEST(testStress);
    CPPUNIT_TEST_SUITE_END();
private:
    /**
     * Adds a sequence of consecutive integers to the work queue.
     */
    static void producerThread(WorkQueue<int> &queue, int start, int count);

    /**
     * Pulls integers from a work queue and appends them to a vector. The
     * vector is locked while adding to it. A negative value in the queue is
     * used to signal shutdown.
     */
    static void consumerThread(WorkQueue<int> &queue, vector<int> &out, boost::mutex &mutex);

public:
    void testStress();           ///< Stress test with multiple consumers and producers
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
    while ((next = queue.pop()) > 0)
    {
        boost::lock_guard<boost::mutex> lock(mutex);
        out.push_back(next);
    }
}

void TestWorkQueue::testStress()
{
    const int numProducers = 8;
    const int numConsumers = 8;
    const int elements = 1000000;
    boost::ptr_vector<boost::thread> producers;
    boost::ptr_vector<boost::thread> consumers;
    vector<int> out;
    boost::mutex outMutex;
    WorkQueue<int> queue;

    for (int i = 0; i < numProducers; i++)
    {
        int start = 1 + elements * i / numProducers;
        int end = 1 + elements * (i + 1) / numProducers;
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
    queue.stop();
    for (int i = 0; i < numConsumers; i++)
        consumers[i].join();

    CPPUNIT_ASSERT_EQUAL(elements, int(out.size()));
    sort(out.begin(), out.end());
    for (int i = 0; i < elements; i++)
        CPPUNIT_ASSERT_EQUAL(i + 1, out[i]);
}
