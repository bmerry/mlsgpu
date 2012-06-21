/**
 * @file
 *
 * Test code for @ref CircularBuffer.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif

#include <cstddef>
#include <algorithm>
#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/extensions/HelperMacros.h>
#include <boost/bind.hpp>
#include <boost/tr1/random.hpp>
#include <boost/thread.hpp>
#include "testmain.h"
#include "../src/circular_buffer.h"
#include "../src/work_queue.h"
#include "../src/tr1_cstdint.h"

/// Stress tests for @ref CircularBuffer
class TestCircularBufferStress : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestCircularBufferStress);
    CPPUNIT_TEST(testStress);
    CPPUNIT_TEST_SUITE_END();

public:
    TestCircularBufferStress()
        : buffer("mem.TestCircularBufferStress", 123), workQueue(10) {}

private:
    struct Item
    {
        std::tr1::uint64_t *ptr;
        std::size_t elements;
    };

    CircularBuffer buffer;
    WorkQueue<Item> workQueue;   ///< Ranges sent from producer to consumer

    /**
     * Generates the numbers from 0 up to @a total and places them
     * in chunks of the buffer. The subranges are enqueued on @ref workQueue.
     */
    void producerThread(std::tr1::uint64_t total);

    /**
     * Pass a lot of numbers from @ref producerThread to the main thread,
     * checking that they arrive correctly formed.
     */
    void testStress();
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestCircularBufferStress, TestSet::perCommit());

void TestCircularBufferStress::producerThread(std::tr1::uint64_t total)
{
    std::tr1::mt19937 engine;
    std::tr1::uint64_t cur = 0;
    std::tr1::uniform_int<std::tr1::uint32_t> chunkDist(1, buffer.size() * 2 / sizeof(cur));

    while (cur < total)
    {
        std::tr1::uint64_t max = chunkDist(engine);
        max = std::min(max, total - cur);
        std::pair<void *, std::size_t> chunk = buffer.allocate(sizeof(cur), max);
        CPPUNIT_ASSERT(chunk.first != NULL);
        CPPUNIT_ASSERT(chunk.second > 0);
        CPPUNIT_ASSERT(chunk.second <= max);

        std::tr1::uint64_t *ptr = static_cast<std::tr1::uint64_t *>(chunk.first);
        for (std::size_t i = 0; i < chunk.second; i++)
        {
            ptr[i] = cur++;
        }

        Item item;
        item.ptr = ptr;
        item.elements = chunk.second;
        workQueue.push(item);
    }

    Item item;
    item.ptr = NULL;
    workQueue.push(item);
}

void TestCircularBufferStress::testStress()
{
    const std::size_t total = 10000000;
    boost::thread producer(boost::bind(&TestCircularBufferStress::producerThread, this, total));

    std::tr1::uint64_t expect = 0;
    Item item;

    /* This generator doesn't do anything useful - it's just a way to
     * make sure that the producer and consumer run at about the same
     * rate and hence test both full and empty conditions.
     */
    std::tr1::mt19937 gen;
    std::tr1::uniform_int<std::tr1::uint32_t> chunkDist(1, buffer.size() * 2 / sizeof(std::tr1::uint64_t));

    while ((item = workQueue.pop()).ptr != NULL)
    {
        CPPUNIT_ASSERT(item.elements > 0 && item.elements < buffer.size());
        for (std::size_t i = 0; i < item.elements; i++)
        {
            CPPUNIT_ASSERT_EQUAL(expect, item.ptr[i]);
            expect++;
        }
        buffer.free(item.ptr, sizeof(std::tr1::uint64_t), item.elements);
        (void) chunkDist(gen);
    }
    CPPUNIT_ASSERT_EQUAL(total, expect);

    producer.join();
}
