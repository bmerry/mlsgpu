/*
 * mlsgpu: surface reconstruction from point clouds
 * Copyright (C) 2013  University of Cape Town
 *
 * This file is part of mlsgpu.
 *
 * mlsgpu is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

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
#include "testutil.h"
#include "../src/circular_buffer.h"
#include "../src/work_queue.h"
#include "../src/tr1_cstdint.h"
#include "../src/statistics.h"

/**
 * Functionality tests for @ref CircularBuffer. These tests do not exercise
 * any blocking-related behavior, as that is covered in @ref
 * TestCircularBufferStress.
 */
class TestCircularBuffer : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestCircularBuffer);
    CPPUNIT_TEST(testAllocateFree);
    CPPUNIT_TEST(testSize);
    CPPUNIT_TEST(testStatistics);
#if DEBUG
    CPPUNIT_TEST(testCreateZero);
    CPPUNIT_TEST(testTooLarge);
    CPPUNIT_TEST(testOverflow);
    CPPUNIT_TEST(testZero);
#endif
    CPPUNIT_TEST(testUnallocated);
    CPPUNIT_TEST_SUITE_END();

private:
    void testCreateZero();      ///< Test that an exception is thrown on creating a zero-size buffer
    void testAllocateFree();    ///< Smoke test for @ref CircularBuffer::allocate and @ref CircularBuffer::free
    void testSize();            ///< Test @ref CircularBufferBase::size
    void testStatistics();      ///< Test that memory allocation is accounted for
    void testTooLarge();        ///< Test exception handling when asking for too much memory
    void testOverflow();        ///< Test exception handling when total size overflows
    void testZero();            ///< Test that an exception is thrown when asking for zero elements
    void testUnallocated();     ///< Test @ref CircularBufferBase::unallocated
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestCircularBuffer, TestSet::perBuild());

void TestCircularBuffer::testCreateZero()
{
    CPPUNIT_ASSERT_THROW(CircularBuffer("zero", 0), std::invalid_argument);
}

void TestCircularBuffer::testAllocateFree()
{
    Timeplot::Worker tworker("test");

    CircularBuffer buffer("test", 10);
    CircularBuffer::Allocation alloc = buffer.allocate(tworker, sizeof(short), 2);
    void *item = alloc.get();
    CPPUNIT_ASSERT(item != NULL);

    // Check that the memory can be safely written
    short *values = reinterpret_cast<short *>(item);
    values[0] = 123;
    values[1] = 456;

    buffer.free(alloc);;
}

void TestCircularBuffer::testSize()
{
    CircularBuffer buffer("test", 1000);
    CPPUNIT_ASSERT_EQUAL(std::size_t(1000), buffer.size());
}

void TestCircularBuffer::testStatistics()
{
    typedef Statistics::Peak Peak;
    Peak &allStat = Statistics::getStatistic<Peak>("mem.all");
    std::size_t oldMem = allStat.get();

    CircularBuffer buffer("test", 1000);

    std::size_t newMem = allStat.get();
    // Can't make this an exact test, because the linked list can also allocate
    CPPUNIT_ASSERT(newMem >= oldMem + 1000);
}

void TestCircularBuffer::testTooLarge()
{
    Timeplot::Worker tworker("test");
    CircularBuffer buffer("test", 999);
    CPPUNIT_ASSERT_THROW(buffer.allocate(tworker, 1000, 1, NULL), std::out_of_range);
    CPPUNIT_ASSERT_THROW(buffer.allocate(tworker, 1, 1000, NULL), std::out_of_range);
    CPPUNIT_ASSERT_THROW(buffer.allocate(tworker, 100, 100, NULL), std::out_of_range);
    CPPUNIT_ASSERT_THROW(buffer.allocate(tworker, 1000), std::out_of_range);
}

void TestCircularBuffer::testOverflow()
{
    Timeplot::Worker tworker("test");
    CircularBuffer buffer("test", 1000);
    CPPUNIT_ASSERT_THROW(buffer.allocate(tworker, 8, std::numeric_limits<std::size_t>::max() / 2 + 2), std::out_of_range);
}

void TestCircularBuffer::testZero()
{
    Timeplot::Worker tworker("test");
    CircularBuffer buffer("test", 16);
    CPPUNIT_ASSERT_THROW(buffer.allocate(tworker, 4, 0, NULL), std::invalid_argument);
    CPPUNIT_ASSERT_THROW(buffer.allocate(tworker, 0, 4, NULL), std::invalid_argument);
    CPPUNIT_ASSERT_THROW(buffer.allocate(tworker, 0), std::invalid_argument);
}

/// Stress tests for @ref CircularBuffer
class TestCircularBufferStress : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestCircularBufferStress);
    CPPUNIT_TEST(testStress);
    CPPUNIT_TEST_SUITE_END();

public:
    TestCircularBufferStress()
        : buffer("mem.TestCircularBufferStress", 123 * sizeof(std::tr1::uint64_t)), workQueue() {}

private:
    struct Item
    {
        std::size_t start, end; // expected start and end values in the buffer
        CircularBuffer::Allocation alloc;

        // Default-constructed item is an end-of-work sentinel
        Item() : start(0), end(0) {}
    };

    std::tr1::uint64_t badCount;
    boost::mutex badMutex;

    CircularBuffer buffer;
    WorkQueue<Item> workQueue;   ///< Ranges sent from producer to consumer

    /**
     * Generates the numbers from @a start up to @a end and places them
     * in chunks of the buffer. The subranges are enqueued on @ref workQueue.
     */
    void producerThread(std::tr1::uint64_t start, std::tr1::uint64_t end);

    /**
     * Pulls items off the queue, verifies the contents and then returns
     * the memory to the buffer.
     */
    void consumerThread();

    /**
     * Pass a lot of numbers from @ref producerThread to the main thread,
     * checking that they arrive correctly formed.
     */
    void testStress();
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestCircularBufferStress, TestSet::perCommit());

void TestCircularBufferStress::producerThread(std::tr1::uint64_t start, std::tr1::uint64_t end)
{
    Timeplot::Worker tworker("producer");

    std::tr1::mt19937 engine;
    std::tr1::uint64_t cur = start;
    std::tr1::uniform_int<std::tr1::uint32_t> chunkDist(1, buffer.size() / sizeof(cur));

    while (cur < end)
    {
        std::tr1::uint64_t elements = chunkDist(engine);
        elements = std::min(elements, end - cur);
        CircularBuffer::Allocation alloc = buffer.allocate(tworker, sizeof(cur), elements);

        std::tr1::uint64_t *ptr = static_cast<std::tr1::uint64_t *>(alloc.get());
        CPPUNIT_ASSERT(ptr != NULL);
        for (std::size_t i = 0; i < elements; i++)
        {
            ptr[i] = cur++;
        }

        Item item;
        item.start = cur - elements;
        item.end = cur;
        item.alloc = alloc;
        workQueue.push(item);
    }
}

void TestCircularBufferStress::consumerThread()
{
    /* This generator doesn't do anything useful - it's just a way to
     * make sure that the producer and consumer run at about the same
     * rate and hence test both full and empty conditions.
     */
    std::tr1::mt19937 gen;
    std::tr1::uniform_int<std::tr1::uint32_t> chunkDist(1, buffer.size() / sizeof(std::tr1::uint64_t));

    std::tr1::uint64_t bad = 0;
    while (true)
    {
        Item item;
        item = workQueue.pop();
        if (item.start == item.end)
            break; // end-of-work marker

        const std::tr1::uint64_t *ptr = (const std::tr1::uint64_t *) item.alloc.get();
        for (std::tr1::uint64_t i = item.start; i != item.end; i++)
        {
            if (*ptr != i)
                bad++;
            ptr++;
        }

        (void) chunkDist(gen);
        buffer.free(item.alloc);
    }

    boost::lock_guard<boost::mutex> lock(badMutex);
    badCount += bad;
}

void TestCircularBufferStress::testStress()
{
    const std::size_t perThread = 10000000;
    const std::size_t numProducers = 4;
    const std::size_t numConsumers = 3;
    boost::thread_group producers;
    boost::thread_group consumers;
    badCount = 0;

    for (std::size_t i = 0; i < numProducers; i++)
        producers.create_thread(boost::bind(&TestCircularBufferStress::producerThread, this,
                                            perThread * i, perThread * (i + 1)));
    for (std::size_t i = 0; i < numConsumers; i++)
        consumers.create_thread(boost::bind(&TestCircularBufferStress::consumerThread, this));

    producers.join_all();
    workQueue.stop();
    consumers.join_all();
    CPPUNIT_ASSERT_EQUAL(std::tr1::uint64_t(0), badCount);
}

void TestCircularBuffer::testUnallocated()
{
    Timeplot::Worker worker("test");
    CircularBuffer buffer("test", 10);

    MLSGPU_ASSERT_EQUAL(10, buffer.unallocated());

    CircularBuffer::Allocation a1 = buffer.allocate(worker, 3);
    CircularBuffer::Allocation a2 = buffer.allocate(worker, 1);

    MLSGPU_ASSERT_EQUAL(6, buffer.unallocated());

    buffer.free(a2); // does not make more space available until a1 freed
    MLSGPU_ASSERT_EQUAL(6, buffer.unallocated());

    CircularBuffer::Allocation a3 = buffer.allocate(worker, 5);
    MLSGPU_ASSERT_EQUAL(1, buffer.unallocated());

    buffer.free(a1);
    MLSGPU_ASSERT_EQUAL(5, buffer.unallocated());

    CircularBuffer::Allocation a4 = buffer.allocate(worker, 3); // wastes 1 slot at end
    MLSGPU_ASSERT_EQUAL(1, buffer.unallocated());

    buffer.free(a3);
    buffer.free(a4);
    MLSGPU_ASSERT_EQUAL(10, buffer.unallocated());
}
