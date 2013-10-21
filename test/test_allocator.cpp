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
 * Tests for @ref Statistics::Allocator and @ref Statistics::Container.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif

#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/extensions/HelperMacros.h>
#include "testutil.h"
#include "../src/statistics.h"
#include "../src/allocator.h"

class TestAllocator : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestAllocator);
    CPPUNIT_TEST(testNoUsage);
    CPPUNIT_TEST(testAllocate);
    CPPUNIT_TEST(testAllocateHint);
    CPPUNIT_TEST(testCopyConstruct);
    CPPUNIT_TEST(testEqual);
#if DEBUG
    CPPUNIT_TEST(testException);
#endif
    CPPUNIT_TEST_SUITE_END();

private:
    void testNoUsage();         ///< Test an allocator that has a null usage
    void testAllocate();        ///< Test allocation (non-hint version)
    void testAllocateHint();    ///< Test allocation that takes a hint
    void testCopyConstruct();   ///< Test copy constructor
    void testEqual();           ///< Test equality operator
    void testException();       ///< Test handling of exception in allocation
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestAllocator, TestSet::perBuild());

void TestAllocator::testNoUsage()
{
    Statistics::Allocator<std::allocator<int> > a;
    int *p = a.allocate(3);
    int *q = a.allocate(4, p);
    a.deallocate(p, 3);
    a.deallocate(q, 4);
}

void TestAllocator::testAllocate()
{
    typedef Statistics::Allocator<std::allocator<int> > A;

    Statistics::Peak peak("peak");
    A a(&peak);
    int *p = a.allocate(3);
    a.deallocate(p, 3);
    MLSGPU_ASSERT_EQUAL(3 * sizeof(int), peak.getMax());
    MLSGPU_ASSERT_EQUAL(0, peak.get());
}

void TestAllocator::testAllocateHint()
{
    typedef Statistics::Allocator<std::allocator<int> > A;

    Statistics::Peak peak("peak");
    A a(&peak);
    int hint;
    int *p = a.allocate(5, &hint);
    a.deallocate(p, 5);
    MLSGPU_ASSERT_EQUAL(5 * sizeof(int), peak.getMax());
    MLSGPU_ASSERT_EQUAL(0, peak.get());
}

void TestAllocator::testCopyConstruct()
{
    typedef Statistics::Allocator<std::allocator<int> > A;

    Statistics::Peak peak1("peak1");
    Statistics::Peak peak2("peak2");
    A alloc(&peak1, &peak2);
    A dup(alloc);

    CPPUNIT_ASSERT_EQUAL(&peak1, dup.usage);
    CPPUNIT_ASSERT_EQUAL(&peak2, dup.allUsage);
}

void TestAllocator::testEqual()
{
    typedef Statistics::Allocator<std::allocator<int> > A;

    Statistics::Peak peak1("peak1"), peak2("peak2");
    A a(&peak1);
    A b(&peak2);
    A c(&peak2);
    CPPUNIT_ASSERT(a != b);
    CPPUNIT_ASSERT(b == c);
}

void TestAllocator::testException()
{
    typedef Statistics::Allocator<std::allocator<int> > A;

    Statistics::Peak peak("peak");
    A a(&peak);

    CPPUNIT_ASSERT_THROW(a.allocate(a.max_size()), std::bad_alloc);
    MLSGPU_ASSERT_EQUAL(0, peak.get());
    MLSGPU_ASSERT_EQUAL(0, peak.getMax());
}

class TestContainers : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestContainers);
    CPPUNIT_TEST(testAll);
    CPPUNIT_TEST_SUITE_END();

private:
    void testAll();     ///< Test all the container types in one test
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestContainers, TestSet::perBuild());

void TestContainers::testAll()
{
    typedef Statistics::Allocator<std::allocator<int> >::size_type size_type;
    typedef Statistics::Peak Peak;
    Peak &peakVector = Statistics::getStatistic<Peak>("mem.vector");
    Peak &peakSet = Statistics::getStatistic<Peak>("mem.set");
    Peak &peakMap = Statistics::getStatistic<Peak>("mem.map");
    Peak &peakAll = Statistics::getStatistic<Peak>("mem.all");

    size_type vectorOld = peakVector.get();
    size_type setOld = peakSet.get();
    size_type mapOld = peakMap.get();
    size_type allOld = peakAll.get();

    Statistics::Container::vector<int> v("mem.vector");
    v.reserve(50);
    Statistics::Container::unordered_set<short> s("mem.set");
    s.insert(123);
    s.insert(567);
    Statistics::Container::unordered_map<int, int> m("mem.map");
    m[1] = 2;
    m[100] = 3;

    size_type vectorNew = peakVector.get();
    size_type setNew = peakSet.get();
    size_type mapNew = peakMap.get();
    size_type allNew = peakAll.get();

    CPPUNIT_ASSERT(vectorNew > vectorOld);
    CPPUNIT_ASSERT(setNew > setOld);
    CPPUNIT_ASSERT(mapNew > mapOld);
    CPPUNIT_ASSERT(allNew > allOld);

    size_type vectorDiff = vectorNew - vectorOld;
    size_type setDiff = setNew - setOld;
    size_type mapDiff = mapNew - mapOld;
    size_type allDiff = allNew - allOld;
    CPPUNIT_ASSERT_EQUAL(vectorDiff + setDiff + mapDiff, allDiff);
}
