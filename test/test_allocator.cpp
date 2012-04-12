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
#include "testmain.h"
#include "../src/statistics.h"
#include "../src/allocator.h"

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
    typedef Statistics::Peak<size_type> Peak;
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
