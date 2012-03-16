/**
 * @file
 *
 * Test code for @ref SplatSet.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/extensions/HelperMacros.h>
#include <boost/smart_ptr/scoped_ptr.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/bind.hpp>
#include <boost/ref.hpp>
#include <vector>
#include <utility>
#include <tr1/cstdint>
#include "../src/splat.h"
#include "../src/grid.h"
#include "../src/splat_set.h"
#include "../src/logging.h"
#include "../src/statistics.h"
#include "test_splat_set.h"
#include "testmain.h"

using namespace std;
using namespace SplatSet;

/**
 * Create a splat with given position and radius. The other fields
 * are given arbitrary values.
 */
static Splat makeSplat(float x, float y, float z, float radius)
{
    Splat splat;
    splat.position[0] = x;
    splat.position[1] = y;
    splat.position[2] = z;
    splat.radius = radius;
    splat.normal[0] = 1.0f;
    splat.normal[1] = 0.0f;
    splat.normal[2] = 0.0f;
    splat.quality = 1.0f;
    return splat;
}

void createSplats(std::vector<std::vector<Splat> > &splats)
{
    const float z = 10.0f;

    splats.clear();
    splats.resize(5);

    splats[0].push_back(makeSplat(10.0f, 20.0f, z, 2.0f));
    splats[0].push_back(makeSplat(30.0f, 17.0f, z, 1.0f));
    splats[0].push_back(makeSplat(32.0f, 12.0f, z, 1.0f));
    splats[0].push_back(makeSplat(32.0f, 18.0f, z, 1.0f));
    splats[0].push_back(makeSplat(37.0f, 18.0f, z, 1.0f));
    splats[0].push_back(makeSplat(35.0f, 16.0f, z, 3.0f));

    splats[1].push_back(makeSplat(12.0f, 37.0f, z, 1.0f));
    splats[1].push_back(makeSplat(13.0f, 37.0f, z, 1.0f));
    splats[1].push_back(makeSplat(12.0f, 38.0f, z, 1.0f));
    splats[1].push_back(makeSplat(13.0f, 38.0f, z, 1.0f));
    splats[1].push_back(makeSplat(17.0f, 32.0f, z, 1.0f));

    // Leave 2 empty to check skipping over empty ranges

    splats[3].push_back(makeSplat(18.0f, 33.0f, z, 1.0f));

    splats[3].push_back(makeSplat(25.0f, 45.0f, z, 4.0f));

    // Leave 4 empty to check empty ranges at the end
}

/// Tests for @ref SplatSet::internal::splatToBuckets
class TestSplatToBuckets : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestSplatToBuckets);
    CPPUNIT_TEST(testSimple);
    CPPUNIT_TEST(testNan);
    CPPUNIT_TEST(testZero);
    CPPUNIT_TEST_SUITE_END();
public:
    void testSimple();         ///< Test standard case
    void testNan();            ///< Test error checking for illegal splats
    void testZero();           ///< Test error checking for zero @a bucketSize
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestSplatToBuckets, TestSet::perBuild());

void TestSplatToBuckets::testSimple()
{
    const float ref[3] = {10.0f, -50.0f, 40.0f};
    Grid grid(ref, 20.0f, -1, 5, 1, 100, 2, 50);
    // grid base is at (-10, -30, 80)
    boost::array<Grid::difference_type, 3> lower, upper;

    Splat s1 = makeSplat(115.0f, -31.0f, 1090.0f, 7.0f);
    SplatSet::internal::splatToBuckets(s1, grid, 3, lower, upper);
    CPPUNIT_ASSERT_EQUAL(1, int(lower[0]));
    CPPUNIT_ASSERT_EQUAL(2, int(upper[0]));
    CPPUNIT_ASSERT_EQUAL(-1, int(lower[1]));
    CPPUNIT_ASSERT_EQUAL(0, int(upper[1]));
    CPPUNIT_ASSERT_EQUAL(16, int(lower[2]));
    CPPUNIT_ASSERT_EQUAL(16, int(upper[2]));

    Splat s2 = makeSplat(-1000.0f, -1000.0f, -1000.0f, 100.0f);
    SplatSet::internal::splatToBuckets(s2, grid, 3, lower, upper);
    CPPUNIT_ASSERT_EQUAL(-19, int(lower[0]));
    CPPUNIT_ASSERT_EQUAL(-15, int(upper[0]));
    CPPUNIT_ASSERT_EQUAL(-18, int(lower[1]));
    CPPUNIT_ASSERT_EQUAL(-15, int(upper[1]));
    CPPUNIT_ASSERT_EQUAL(-20, int(lower[2]));
    CPPUNIT_ASSERT_EQUAL(-17, int(upper[2]));
}

void TestSplatToBuckets::testNan()
{
    const float ref[3] = {10.0f, -50.0f, 40.0f};
    Grid grid(ref, 20.0f, -1, 5, 1, 100, 2, 50);
    // grid base is at (-10, -30, 80)
    boost::array<Grid::difference_type, 3> lower, upper;
    Splat s = makeSplat(115.0f, std::numeric_limits<float>::quiet_NaN(), 1090.0f, 7.0f);

    CPPUNIT_ASSERT_THROW(SplatSet::internal::splatToBuckets(s, grid, 3, lower, upper), std::invalid_argument);
}

void TestSplatToBuckets::testZero()
{
    const float ref[3] = {10.0f, -50.0f, 40.0f};
    Grid grid(ref, 20.0f, -1, 5, 1, 100, 2, 50);
    // grid base is at (-10, -30, 80)
    boost::array<Grid::difference_type, 3> lower, upper;

    Splat s = makeSplat(115.0f, -31.0f, 1090.0f, 7.0f);

    CPPUNIT_ASSERT_THROW(SplatSet::internal::splatToBuckets(s, grid, 0, lower, upper), std::invalid_argument);
}

/// Base class for testing @ref SplatSet::SimpleSet and equivalent classes.
template<typename SetType>
class TestSplatSet : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestSplatSet);
    CPPUNIT_TEST(testForEach);
    CPPUNIT_TEST(testForEachRange);
    CPPUNIT_TEST(testForEachRangeAll);
    CPPUNIT_TEST(testNan);
    CPPUNIT_TEST_SUITE_END_ABSTRACT();

protected:
    typedef SetType Set;
    typedef StdVectorCollection<Splat> SplatCollection;
    typedef boost::ptr_vector<SplatCollection> SplatCollections;

    std::vector<std::vector<Splat> > splatData;
    SplatCollections splats;
    boost::scoped_ptr<Set> set;

    virtual Set *setFactory(const SplatCollections &splats,
                            float spacing, Grid::size_type bucketSize) = 0;

    /// Captures the parameters given to the function object
    struct Entry
    {
        typename Set::scan_type scan;
        typename Set::index_type first;
        typename Set::index_type last;
        boost::array<Grid::difference_type, 3> lower;
        boost::array<Grid::difference_type, 3> upper;
    };

    /// Function to collect callback data from @c forEach and its like
    void callback(std::vector<Entry> &entries,
                  typename Set::scan_type scan,
                  typename Set::index_type first,
                  typename Set::index_type last,
                  const boost::array<Grid::difference_type, 3> &lower,
                  const boost::array<Grid::difference_type, 3> &upper);

    /// Check that retrieved entries match what is expected
    void validate(const std::vector<Entry> &entries,
                  const std::vector<Range> &ranges,
                  const Set &set,
                  const Grid &grid,
                  Grid::size_type bucketSize);

public:
    virtual void setUp();
    virtual void tearDown();

    virtual void testForEach();      ///< Tests @c forEach
    void testForEachRange();         ///< Tests @c forEachRange
    virtual void testForEachRangeAll(); ///< Tests @c forEachRange where the ranges span everything
    void testNan();                  ///< Tests handling of invalid splats
};

/// Tests for @ref SplatSet::SimpleSet
class TestSplatSetSimple : public TestSplatSet<SimpleSet<boost::ptr_vector<StdVectorCollection<Splat> > > >
{
    CPPUNIT_TEST_SUB_SUITE(TestSplatSetSimple, TestSplatSet<Set>);
    CPPUNIT_TEST_SUITE_END();
protected:
    virtual Set *setFactory(const SplatCollections &splats,
                            float spacing, Grid::size_type bucketSize);
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestSplatSetSimple, TestSet::perBuild());

/// Tests for @ref SplatSet::BlobSet
class TestSplatSetBlob : public TestSplatSet<BlobSet<boost::ptr_vector<StdVectorCollection<Splat> >, std::vector<Blob> > >
{
    CPPUNIT_TEST_SUB_SUITE(TestSplatSetBlob, TestSplatSet<Set>);
    CPPUNIT_TEST(testGetBoundingGrid);
    CPPUNIT_TEST_SUITE_END();
protected:
    virtual Set *setFactory(const SplatCollections &splats,
                            float spacing, Grid::size_type bucketSize);
public:
    void testGetBoundingGrid();      ///< Tests construction of the bounding grid

    // Overloads that check that the fast path was hit
    virtual void testForEach();
    virtual void testForEachRangeAll();
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestSplatSetBlob, TestSet::perBuild());


template<typename SetType>
void TestSplatSet<SetType>::setUp()
{
    CppUnit::TestFixture::setUp();
    createSplats(splatData, splats);
    set.reset(setFactory(splats, 2.5f, 2));
    /* The testNan test normally causes a warning to be printed about
     * invalid splats, but in this case it's intentional, so we
     * suppress the warning.
     */
    Log::log.setLevel(Log::error);
}

template<typename SetType>
void TestSplatSet<SetType>::tearDown()
{
    // Restore the default log level
    Log::log.setLevel(Log::warn);
    CppUnit::TestFixture::tearDown();
}

template<typename SetType>
void TestSplatSet<SetType>::callback(
    std::vector<Entry> &entries,
    typename Set::scan_type scan,
    typename Set::index_type first,
    typename Set::index_type last,
    const boost::array<Grid::difference_type, 3> &lower,
    const boost::array<Grid::difference_type, 3> &upper)
{
    Entry e;
    e.scan = scan;
    e.first = first;
    e.last = last;
    e.lower = lower;
    e.upper = upper;
    entries.push_back(e);
}

template<typename SetType>
void TestSplatSet<SetType>::validate(const std::vector<Entry> &entries,
                                     const std::vector<Range> &ranges,
                                     const Set &set,
                                     const Grid &grid,
                                     Grid::size_type bucketSize)
{
    typedef typename Set::scan_type scan_type;
    typedef typename Set::index_type index_type;

    /* First check that we got exactly the splats we expected and in the
     * same order. We do this the lazy way, by generating lists of expected
     * and actual values.
     */
    std::vector<std::pair<scan_type, index_type> > actualIds, expectedIds;

    for (std::size_t i = 0; i < entries.size(); i++)
    {
        for (index_type j = entries[i].first; j < entries[i].last; j++)
            actualIds.push_back(std::make_pair(entries[i].scan, j));
    }
    for (std::size_t i = 0; i < ranges.size(); i++)
    {
        for (index_type j = ranges[i].start; j < ranges[i].start + ranges[i].size; j++)
            expectedIds.push_back(std::make_pair(scan_type(ranges[i].scan), j));
    }

    CPPUNIT_ASSERT_EQUAL(expectedIds.size(), actualIds.size());
    for (std::size_t i = 0; i < expectedIds.size(); i++)
    {
        CPPUNIT_ASSERT_EQUAL(expectedIds[i].first, actualIds[i].first);
        CPPUNIT_ASSERT_EQUAL(expectedIds[i].second, actualIds[i].second);
    }

    /* Now check that in each case the right range of buckets was given */
    for (size_t i = 0; i < entries.size(); i++)
    {
        for (index_type j = entries[i].first; j < entries[i].last; j++)
        {
            Splat splat;
            set.getSplats()[entries[i].scan].read(j, j + 1, &splat);

            float loWorld[3], hiWorld[3];
            Grid::difference_type loCell[3], hiCell[3];
            Grid::difference_type loBucket[3], hiBucket[3];
            for (unsigned int k = 0; k < 3; k++)
            {
                loWorld[k] = splat.position[k] - splat.radius;
                hiWorld[k] = splat.position[k] + splat.radius;
            }
            grid.worldToCell(loWorld, loCell);
            grid.worldToCell(hiWorld, hiCell);
            for (unsigned int k = 0; k < 3; k++)
            {
                loBucket[k] = divDown(loCell[k], bucketSize);
                hiBucket[k] = divDown(hiCell[k], bucketSize);
            }
            for (unsigned int k = 0; k < 3; k++)
            {
                CPPUNIT_ASSERT_EQUAL(loBucket[k], entries[i].lower[k]);
                CPPUNIT_ASSERT_EQUAL(hiBucket[k], entries[i].upper[k]);
            }
        }
    }
}

template<typename SetType>
void TestSplatSet<SetType>::testForEach()
{
    const float ref[3] = {0.0f, 0.0f, 0.0f};
    Grid grid(ref, 2.5f, 0, 20, 0, 20, 0, 20);
    std::vector<Entry> entries;
    set->forEach(grid, 4, boost::bind(&TestSplatSet<SetType>::callback, this,
                                      boost::ref(entries), _1, _2, _3, _4, _5));

    std::vector<Range> ranges;
    for (size_t i = 0; i < splatData.size(); i++)
        ranges.push_back(Range(i, 0, splatData[i].size()));
    validate(entries, ranges, *set, grid, 4);
}

template<typename SetType>
void TestSplatSet<SetType>::testForEachRange()
{
    std::vector<Range> ranges;
    ranges.push_back(Range(0, 1, 2));
    ranges.push_back(Range(0, 3, 1));
    ranges.push_back(Range(1, 0, 1));
    ranges.push_back(Range(1, 3, 2));
    ranges.push_back(Range(3, 0, 1));

    const float ref[3] = {0.0f, 0.0f, 0.0f};
    Grid grid(ref, 2.5f, 0, 20, 0, 20, 0, 20);
    std::vector<Entry> entries;
    set->forEachRange(ranges.begin(), ranges.end(),
                      grid, 4, boost::bind(&TestSplatSet<SetType>::callback, this,
                                           boost::ref(entries), _1, _2, _3, _4, _5));
    validate(entries, ranges, *set, grid, 4);
}

template<typename SetType>
void TestSplatSet<SetType>::testForEachRangeAll()
{
    std::vector<Range> ranges;
    for (size_t i = 0; i < splatData.size(); i++)
    {
        if (!splatData[i].empty())
            ranges.push_back(Range(i, 0, splatData[i].size()));
    }

    const float ref[3] = {0.0f, 0.0f, 0.0f};
    Grid grid(ref, 2.5f, -20, 20, 4, 20, -20, 20);
    std::vector<Entry> entries;
    set->forEachRange(ranges.begin(), ranges.end(),
                      grid, 4, boost::bind(&TestSplatSet<SetType>::callback, this,
                                           boost::ref(entries), _1, _2, _3, _4, _5));
    validate(entries, ranges, *set, grid, 4);
}

template<typename SetType>
void TestSplatSet<SetType>::testNan()
{
    std::vector<Range> ranges;
    for (size_t i = 0; i < splatData.size(); i++)
        ranges.push_back(Range(i, 0, splatData[i].size()));

    // To make things more interesting, make all the coordinates negative to check rounding
    for (size_t i = 0; i < splatData.size(); i++)
        for (size_t j = 0; j < splatData[i].size(); j++)
        {
            splatData[i][j].position[0] *= -1.0f;
            splatData[i][j].position[1] *= -1.0f;
            splatData[i][j].position[2] *= -1.0f;
        }
    // Add some extra invalid entries, AFTER the ranges have been set up,
    // so that validate will not expect to see those splats.
    const float n = std::numeric_limits<float>::quiet_NaN();
    splatData[0].push_back(makeSplat(n, 0.0f, 0.0f, 1.0f));
    splatData[0].push_back(makeSplat(1.0f, 0.0f, 0.0f, n));

    // Need to recreate the blobs are that
    set.reset(setFactory(splats, 2.5f, 2));

    const float ref[3] = {0.0f, 0.0f, 0.0f};
    Grid grid(ref, 2.5f, 0, 20, 0, 20, 0, 20);
    std::vector<Entry> entries;
    set->forEach(grid, 4, boost::bind(&TestSplatSet<SetType>::callback, this,
                                      boost::ref(entries), _1, _2, _3, _4, _5));
    validate(entries, ranges, *set, grid, 4);
}

TestSplatSetSimple::Set *TestSplatSetSimple::setFactory(
    const SplatCollections &collections, float spacing, Grid::size_type bucketSize)
{
    (void) spacing;
    (void) bucketSize;
    return new Set(collections);
}

TestSplatSetBlob::Set *TestSplatSetBlob::setFactory(
    const SplatCollections &collections, float spacing, Grid::size_type bucketSize)
{
    return new Set(collections, spacing, bucketSize);
}

void TestSplatSetBlob::testGetBoundingGrid()
{
    Grid grid = set->getBoundingGrid();
    CPPUNIT_ASSERT_EQUAL(2.5f, grid.getSpacing());
    CPPUNIT_ASSERT_EQUAL(0.0f, grid.getReference()[0]);
    CPPUNIT_ASSERT_EQUAL(0.0f, grid.getReference()[1]);
    CPPUNIT_ASSERT_EQUAL(0.0f, grid.getReference()[2]);
    CPPUNIT_ASSERT_EQUAL(Grid::difference_type(2), grid.getExtent(0).first);
    CPPUNIT_ASSERT_EQUAL(Grid::difference_type(16), grid.getExtent(0).second);
    CPPUNIT_ASSERT_EQUAL(Grid::difference_type(4), grid.getExtent(1).first);
    CPPUNIT_ASSERT_EQUAL(Grid::difference_type(20), grid.getExtent(1).second);
    CPPUNIT_ASSERT_EQUAL(Grid::difference_type(2), grid.getExtent(2).first);
    CPPUNIT_ASSERT_EQUAL(Grid::difference_type(6), grid.getExtent(2).second);
}

void TestSplatSetBlob::testForEach()
{
    Statistics::Variable &hits = Statistics::getStatistic<Statistics::Variable>("blobset.foreach.fast");
    double oldTotal = hits.getNumSamples() > 0 ? hits.getMean() * hits.getNumSamples() : 0.0;
    TestSplatSet<Set>::testForEach();
    double newTotal = hits.getMean() * hits.getNumSamples();
    CPPUNIT_ASSERT_DOUBLES_EQUAL(oldTotal + 1.0, newTotal, 1e-2);
}

void TestSplatSetBlob::testForEachRangeAll()
{
    Statistics::Variable &hits = Statistics::getStatistic<Statistics::Variable>("blobset.foreachrange.fast");
    double oldTotal = hits.getNumSamples() > 0 ? hits.getMean() * hits.getNumSamples() : 0.0;
    TestSplatSet<Set>::testForEachRangeAll();
    double newTotal = hits.getMean() * hits.getNumSamples();
    CPPUNIT_ASSERT_DOUBLES_EQUAL(oldTotal + 1.0, newTotal, 1e-2);
}
