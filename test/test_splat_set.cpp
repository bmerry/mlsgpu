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
#include "../src/collection.h"
#include "test_splat_set.h"
#include "testmain.h"

using namespace std;
using namespace SplatSet;

/**
 * Tests for Range.
 */
class TestRange : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestRange);
    CPPUNIT_TEST(testConstructor);
    CPPUNIT_TEST(testAppendEmpty);
    CPPUNIT_TEST(testAppendOverflow);
    CPPUNIT_TEST(testAppendMiddle);
    CPPUNIT_TEST(testAppendEnd);
    CPPUNIT_TEST(testAppendGap);
    CPPUNIT_TEST(testAppendNewScan);
    CPPUNIT_TEST_SUITE_END();

public:
    void testConstructor();          ///< Test the constructors
    void testAppendEmpty();          ///< Appending to an empty range
    void testAppendOverflow();       ///< An append that would overflow the size
    void testAppendMiddle();         ///< Append to the middle of an existing range
    void testAppendEnd();            ///< Extend a range
    void testAppendGap();            ///< Append outside (and not adjacent) to an existing range
    void testAppendNewScan();        ///< Append with a different scan
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestRange, TestSet::perBuild());

void TestRange::testConstructor()
{
    Range empty;
    Range single(3, 6);
    Range range(2, UINT64_C(0xFFFFFFFFFFFFFFF0), 0x10);

    CPPUNIT_ASSERT_EQUAL(Range::size_type(0), empty.size);

    CPPUNIT_ASSERT_EQUAL(Range::scan_type(3), single.scan);
    CPPUNIT_ASSERT_EQUAL(Range::index_type(6), single.start);
    CPPUNIT_ASSERT_EQUAL(Range::size_type(1), single.size);

    CPPUNIT_ASSERT_EQUAL(Range::scan_type(2), range.scan);
    CPPUNIT_ASSERT_EQUAL(Range::size_type(0x10), range.size);
    CPPUNIT_ASSERT_EQUAL(Range::index_type(UINT64_C(0xFFFFFFFFFFFFFFF0)), range.start);

    CPPUNIT_ASSERT_THROW(Range(2, UINT64_C(0xFFFFFFFFFFFFFFF0), 0x11), std::out_of_range);
}

void TestRange::testAppendEmpty()
{
    Range range;
    bool success;

    success = range.append(3, 6);
    CPPUNIT_ASSERT_EQUAL(true, success);
    CPPUNIT_ASSERT_EQUAL(Range::size_type(1), range.size);
    CPPUNIT_ASSERT_EQUAL(Range::scan_type(3), range.scan);
    CPPUNIT_ASSERT_EQUAL(Range::index_type(6), range.start);
}

void TestRange::testAppendOverflow()
{
    Range range;
    range.scan = 3;
    range.start = 0x90000000U;
    range.size = 0xFFFFFFFFU;
    bool success = range.append(3, range.start + range.size);
    CPPUNIT_ASSERT_EQUAL(false, success);
    CPPUNIT_ASSERT_EQUAL(Range::size_type(0xFFFFFFFFU), range.size);
    CPPUNIT_ASSERT_EQUAL(Range::scan_type(3), range.scan);
    CPPUNIT_ASSERT_EQUAL(Range::index_type(0x90000000U), range.start);
}

void TestRange::testAppendMiddle()
{
    Range range;
    range.scan = 4;
    range.start = UINT64_C(0x123456781234);
    range.size = 0x10000;
    bool success = range.append(4, UINT64_C(0x12345678FFFF));
    CPPUNIT_ASSERT_EQUAL(true, success);
    CPPUNIT_ASSERT_EQUAL(Range::size_type(0x10000), range.size);
    CPPUNIT_ASSERT_EQUAL(Range::scan_type(4), range.scan);
    CPPUNIT_ASSERT_EQUAL(Range::index_type(UINT64_C(0x123456781234)), range.start);
}

void TestRange::testAppendEnd()
{
    Range range;
    range.scan = 4;
    range.start = UINT64_C(0x123456781234);
    range.size = 0x10000;
    bool success = range.append(4, range.start + range.size);
    CPPUNIT_ASSERT_EQUAL(true, success);
    CPPUNIT_ASSERT_EQUAL(Range::size_type(0x10001), range.size);
    CPPUNIT_ASSERT_EQUAL(Range::scan_type(4), range.scan);
    CPPUNIT_ASSERT_EQUAL(Range::index_type(UINT64_C(0x123456781234)), range.start);
}

void TestRange::testAppendGap()
{
    Range range;
    range.scan = 4;
    range.start = UINT64_C(0x123456781234);
    range.size = 0x10000;
    bool success = range.append(4, range.start + range.size + 1);
    CPPUNIT_ASSERT_EQUAL(false, success);
    CPPUNIT_ASSERT_EQUAL(Range::size_type(0x10000), range.size);
    CPPUNIT_ASSERT_EQUAL(Range::scan_type(4), range.scan);
    CPPUNIT_ASSERT_EQUAL(Range::index_type(UINT64_C(0x123456781234)), range.start);
}

void TestRange::testAppendNewScan()
{
    Range range;
    range.scan = 4;
    range.start = UINT64_C(0x123456781234);
    range.size = 0x10000;
    bool success = range.append(5, range.start + range.size);
    CPPUNIT_ASSERT_EQUAL(false, success);
    CPPUNIT_ASSERT_EQUAL(Range::size_type(0x10000), range.size);
    CPPUNIT_ASSERT_EQUAL(Range::scan_type(4), range.scan);
    CPPUNIT_ASSERT_EQUAL(Range::index_type(UINT64_C(0x123456781234)), range.start);
}

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
    splats.resize(3);

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

    splats[2].push_back(makeSplat(18.0f, 33.0f, z, 1.0f));

    splats[2].push_back(makeSplat(25.0f, 45.0f, z, 4.0f));
}

void createSplats(std::vector<std::vector<Splat> > &splats,
                  boost::ptr_vector<StdVectorCollection<Splat> > &collections)
{
    createSplats(splats);
    collections.clear();
    collections.reserve(splats.size());
    for (std::size_t i = 0; i < splats.size(); i++)
    {
        collections.push_back(new StdVectorCollection<Splat>(splats[i]));
    }
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

    void testForEach();              ///< Tests @c forEach
    void testForEachRange();         ///< Tests @c forEachRange
    void testForEachRangeAll();      ///< Tests @c forEachRange where the ranges span everything
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
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestSplatSetBlob, TestSet::perBuild());


template<typename SetType>
void TestSplatSet<SetType>::setUp()
{
    createSplats(splatData, splats);
    set.reset(setFactory(splats, 2.5f, 2));
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
            float loVertex[3], hiVertex[3];
            Grid::difference_type loCell[3], hiCell[3];
            Grid::difference_type loBucket[3], hiBucket[3];
            for (unsigned int k = 0; k < 3; k++)
            {
                loWorld[k] = splat.position[k] - splat.radius;
                hiWorld[k] = splat.position[k] + splat.radius;
            }
            grid.worldToVertex(loWorld, loVertex);
            grid.worldToVertex(hiWorld, hiVertex);
            for (unsigned int k = 0; k < 3; k++)
            {
                loCell[k] = GridRoundDown::convert(loVertex[k]);
                hiCell[k] = GridRoundDown::convert(hiVertex[k]);
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
    ranges.push_back(Range(2, 0, 1));

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
        ranges.push_back(Range(i, 0, splatData[i].size()));

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
