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
#include <boost/foreach.hpp>
#include <boost/iostreams/device/null.hpp>
#include <boost/iostreams/stream.hpp>
#include <vector>
#include <utility>
#include <limits>
#include <memory>
#include <string>
#include <sstream>
#include <algorithm>
#include <iterator>
#include "../src/tr1_cstdint.h"
#include <boost/tr1/random.hpp>
#include "../src/splat.h"
#include "../src/grid.h"
#include "../src/splat_set.h"
#include "../src/logging.h"
#include "../src/statistics.h"
#include "../src/fast_ply.h"
#include "../src/allocator.h"
#include "test_splat_set.h"
#include "memory_reader.h"
#include "testutil.h"

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

/**
 * Like @ref createSplats, but creates data better suited to testing the splat
 * sets themselves. It includes
 * - non-finite splats
 * - an empty range at the start, in the middle, and at the end
 * - a range with a large number of splats, to test buffering in @ref SplatSet::FileSet.
 *
 * The splat coordinates are set based on the scan and offset, so are not
 * useful for bucketing tests.
 */
static void createSplats2(std::vector<std::vector<Splat> > &splats)
{
    splats.clear();
    splats.resize(10);
    float NaN = std::numeric_limits<float>::quiet_NaN();

    splats[2].push_back(makeSplat(2, 0, 0, NaN));
    splats[2].push_back(makeSplat(2, 1, 0, 1));
    splats[2].push_back(makeSplat(2, 2, 0, 2));

    splats[4].push_back(makeSplat(4, 0, NaN, 1));
    for (unsigned int i = 0; i < 50000; i++)
        splats[5].push_back(makeSplat(5, i, 0, 1));

    splats[6].push_back(makeSplat(6, 0, 0, 1));
    splats[6].push_back(makeSplat(6, NaN, 0, 1));
    splats[6].push_back(makeSplat(NaN, 2, 0, 1));
    splats[6].push_back(makeSplat(6, 3, 0, 100));

    splats[7].push_back(makeSplat(7, 0, 0, 1.5f));
    splats[7].push_back(makeSplat(7, 1, 0, NaN));
}

/// Tests for @ref SplatSet::detail::splatToBuckets.
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
    SplatSet::detail::splatToBuckets(s1, grid, 3, lower, upper);
    CPPUNIT_ASSERT_EQUAL(1, int(lower[0]));
    CPPUNIT_ASSERT_EQUAL(2, int(upper[0]));
    CPPUNIT_ASSERT_EQUAL(-1, int(lower[1]));
    CPPUNIT_ASSERT_EQUAL(0, int(upper[1]));
    CPPUNIT_ASSERT_EQUAL(16, int(lower[2]));
    CPPUNIT_ASSERT_EQUAL(16, int(upper[2]));

    Splat s2 = makeSplat(-1000.0f, -1000.0f, -1000.0f, 100.0f);
    SplatSet::detail::splatToBuckets(s2, grid, 3, lower, upper);
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

    CPPUNIT_ASSERT_THROW(SplatSet::detail::splatToBuckets(s, grid, 3, lower, upper), std::invalid_argument);
}

void TestSplatToBuckets::testZero()
{
    const float ref[3] = {10.0f, -50.0f, 40.0f};
    Grid grid(ref, 20.0f, -1, 5, 1, 100, 2, 50);
    // grid base is at (-10, -30, 80)
    boost::array<Grid::difference_type, 3> lower, upper;

    Splat s = makeSplat(115.0f, -31.0f, 1090.0f, 7.0f);

    CPPUNIT_ASSERT_THROW(SplatSet::detail::splatToBuckets(s, grid, 0, lower, upper), std::invalid_argument);
}

/// Tests for @ref SplatSet::detail::SplatToBuckets (the fast-path class)
class TestSplatToBucketsClass : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestSplatToBucketsClass);
    CPPUNIT_TEST(testSimple);
    CPPUNIT_TEST(testFloatRounding);
    CPPUNIT_TEST(testIntRounding);
    CPPUNIT_TEST_SUITE_END();

public:
    void testSimple();          ///< Test case that tests a bit of everything
    void testFloatRounding();   ///< Test the rounding on the float operations
    void testIntRounding();     ///< Test the rounding on the integer division
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestSplatToBucketsClass, TestSet::perBuild());

void TestSplatToBucketsClass::testSimple()
{
    SplatSet::detail::SplatToBuckets s2b(4.0f, 10);
    boost::array<Grid::difference_type, 3> lower, upper;
    s2b(makeSplat(-9.0f, 100.0f, -125.0f, 5.0f), lower, upper);
    MLSGPU_ASSERT_EQUAL(-1, lower[0]);
    MLSGPU_ASSERT_EQUAL(2, lower[1]);
    MLSGPU_ASSERT_EQUAL(-4, lower[2]);
    MLSGPU_ASSERT_EQUAL(-1, upper[0]);
    MLSGPU_ASSERT_EQUAL(2, upper[1]);
    MLSGPU_ASSERT_EQUAL(-3, upper[2]);
}

void TestSplatToBucketsClass::testFloatRounding()
{
    SplatSet::detail::SplatToBuckets s2b(8.0f, 1);
    boost::array<Grid::difference_type, 3> lower, upper;
    // Radius is big to give positive and negative values.
    // x rounds by more than 1/2, y by less than half, z by exactly half
    s2b(makeSplat(-1.0f, -13.0f, -12.0f, 32.0f), lower, upper);
    MLSGPU_ASSERT_EQUAL(-5, lower[0]);
    MLSGPU_ASSERT_EQUAL(-6, lower[1]);
    MLSGPU_ASSERT_EQUAL(-6, lower[2]);
    MLSGPU_ASSERT_EQUAL(3, upper[0]);
    MLSGPU_ASSERT_EQUAL(2, upper[1]);
    MLSGPU_ASSERT_EQUAL(2, upper[2]);
}

void TestSplatToBucketsClass::testIntRounding()
{
    SplatSet::detail::SplatToBuckets s2b(1.0f, 80);
    boost::array<Grid::difference_type, 3> lower, upper;
    // Radius is big to give positive and negative values.
    // x rounds by more than 1/2, y by less than half, z by exactly half
    s2b(makeSplat(-10.0f, -130.0f, -120.0f, 320.0f), lower, upper);
    MLSGPU_ASSERT_EQUAL(-5, lower[0]);
    MLSGPU_ASSERT_EQUAL(-6, lower[1]);
    MLSGPU_ASSERT_EQUAL(-6, lower[2]);
    MLSGPU_ASSERT_EQUAL(3, upper[0]);
    MLSGPU_ASSERT_EQUAL(2, upper[1]);
    MLSGPU_ASSERT_EQUAL(2, upper[2]);
}

/// Base class for testing models of @ref SplatSet::SetConcept.
template<typename SetType>
class TestSplatSet : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestSplatSet);
    CPPUNIT_TEST(testSplatStream);
    CPPUNIT_TEST(testBlobStream);
    CPPUNIT_TEST(testSplatStreamEmpty);
    CPPUNIT_TEST(testBlobStreamEmpty);
    CPPUNIT_TEST(testBlobStreamZeroBucket);
    CPPUNIT_TEST(testOtherGridSpacing);
    CPPUNIT_TEST(testOtherGridExtent);
    CPPUNIT_TEST(testOtherBucketSizeMultiple);
    CPPUNIT_TEST(testOtherBucketSize);
    CPPUNIT_TEST(testMaxSplats);
    CPPUNIT_TEST_SUITE_END_ABSTRACT();

protected:
    typedef SetType Set;

    std::vector<Splat> flatSplats; ///< Flattened @ref splatData with NaN's removed

    /**
     * Factory method that generates a set to run tests on. The returned set must
     * contain the data from @a splatData, and if it is backed by a container
     * of containers (e.g., @ref SplatSet::FileSet) then the structure should
     * match @a splatData.
     *
     * @a spacing and @a bucketSize are hints for future blob queries and should
     * be used when testing a @ref SplatSet::FastBlobSet. They may be ignored.
     *
     * Some subclasses do not allow an empty set. This method may thus return
     * @c NULL to indicate that the set could not be constructed and the test
     * should be skipped.
     */
    virtual Set *setFactory(const std::vector<std::vector<Splat> > &splatData,
                            float spacing, Grid::size_type bucketSize) = 0;

    /**
     * Fixture data generated by @ref createSplats2 that will normally be
     * passed to @ref setFactory.
     */
    std::vector<std::vector<Splat> > splatData;
    Grid grid;                     ///< Grid for hitting the fast path

private:
    /// Captures the parameters given to the function object
    struct Entry
    {
        typename Set::scan_type scan;
        typename Set::index_type first;
        typename Set::index_type last;
        boost::array<Grid::difference_type, 3> lower;
        boost::array<Grid::difference_type, 3> upper;
    };

    /**
     * Check that retrieved splats match what is expected.  The @a splatIds can
     * have any values provided that they're strictly increasing.
     */
    void validateSplats(const std::vector<Splat> &expected,
                        const std::vector<Splat> &actual,
                        const std::vector<splat_id> &ids);

    /// Check that retrieved blobs match what is expected
    void validateBlobs(const std::vector<Splat> &expected,
                       const std::vector<BlobInfo> &actual,
                       const Grid &grid, Grid::size_type bucketSize);

    /**
     * Constructs a set by passing @a factorySpacing and @a factorySize to
     * @ref setFactory, then validates its blob stream generated from @a grid
     * and @a bucketSize.
     */
    void testBlobStreamHelper(float factorySpacing, Grid::size_type factorySize,
                              const Grid &grid, Grid::size_type bucketSize);

public:
    /**
     * Generates fixture data.
     */
    virtual void setUp();
    /// Re-enable the warning log
    virtual void tearDown();

    void testSplatStream();              ///< Checks that the right splats are iterated
    void testBlobStream();               ///< Checks basic blob stream operation
    void testSplatStreamEmpty();         ///< Tests iteration over an empty splat stream
    void testBlobStreamEmpty();          ///< Tests iteration over an empty blob stream
    void testBlobStreamZeroBucket();     ///< Tests error check when trying to pass a zero @a bucketSize
    void testOtherGridSpacing();         ///< Tests blob stream where grid spacing does not match construction
    void testOtherGridExtent();          ///< Tests blob stream where the grid extents are changed
    void testOtherBucketSizeMultiple();  ///< Tests blob stream where the @a bucketSize is a multiple of that given during construction
    void testOtherBucketSize();          ///< Tests blob stream where the @a bucketSize is unrelated to that given during construction
    void testMaxSplats();                ///< Tests @ref SplatSet::SetConcept::maxSplats.
};

/// Tests for @ref SplatSet::SubsettableConcept.
template<typename SetType>
class TestSplatSubsettable : public TestSplatSet<SetType>
{
    CPPUNIT_TEST_SUB_SUITE(TestSplatSubsettable<SetType>, TestSplatSet<SetType>);
    CPPUNIT_TEST(testSplatStreamSeek);
    CPPUNIT_TEST(testSplatStreamSeekZeroRanges);
    CPPUNIT_TEST(testSplatStreamSeekEmptyRange);
    CPPUNIT_TEST(testSplatStreamSeekNegativeRange);
    CPPUNIT_TEST_SUITE_END_ABSTRACT();

protected:
    /// Tests that iteration over a list of ranges gives the expected results
    template<typename RangeIterator>
    void testSplatStreamSeekHelper(RangeIterator first, RangeIterator last);

public:
    typedef SetType Set;

    void testSplatStreamSeek();               ///< Tests basic operation
    void testSplatStreamSeekZeroRanges();     ///< Tests an empty list of ranges
    void testSplatStreamSeekEmptyRange();     ///< Tests iteration over an empty range
    void testSplatStreamSeekNegativeRange();  ///< Tests handling of first > last
};

/// Tests for @ref SplatSet::FileSet
class TestFileSet : public TestSplatSubsettable<FileSet>
{
    CPPUNIT_TEST_SUB_SUITE(TestFileSet, TestSplatSubsettable<FileSet>);
    CPPUNIT_TEST_SUITE_END();

private:
    /// Backing store for PLY "files"
    std::vector<std::string> store;

protected:
    virtual Set *setFactory(const std::vector<std::vector<Splat> > &splatData,
                            float spacing, Grid::size_type bucketSize);
public:
    /**
     * Adds all splats in @a splatData to the set. Each element of @a splatData is
     * converted to PLY format and appended as a new @ref FastPly::ReaderBase to @a set.
     * The converted data are stored in @a store, which must remain live and unmodified
     * for as long as @a set is live.
     */
    static void populate(FileSet &set, const std::vector<std::vector<Splat> > &splatData,
                         vector<string> &store);
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestFileSet, TestSet::perBuild());

/// Tests for @ref SplatSet::SequenceSet
class TestSequenceSet : public TestSplatSubsettable<SequenceSet<const Splat *> >
{
    CPPUNIT_TEST_SUB_SUITE(TestSequenceSet, TestSplatSubsettable<SequenceSet<const Splat *> >);
    CPPUNIT_TEST_SUITE_END();
private:
    std::vector<Splat> store; ///< Backing data for the returned sets
protected:
    virtual Set *setFactory(const std::vector<std::vector<Splat> > &splatData,
                            float spacing, Grid::size_type bucketSize);
public:
    /// Adds all splats in @a splatData to the store and creates the set.
    static void populate(SequenceSet<const Splat *> &set, const std::vector<std::vector<Splat> > &splatData,
                         std::vector<Splat> &store);
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestSequenceSet, TestSet::perBuild());

/// Tests for @ref SplatSet::FastBlobSet.
template<typename BaseType>
class TestFastBlobSet : public TestSplatSubsettable<FastBlobSet<BaseType, Statistics::Container::vector<BlobData> > >
{
    typedef TestSplatSubsettable<FastBlobSet<BaseType, Statistics::Container::vector<BlobData> > > BaseFixture;
    CPPUNIT_TEST_SUB_SUITE(TestFastBlobSet<BaseType>, BaseFixture);
    CPPUNIT_TEST(testBoundingGrid);
    CPPUNIT_TEST(testAddBlob);
    CPPUNIT_TEST_SUITE_END_ABSTRACT();
public:
    typedef typename BaseFixture::Set Set;

    void testBoundingGrid();         ///< Tests that the extracted bounding box is correct
    void testAddBlob();              ///< Tests the encoding of blobs
};

/// Tests for @ref SplatSet::FastBlobSet<SplatSet::FileSet>.
class TestFastFileSet : public TestFastBlobSet<FileSet>
{
    typedef TestFastBlobSet<FileSet> BaseFixture;
    CPPUNIT_TEST_SUB_SUITE(TestFastFileSet, BaseFixture);
    CPPUNIT_TEST(testEmpty);
    CPPUNIT_TEST(testProgress);
    CPPUNIT_TEST_SUITE_END();

private:
    std::vector<std::string> store;

protected:
    virtual Set *setFactory(const std::vector<std::vector<Splat> > &splatData,
                            float spacing, Grid::size_type bucketSize);
public:
    void testEmpty();            ///< Test error checking for an empty set
    void testProgress();         ///< Run with a progress stream (does not check output)
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestFastFileSet, TestSet::perBuild());

/// Tests for @ref SplatSet::FastBlobSet<SplatSet::SequenceSet<const Splat *> >.
class TestFastSequenceSet : public TestFastBlobSet<SequenceSet<const Splat *> >
{
    typedef TestFastBlobSet<SequenceSet<const Splat *> > BaseFixture;
    CPPUNIT_TEST_SUB_SUITE(TestFastSequenceSet, BaseFixture);
    CPPUNIT_TEST_SUITE_END();
private:
    std::vector<Splat> store; ///< Backing data for the returned sets
protected:
    virtual Set *setFactory(const std::vector<std::vector<Splat> > &splatData,
                            float spacing, Grid::size_type bucketSize);
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestFastSequenceSet, TestSet::perBuild());

/// Tests for @ref SplatSet::merge
class TestMerge : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestMerge);
    CPPUNIT_TEST(testMergeEmpty);
    CPPUNIT_TEST(testMergeTail);
    CPPUNIT_TEST(testMergeGeneral);
    CPPUNIT_TEST_SUITE_END();
protected:
    void testMergeHelper(
        std::size_t numA,
        const splat_id rangesA[][2],
        std::size_t numB,
        const splat_id rangesB[][2],
        std::size_t numExpected,
        const splat_id rangesExpected[][2]);
public:
    void testMergeEmpty();     ///< Test @ref SplatSet::merge with two empty subsets
    void testMergeTail();      ///< Test @ref SplatSet::merge with tail elements in one set
    void testMergeGeneral();   ///< Miscellaneous tests for @ref SplatSet::merge.
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestMerge, TestSet::perBuild());

/// Tests for @ref SplatSet::Subset
class TestSubset : public TestSplatSet<Subset<FastBlobSet<SequenceSet<const Splat *>, std::vector<BlobData> > > >
{
    typedef FastBlobSet<SequenceSet<const Splat *>, std::vector<BlobData> > Super;
    typedef TestSplatSet<Subset<FastBlobSet<SequenceSet<const Splat *>, std::vector<BlobData> > > > BaseFixture;
    CPPUNIT_TEST_SUB_SUITE(TestSubset, BaseFixture);
    CPPUNIT_TEST_SUITE_END();
private:
    Super super;
    std::vector<Splat> store; ///< Backing data for the returned sets
protected:
    virtual Set *setFactory(const std::vector<std::vector<Splat> > &splatData,
                            float spacing, Grid::size_type bucketSize);
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestSubset, TestSet::perBuild());

template<typename SetType>
void TestSplatSet<SetType>::setUp()
{
    CppUnit::TestFixture::setUp();
    createSplats2(splatData);

    flatSplats.clear();
    for (std::size_t i = 0; i < splatData.size(); i++)
        for (std::size_t j = 0; j < splatData[i].size(); j++)
            if (splatData[i][j].isFinite())
                flatSplats.push_back(splatData[i][j]);

    const float ref[3] = {0.0f, 0.0f, 0.0f};
    grid = Grid(ref, 2.5f, 2, 20, -30, 25, 0, 30);
}

template<typename SetType>
void TestSplatSet<SetType>::tearDown()
{
    splatData.clear();
    flatSplats.clear();
    CppUnit::TestFixture::tearDown();
}

template<typename SetType>
void TestSplatSet<SetType>::validateSplats(
    const std::vector<Splat> &expected,
    const std::vector<Splat> &actual,
    const std::vector<splat_id> &ids)
{
    CPPUNIT_ASSERT_EQUAL(expected.size(), actual.size());
    CPPUNIT_ASSERT_EQUAL(expected.size(), ids.size());
    for (std::size_t i = 0; i < expected.size(); i++)
    {
        CPPUNIT_ASSERT(actual[i].isFinite()); // avoids false NaN comparisons later
        CPPUNIT_ASSERT_EQUAL(expected[i].position[0], actual[i].position[0]);
        CPPUNIT_ASSERT_EQUAL(expected[i].position[1], actual[i].position[1]);
        CPPUNIT_ASSERT_EQUAL(expected[i].position[2], actual[i].position[2]);
        CPPUNIT_ASSERT_EQUAL(expected[i].radius, actual[i].radius);
    }

    for (std::size_t i = 1; i < ids.size(); i++)
    {
        CPPUNIT_ASSERT(ids[i - 1] < ids[i]);
    }
}

template<typename SetType>
void TestSplatSet<SetType>::validateBlobs(
    const std::vector<Splat> &expected,
    const std::vector<BlobInfo> &actual,
    const Grid &grid,
    Grid::size_type bucketSize)
{
    std::size_t nextSplat = 0;
    for (std::size_t i = 0; i < actual.size(); i++)
    {
        CPPUNIT_ASSERT(i == 0 || actual[i].firstSplat >= actual[i - 1].lastSplat);
        const BlobInfo &cur = actual[i];
        CPPUNIT_ASSERT(cur.lastSplat > cur.firstSplat);
        splat_id numSplats = cur.lastSplat - cur.firstSplat;
        CPPUNIT_ASSERT(nextSplat + numSplats <= expected.size());
        for (std::size_t j = 0; j < numSplats; j++)
        {
            boost::array<Grid::difference_type, 3> lower, upper;
            SplatSet::detail::splatToBuckets(
                expected[nextSplat + j], grid, bucketSize, lower, upper);
            for (unsigned int k = 0; k < 3; k++)
            {
                CPPUNIT_ASSERT_EQUAL(lower[k], cur.lower[k]);
                CPPUNIT_ASSERT_EQUAL(upper[k], cur.upper[k]);
            }
        }
        nextSplat += numSplats;
    }
    CPPUNIT_ASSERT_EQUAL(expected.size(), nextSplat);
}

template<typename SetType>
void TestSplatSet<SetType>::testSplatStream()
{
    boost::scoped_ptr<Set> set(setFactory(splatData, 2.5f, 5));
    if (set.get())
    {
        boost::scoped_ptr<SplatStream> stream(set->makeSplatStream());
        std::vector<Splat> actual;
        std::vector<splat_id> ids;
        const std::size_t count = 5; // there are files bigger, smaller and exactly this size
        Splat buffer[count];
        splat_id bufferIds[count];
        while (true)
        {
            std::size_t n = stream->read(buffer, bufferIds, count);
            if (n == 0)
                break;
            for (std::size_t i = 0; i < n; i++)
            {
                actual.push_back(buffer[i]);
                ids.push_back(bufferIds[i]);
            }
        }
        validateSplats(flatSplats, actual, ids);
        MLSGPU_ASSERT_EQUAL(0, stream->read(buffer, bufferIds, count));
    }
    else
        CPPUNIT_ASSERT(flatSplats.empty()); // some classes don't allow empty sets
}

template<typename SetType>
void TestSplatSet<SetType>::testBlobStreamHelper(
    float factorySpacing, Grid::size_type factorySize,
    const Grid &grid, Grid::size_type bucketSize)
{
    boost::scoped_ptr<Set> set(setFactory(splatData, factorySpacing, factorySize));
    if (set.get())
    {
        boost::scoped_ptr<BlobStream> stream(set->makeBlobStream(grid, bucketSize));
        std::vector<BlobInfo> actual;
        while (!stream->empty())
        {
            actual.push_back(**stream);
            ++*stream;
        }
        validateBlobs(flatSplats, actual, grid, bucketSize);
    }
    else
        CPPUNIT_ASSERT(flatSplats.empty()); // some classes don't allow empty sets
}

template<typename SetType>
void TestSplatSet<SetType>::testBlobStream()
{
    testBlobStreamHelper(grid.getSpacing(), 5, grid, 5);
}

template<typename SetType>
void TestSplatSet<SetType>::testSplatStreamEmpty()
{
    splatData.clear();
    flatSplats.clear();
    testSplatStream();
}

template<typename SetType>
void TestSplatSet<SetType>::testBlobStreamEmpty()
{
    splatData.clear();
    flatSplats.clear();
    testBlobStreamHelper(grid.getSpacing(), 5, grid, 5);
}

template<typename SetType>
void TestSplatSet<SetType>::testBlobStreamZeroBucket()
{
    boost::scoped_ptr<Set> set(setFactory(splatData, 2.5f, 5));
    if (set.get())
    {
        boost::scoped_ptr<BlobStream> stream;
        CPPUNIT_ASSERT_THROW(stream.reset(set->makeBlobStream(grid, 0)), std::invalid_argument);
    }
}

template<typename SetType>
void TestSplatSet<SetType>::testOtherGridSpacing()
{
    Grid otherGrid = grid;
    otherGrid.setSpacing(3.0f);
    testBlobStreamHelper(grid.getSpacing(), 5, otherGrid, 5);
}

template<typename SetType>
void TestSplatSet<SetType>::testOtherGridExtent()
{
    Grid otherGrid = grid;
    otherGrid.setExtent(1, -50, 50);
    testBlobStreamHelper(grid.getSpacing(), 5, otherGrid, 5);
}

template<typename SetType>
void TestSplatSet<SetType>::testOtherBucketSizeMultiple()
{
    testBlobStreamHelper(grid.getSpacing(), 5, grid, 10);
}

template<typename SetType>
void TestSplatSet<SetType>::testOtherBucketSize()
{
    testBlobStreamHelper(grid.getSpacing(), 5, grid, 4);
}

template<typename SetType>
void TestSplatSet<SetType>::testMaxSplats()
{
    const unsigned int bucketSize = 5;
    boost::scoped_ptr<Set> set(setFactory(splatData, 2.5f, bucketSize));
    CPPUNIT_ASSERT(flatSplats.size() <= set->maxSplats());
    // No upper bound in the spec, but if it's bigger than this then it's an
    // unreasonable overestimate
    std::size_t total = 0;
    for (std::size_t i = 0; i < splatData.size(); i++)
        total += splatData[i].size();
    CPPUNIT_ASSERT(set->maxSplats() <= total);
}


template<typename SetType>
template<typename RangeIterator>
void TestSplatSubsettable<SetType>::testSplatStreamSeekHelper(RangeIterator firstRange, RangeIterator lastRange)
{
    vector<Splat> expected, actual;
    vector<splat_id> expectedIds, actualIds;

    const unsigned int bucketSize = 5;
    boost::scoped_ptr<Set> set(this->setFactory(this->splatData, this->grid.getSpacing(), bucketSize));

    for (RangeIterator curRange = firstRange; curRange != lastRange; ++curRange)
    {
        boost::scoped_ptr<SplatStream> splatStream(set->makeSplatStream());
        Splat splat;
        splat_id id;
        while (splatStream->read(&splat, &id, 1) != 0)
        {
            if (id >= curRange->first && id < curRange->second)
            {
                expected.push_back(splat);
                expectedIds.push_back(id);
            }
        }
    }

    {
        boost::scoped_ptr<SplatStream> splatStream(set->makeSplatStream(firstRange, lastRange));
        const std::size_t count = 3;
        while (true)
        {
            Splat buffer[count];
            splat_id bufferIds[count];
            std::size_t n = splatStream->read(buffer, bufferIds, count);
            if (n == 0)
                break;
            for (std::size_t i = 0; i < n; i++)
            {
                actual.push_back(buffer[i]);
                actualIds.push_back(bufferIds[i]);
            }
        }
        MLSGPU_ASSERT_EQUAL(0, splatStream->read(NULL, NULL, 1));
    }

    CPPUNIT_ASSERT_EQUAL(expected.size(), actual.size());
    for (std::size_t i = 0; i < expected.size(); i++)
    {
        CPPUNIT_ASSERT_EQUAL(expectedIds[i], actualIds[i]);
        CPPUNIT_ASSERT_EQUAL(expected[i].position[0], actual[i].position[0]);
        CPPUNIT_ASSERT_EQUAL(expected[i].position[1], actual[i].position[1]);
        CPPUNIT_ASSERT_EQUAL(expected[i].position[2], actual[i].position[2]);
        CPPUNIT_ASSERT_EQUAL(expected[i].radius, actual[i].radius);
    }
}

template<typename SetType>
void TestSplatSubsettable<SetType>::testSplatStreamSeek()
{
    typedef std::pair<splat_id, splat_id> Range;
    std::vector<Range> ranges;

    ranges.push_back(Range(0, 1000000));
    ranges.push_back(Range(3, splat_id(3) << 40));
    ranges.push_back(Range(2, (splat_id(3) << 40) - 1));
    ranges.push_back(Range((splat_id(1) << 40) + 100, (splat_id(6) << 40) - 1));
    ranges.push_back(Range((splat_id(5) << 40), (splat_id(5) << 40) + 20000));
    ranges.push_back(Range((splat_id(4) << 40), (splat_id(50) << 40) - 1));
    testSplatStreamSeekHelper(ranges.begin(), ranges.end());
}

template<typename SetType>
void TestSplatSubsettable<SetType>::testSplatStreamSeekZeroRanges()
{
    typedef std::pair<splat_id, splat_id> Range;
    std::vector<Range> ranges;

    testSplatStreamSeekHelper(ranges.begin(), ranges.end());
}

template<typename SetType>
void TestSplatSubsettable<SetType>::testSplatStreamSeekEmptyRange()
{
    typedef std::pair<splat_id, splat_id> Range;
    std::vector<Range> ranges;

    ranges.push_back(Range(0, 0));
    ranges.push_back(Range(3, 3));
    ranges.push_back(Range(1000000000, 1000000000));
    testSplatStreamSeekHelper(ranges.begin(), ranges.end());
}

template<typename SetType>
void TestSplatSubsettable<SetType>::testSplatStreamSeekNegativeRange()
{
    typedef std::pair<splat_id, splat_id> Range;
    std::vector<Range> ranges;

    ranges.push_back(Range(1, 0));
    ranges.push_back(Range(splat_id(1) << 33, 1));
    testSplatStreamSeekHelper(ranges.begin(), ranges.end());
}


void TestFileSet::populate(FileSet &set, const std::vector<std::vector<Splat> > &splatData, vector<string> &store)
{
    store.clear();
    store.reserve(splatData.size());
    BOOST_FOREACH(const std::vector<Splat> &splats, splatData)
    {
        std::ostringstream data;
        data <<
            "ply\n"
            "format binary_little_endian 1.0\n"
            "element vertex " << splats.size() << "\n"
            "property float32 x\n"
            "property float32 y\n"
            "property float32 z\n"
            "property float32 nx\n"
            "property float32 ny\n"
            "property float32 nz\n"
            "property float32 radius\n"
            "end_header\n";
        BOOST_FOREACH(const Splat &splat, splats)
        {
            data.write((const char *) splat.position, 3 * sizeof(float));
            data.write((const char *) splat.normal, 3 * sizeof(float));
            data.write((const char *) &splat.radius, sizeof(float));
        }
        store.push_back(data.str());
        set.addFile(new MemoryReader(store.back().data(), store.back().size(), 1.0f, std::numeric_limits<float>::infinity()));
    }
}

FileSet *TestFileSet::setFactory(
    const std::vector<std::vector<Splat> > &splatData,
    float spacing, Grid::size_type bucketSize)
{
    (void) spacing;
    (void) bucketSize;
    std::auto_ptr<Set> set(new Set);
    populate(*set, splatData, store);
    set->setBufferSize(16384);
    return set.release();
}

void TestSequenceSet::populate(
    SequenceSet<const Splat *> &set,
    const std::vector<std::vector<Splat> > &splatData,
    std::vector<Splat> &store)
{
    for (std::size_t i = 0; i < splatData.size(); i++)
        store.insert(store.end(), splatData[i].begin(), splatData[i].end());
    set.reset(&store[0], &store[0] + store.size());
}

SequenceSet<const Splat *> *TestSequenceSet::setFactory(
    const std::vector<std::vector<Splat> > &splatData,
    float spacing, Grid::size_type bucketSize)
{
    (void) spacing;
    (void) bucketSize;

    std::auto_ptr<Set> set(new Set);
    populate(*set, splatData, store);
    return set.release();
}

template<typename BaseType>
void TestFastBlobSet<BaseType>::testBoundingGrid()
{
    const unsigned int bucketSize = 5;
    boost::scoped_ptr<Set> set(this->setFactory(this->splatData, 2.5f, bucketSize));
    Grid bbox = set->getBoundingGrid();
    CPPUNIT_ASSERT_EQUAL(2.5f, bbox.getSpacing());
    CPPUNIT_ASSERT_EQUAL(0.0f, bbox.getReference()[0]);
    CPPUNIT_ASSERT_EQUAL(0.0f, bbox.getReference()[1]);
    CPPUNIT_ASSERT_EQUAL(0.0f, bbox.getReference()[2]);
    // Actual bounding box is (-94, -97, -100) to (106, 50000, 100)
    // Divided by 2.5:        (-37.6, -38.8, -40) to (42.4, 20000, 40)
    // Rounded:               (-40, -40, -40) to (43, 20000, 40)
    CPPUNIT_ASSERT_EQUAL(-40, bbox.getExtent(0).first);
    CPPUNIT_ASSERT_EQUAL(-40, bbox.getExtent(1).first);
    CPPUNIT_ASSERT_EQUAL(-40, bbox.getExtent(2).first);
    CPPUNIT_ASSERT_EQUAL(43, bbox.getExtent(0).second);
    CPPUNIT_ASSERT_EQUAL(20000, bbox.getExtent(1).second);
    CPPUNIT_ASSERT_EQUAL(40, bbox.getExtent(2).second);
}

template<typename BaseType>
void TestFastBlobSet<BaseType>::testAddBlob()
{
    Statistics::Container::vector<BlobData> blobData("mem.test.blobData");
    BlobInfo prevBlob, curBlob;

    // Full encoding
    curBlob.firstSplat = UINT64_C(0x123456781234);
    curBlob.lastSplat = UINT64_C(0x234567801234);
    curBlob.lower[0] = -128;
    curBlob.lower[1] = -64;
    curBlob.lower[2] = -32;
    curBlob.upper[0] = -1;
    curBlob.upper[1] = 0;
    curBlob.upper[2] = 1023;

    Set::addBlob(blobData, prevBlob, curBlob);
    CPPUNIT_ASSERT_EQUAL(10, int(blobData.size()));
    CPPUNIT_ASSERT_EQUAL(UINT32_C(0x1234), blobData[0]);     // first hi
    CPPUNIT_ASSERT_EQUAL(UINT32_C(0x56781234), blobData[1]); // first lo
    CPPUNIT_ASSERT_EQUAL(UINT32_C(0x2345), blobData[2]);     // last hi
    CPPUNIT_ASSERT_EQUAL(UINT32_C(0x67801234), blobData[3]); // last lo
    CPPUNIT_ASSERT_EQUAL(UINT32_C(0xFFFFFF80), blobData[4]); // lower[0]
    CPPUNIT_ASSERT_EQUAL(UINT32_C(0xFFFFFFFF), blobData[5]); // upper[0]
    CPPUNIT_ASSERT_EQUAL(UINT32_C(0xFFFFFFC0), blobData[6]); // lower[1]
    CPPUNIT_ASSERT_EQUAL(UINT32_C(0),          blobData[7]); // upper[1]
    CPPUNIT_ASSERT_EQUAL(UINT32_C(0xFFFFFFE0), blobData[8]); // lower[2]
    CPPUNIT_ASSERT_EQUAL(UINT32_C(1023),       blobData[9]); // upper[2]

    // Differential encoding
    prevBlob = curBlob;
    curBlob.firstSplat = UINT64_C(0x234567801234);
    curBlob.lastSplat = curBlob.firstSplat + (1 << 19) - 1;
    curBlob.lower[0] = -5;
    curBlob.upper[0] = -4;
    curBlob.lower[1] = 3;
    curBlob.upper[1] = 3;
    curBlob.lower[2] = 1022;
    curBlob.upper[2] = 1023;
    Set::addBlob(blobData, prevBlob, curBlob);
    CPPUNIT_ASSERT_EQUAL(11, int(blobData.size()));
    // 1 1111111111111111111 1 111 0 011 1 100
    CPPUNIT_ASSERT_EQUAL(UINT32_C(0xFFFFFF3C), blobData[10]);

    // Make sure the decoding works
    Set set("mem.test.blobData");
    set.blobData = blobData;
    set.internalBucketSize = 1;

    BlobInfo blob;
    boost::scoped_ptr<BlobStream> stream(set.makeBlobStream(set.boundingGrid, set.internalBucketSize));
    CPPUNIT_ASSERT(!stream->empty());
    blob = **stream;
    CPPUNIT_ASSERT(blob == prevBlob);
    ++*stream;
    CPPUNIT_ASSERT(!stream->empty());
    blob = **stream;
    CPPUNIT_ASSERT(blob == curBlob);
    ++*stream;
    CPPUNIT_ASSERT(stream->empty());
}

FastBlobSet<FileSet, Statistics::Container::vector<BlobData> > *TestFastFileSet::setFactory(
    const std::vector<std::vector<Splat> > &splatData,
    float spacing, Grid::size_type bucketSize)
{
    if (splatData.empty())
        return NULL; // otherwise computeBlobs will throw
    std::auto_ptr<Set> set(new Set("mem.test.blobData"));
    TestFileSet::populate(*set, splatData, store);
    set->computeBlobs(spacing, bucketSize, NULL, false);
    return set.release();
}

void TestFastFileSet::testEmpty()
{
    boost::scoped_ptr<Set> set(new Set("mem.test.blobData"));
    CPPUNIT_ASSERT_THROW(set->computeBlobs(2.5f, 5, NULL, false), std::runtime_error);
}

void TestFastFileSet::testProgress()
{
    boost::scoped_ptr<Set> set(new Set("mem.test.blobData"));
    TestFileSet::populate(*set, splatData, store);

    boost::iostreams::null_sink nullSink;
    boost::iostreams::stream<boost::iostreams::null_sink> nullStream(nullSink);
    set->computeBlobs(2.5f, 5, &nullStream, false);
}

FastBlobSet<SequenceSet<const Splat *>, Statistics::Container::vector<BlobData> > *TestFastSequenceSet::setFactory(
    const std::vector<std::vector<Splat> > &splatData,
    float spacing, Grid::size_type bucketSize)
{
    if (splatData.empty())
        return NULL;
    std::auto_ptr<Set> set(new Set("mem.test.blobData"));
    TestSequenceSet::populate(*set, splatData, store);
    set->computeBlobs(spacing, bucketSize, NULL, false);
    return set.release();
}

Subset<FastBlobSet<SequenceSet<const Splat *>, std::vector<BlobData> > > *
TestSubset::setFactory(const std::vector<std::vector<Splat> > &splatData,
                       float spacing, Grid::size_type bucketSize)
{
    if (splatData.empty())
        return NULL;

    std::tr1::mt19937 engine;
    std::tr1::bernoulli_distribution dist(0.75);
    std::tr1::variate_generator<std::tr1::mt19937 &, std::tr1::bernoulli_distribution> gen(engine, dist);

    TestSequenceSet::populate(super, splatData, store);
    super.computeBlobs(spacing, bucketSize, NULL, false);
    std::auto_ptr<Set> set(new Set(super));

    // Select a random subset of the blobs in the superset
    vector<Splat> flatSubset;
    unsigned int offset = 0;
    boost::scoped_ptr<BlobStream> superBlobs(super.makeBlobStream(super.getBoundingGrid(), bucketSize));
    while (!superBlobs->empty())
    {
        const BlobInfo blob = **superBlobs;
        splat_id numSplats = blob.lastSplat - blob.firstSplat;
        if (gen())
        {
            std::copy(flatSplats.begin() + offset, flatSplats.begin() + offset + numSplats,
                      std::back_inserter(flatSubset));
            set->addBlob(blob);
        }
        offset += numSplats;
        ++*superBlobs;
    }
    set->flush();
    CPPUNIT_ASSERT_EQUAL((unsigned int) flatSplats.size(), offset);
    flatSplats.swap(flatSubset);
    return set.release();
}

void TestMerge::testMergeHelper(
    std::size_t numA,
    const splat_id rangesA[][2],
    std::size_t numB,
    const splat_id rangesB[][2],
    std::size_t numExpected,
    const splat_id rangesExpected[][2])
{
    SubsetBase a, b;
    for (std::size_t i = 0; i < numA; i++)
        a.addRange(rangesA[i][0], rangesA[i][1]);
    for (std::size_t i = 0; i < numB; i++)
        b.addRange(rangesB[i][0], rangesB[i][1]);
    a.flush();
    b.flush();

    SubsetBase ans;
    SplatSet::merge(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(ans));
    ans.flush();
    std::size_t pos = 0;
    for (SubsetBase::const_iterator i = ans.begin(); i != ans.end(); ++i)
    {
        CPPUNIT_ASSERT(pos < numExpected);
        CPPUNIT_ASSERT_EQUAL(rangesExpected[pos][0], i->first);
        CPPUNIT_ASSERT_EQUAL(rangesExpected[pos][1], i->second);
        pos++;
    }
    CPPUNIT_ASSERT_EQUAL(pos, numExpected);
}

void TestMerge::testMergeEmpty()
{
    testMergeHelper(0, NULL, 0, NULL, 0, NULL);
}

void TestMerge::testMergeTail()
{
    const splat_id rangesA[][2] =
    {
        { 1, 3 },
        { 20, 22 },
        { 25, 30 }
    };
    const splat_id rangesB[][2] =
    {
        { 3, 5 },
        { 7, 10 }
    };
    const splat_id rangesExpected[][2] =
    {
        { 1, 5 },
        { 7, 10 },
        { 20, 22 },
        { 25, 30 }
    };
    testMergeHelper(3, rangesA, 2, rangesB, 4, rangesExpected);
    testMergeHelper(2, rangesB, 3, rangesA, 4, rangesExpected);
}

void TestMerge::testMergeGeneral()
{
    const splat_id rangesA[][2] =
    {
        { 1, 10 },
        { 20, 30 },
        { 40, 50 }
    };
    const splat_id rangesB[][2] =
    {
        { 3, 8 },
        { 9, 15 },
        { 18, 22 },
        { 30, 40 }
    };
    const splat_id rangesExpected[][2] =
    {
        { 1, 15 },
        { 18, 50 }
    };
    testMergeHelper(3, rangesA, 4, rangesB, 2, rangesExpected);
}
