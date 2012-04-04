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
#include <tr1/cstdint>
#include <boost/tr1/random.hpp>
#include "../src/splat.h"
#include "../src/grid.h"
#include "../src/splat_set.h"
#include "../src/logging.h"
#include "../src/statistics.h"
#include "../src/fast_ply.h"
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

/// Tests for @ref SplatSet::internal::splatToBuckets.
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
     * Generates fixture data, and disables the warning log to prevent
     * warnings about NaN values (which are expected in the test).
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
    CPPUNIT_TEST(testSplatStreamReset);
    CPPUNIT_TEST(testSplatStreamResetEmptyRange);
    CPPUNIT_TEST(testSplatStreamResetNegativeRange);
    CPPUNIT_TEST_SUITE_END_ABSTRACT();

protected:
    /// Resets a splat stream to a given range and tests iteration over it
    void testSplatStreamResetHelper(splat_id first, splat_id last);

public:
    typedef SetType Set;

    void testSplatStreamReset();               ///< Tests basic reset operation
    void testSplatStreamResetEmptyRange();     ///< Tests iteration over an empty range
    void testSplatStreamResetNegativeRange();  ///< Tests error check for a negative range
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
     * converted to PLY format and appended as a new @ref FastPly::Reader to @a set.
     * The converted data are stored in @a store, which must remain live and unmodified
     * for as long as @a set is live.
     */
    static void populate(FileSet &set, const std::vector<std::vector<Splat> > &splatData,
                         vector<string> &store);
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestFileSet, TestSet::perBuild());

/// Tests for @ref SplatSet::VectorSet
class TestVectorSet : public TestSplatSubsettable<VectorSet>
{
    CPPUNIT_TEST_SUB_SUITE(TestVectorSet, TestSplatSubsettable<VectorSet>);
    CPPUNIT_TEST_SUITE_END();
protected:
    virtual Set *setFactory(const std::vector<std::vector<Splat> > &splatData,
                            float spacing, Grid::size_type bucketSize);
public:
    /// Adds all splats in @a splatData to the vector
    static void populate(VectorSet &set, const std::vector<std::vector<Splat> > &splatData);
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestVectorSet, TestSet::perBuild());

/// Tests for @ref SplatSet::FastBlobSet.
template<typename BaseType>
class TestFastBlobSet : public TestSplatSubsettable<FastBlobSet<BaseType, std::vector<BlobData> > >
{
    typedef TestSplatSubsettable<FastBlobSet<BaseType, std::vector<BlobData> > > BaseFixture;
    CPPUNIT_TEST_SUB_SUITE(TestFastBlobSet<BaseType>, BaseFixture);
    CPPUNIT_TEST(testBoundingGrid);
    CPPUNIT_TEST_SUITE_END_ABSTRACT();
public:
    typedef typename BaseFixture::Set Set;

    void testBoundingGrid();         ///< Tests that the extracted bounding box is correct
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

/// Tests for @ref SplatSet::FastBlobSet<SplatSet::VectorSet>.
class TestFastVectorSet : public TestFastBlobSet<VectorSet>
{
    typedef TestFastBlobSet<VectorSet> BaseFixture;
    CPPUNIT_TEST_SUB_SUITE(TestFastVectorSet, BaseFixture);
    CPPUNIT_TEST_SUITE_END();
protected:
    virtual Set *setFactory(const std::vector<std::vector<Splat> > &splatData,
                            float spacing, Grid::size_type bucketSize);
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestFastVectorSet, TestSet::perBuild());

/// Tests for @ref SplatSet::Subset
class TestSubset : public TestSplatSet<Subset<FastBlobSet<VectorSet, std::vector<BlobData> > > >
{
    typedef FastBlobSet<VectorSet, std::vector<BlobData> > Super;
    typedef TestSplatSet<Subset<FastBlobSet<VectorSet, std::vector<BlobData> > > > BaseFixture;
    CPPUNIT_TEST_SUB_SUITE(TestSubset, BaseFixture);
    CPPUNIT_TEST_SUITE_END();
private:
    Super super;
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

    /* The NaN test normally causes a warning to be printed about
     * invalid splats, but in this case it's intentional, so we
     * suppress the warning.
     *
     * TODO: move up to FastBlobSet test.
     */
    Log::log.setLevel(Log::error);
}

template<typename SetType>
void TestSplatSet<SetType>::tearDown()
{
    splatData.clear();
    flatSplats.clear();
    // Restore the default log level
    Log::log.setLevel(Log::warn);
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
            SplatSet::internal::splatToBuckets(
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
        while (!stream->empty())
        {
            actual.push_back(**stream);
            ids.push_back(stream->currentId());
            ++*stream;
        }
        validateSplats(flatSplats, actual, ids);
        CPPUNIT_ASSERT_THROW(**stream, std::out_of_range);
        CPPUNIT_ASSERT_THROW(++*stream, std::out_of_range);
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
void TestSplatSubsettable<SetType>::testSplatStreamResetHelper(splat_id first, splat_id last)
{
    vector<Splat> expected, actual;
    vector<splat_id> expectedIds, actualIds;

    const unsigned int bucketSize = 5;
    boost::scoped_ptr<Set> set(this->setFactory(this->splatData, this->grid.getSpacing(), bucketSize));

    {
        boost::scoped_ptr<SplatStream> splatStream(set->makeSplatStream());
        while (!splatStream->empty())
        {
            splat_id id = splatStream->currentId();
            if (id >= first && id < last)
            {
                expected.push_back(**splatStream);
                expectedIds.push_back(id);
            }
            ++*splatStream;
        }
    }

    {
        boost::scoped_ptr<SplatStreamReset> splatStreamReset(set->makeSplatStreamReset());
        splatStreamReset->reset(first, last);
        while (!splatStreamReset->empty())
        {
            actual.push_back(**splatStreamReset);
            actualIds.push_back(splatStreamReset->currentId());
            ++*splatStreamReset;
        }
        CPPUNIT_ASSERT_THROW(**splatStreamReset, std::out_of_range);
        CPPUNIT_ASSERT_THROW(++*splatStreamReset, std::out_of_range);
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
void TestSplatSubsettable<SetType>::testSplatStreamReset()
{
    testSplatStreamResetHelper(0, 1000000);
    testSplatStreamResetHelper(3, splat_id(3) << 40);
    testSplatStreamResetHelper(2, (splat_id(3) << 40) - 1);
    testSplatStreamResetHelper((splat_id(1) << 40) + 100, (splat_id(6) << 40) - 1);
    testSplatStreamResetHelper((splat_id(5) << 40), (splat_id(5) << 40) + 20000);
    testSplatStreamResetHelper((splat_id(4) << 40), (splat_id(50) << 40) - 1);
}

template<typename SetType>
void TestSplatSubsettable<SetType>::testSplatStreamResetEmptyRange()
{
    testSplatStreamResetHelper(0, 0);
    testSplatStreamResetHelper(3, 3);
    testSplatStreamResetHelper(1000000000, 1000000000);
}

template<typename SetType>
void TestSplatSubsettable<SetType>::testSplatStreamResetNegativeRange()
{
    boost::scoped_ptr<Set> set(this->setFactory(this->splatData, this->grid.getSpacing(), 5));
    boost::scoped_ptr<SplatStreamReset> splatStream(set->makeSplatStreamReset());

    CPPUNIT_ASSERT_THROW(splatStream->reset(1, 0), std::invalid_argument);
    CPPUNIT_ASSERT_THROW(splatStream->reset(splat_id(1) << 33, 1), std::invalid_argument);
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
        set.addFile(new FastPly::Reader(store.back().data(), store.back().size(), 1.0f));
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
    return set.release();
}

void TestVectorSet::populate(
    VectorSet &set,
    const std::vector<std::vector<Splat> > &splatData)
{
    for (std::size_t i = 0; i < splatData.size(); i++)
    {
        set.insert(set.end(), splatData[i].begin(), splatData[i].end());
    }
}

VectorSet *TestVectorSet::setFactory(
    const std::vector<std::vector<Splat> > &splatData,
    float spacing, Grid::size_type bucketSize)
{
    (void) spacing;
    (void) bucketSize;

    std::auto_ptr<Set> set(new Set);
    populate(*set, splatData);
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

FastBlobSet<FileSet, std::vector<BlobData> > *TestFastFileSet::setFactory(
    const std::vector<std::vector<Splat> > &splatData,
    float spacing, Grid::size_type bucketSize)
{
    if (splatData.empty())
        return NULL; // otherwise computeBlobs will throw
    std::auto_ptr<Set> set(new Set);
    TestFileSet::populate(*set, splatData, store);
    set->computeBlobs(spacing, bucketSize);
    return set.release();
}

void TestFastFileSet::testEmpty()
{
    boost::scoped_ptr<Set> set(new Set);
    CPPUNIT_ASSERT_THROW(set->computeBlobs(2.5f, 5), std::runtime_error);
}

void TestFastFileSet::testProgress()
{
    boost::scoped_ptr<Set> set(new Set);
    TestFileSet::populate(*set, splatData, store);

    boost::iostreams::null_sink nullSink;
    boost::iostreams::stream<boost::iostreams::null_sink> nullStream(nullSink);
    set->computeBlobs(2.5f, 5, &nullStream);
}

FastBlobSet<VectorSet, std::vector<BlobData> > *TestFastVectorSet::setFactory(
    const std::vector<std::vector<Splat> > &splatData,
    float spacing, Grid::size_type bucketSize)
{
    if (splatData.empty())
        return NULL;
    std::auto_ptr<Set> set(new Set);
    TestVectorSet::populate(*set, splatData);
    set->computeBlobs(spacing, bucketSize);
    return set.release();
}

Subset<FastBlobSet<VectorSet, std::vector<BlobData> > > *
TestSubset::setFactory(const std::vector<std::vector<Splat> > &splatData,
                       float spacing, Grid::size_type bucketSize)
{
    if (splatData.empty())
        return NULL;

    std::tr1::mt19937 engine;
    std::tr1::bernoulli_distribution dist(0.75);
    std::tr1::variate_generator<std::tr1::mt19937 &, std::tr1::bernoulli_distribution> gen(engine, dist);

    TestVectorSet::populate(super, splatData);
    super.computeBlobs(spacing, bucketSize);
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
    CPPUNIT_ASSERT_EQUAL((unsigned int) flatSplats.size(), offset);
    flatSplats.swap(flatSubset);
    return set.release();
}
