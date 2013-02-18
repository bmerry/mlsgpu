/**
 * @file
 *
 * Utility functions used by @ref TestSplatSet and other classes.
 */

#ifndef TEST_SPLAT_SET_H
#define TEST_SPLAT_SET_H

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <vector>
#include <boost/ptr_container/ptr_vector.hpp>
#include "../src/splat.h"
#include "../src/splat_set.h"
#include "../src/allocator.h"
#include "testutil.h"

namespace SplatSet
{

/**
 * Implementation of the @ref SplatSet::SubsettableConcept that uses a
 * vector of vectors as the backing store, and assigns splat IDs in a similar
 * way to @ref SplatSet::FileSet.
 */
class VectorsSet : public std::vector<std::vector<Splat> >
{
public:
    static const unsigned int scanIdShift = 40;
    static const splat_id splatIdMask = (splat_id(1) << scanIdShift) - 1;

    splat_id maxSplats() const
    {
        splat_id total = 0;
        for (std::size_t i = 0; i < size(); i++)
        {
            total += at(i).size();
        }
        return total;
    }

    SplatStream *makeSplatStream(bool useOMP = true) const
    {
        return makeSplatStream(&detail::rangeAll, &detail::rangeAll + 1, useOMP);
    }

    BlobStream *makeBlobStream(const Grid &grid, Grid::size_type bucketSize) const
    {
        return new SimpleBlobStream(makeSplatStream(), grid, bucketSize);
    }

    template<typename RangeIterator>
    SplatStream *makeSplatStream(RangeIterator firstRange, RangeIterator lastRange, bool useOMP = false) const
    {
        (void) useOMP;
        return new MySplatStream<RangeIterator>(*this, firstRange, lastRange);
    }

private:
    template<typename RangeIterator>
    class MySplatStream : public SplatStream
    {
    public:
        virtual std::size_t read(Splat *splats, splat_id *splatIds, std::size_t count)
        {
            std::size_t oldCount = count;
            while (count > 0)
            {
                if (curRange == lastRange || cur >> scanIdShift >= owner.size())
                    return oldCount - count;

retry:
                if (cur >= curRange->second)
                {
                    ++curRange;
                    if (curRange == lastRange)
                        return oldCount - count;
                    cur = curRange->first;
                    goto retry;
                }
                std::size_t scan = cur >> scanIdShift;
                if (scan >= owner.size())
                    return oldCount - count;
                splat_id scanEnd = owner[scan].size() + (splat_id(scan) << scanIdShift);
                if (cur >= scanEnd)
                {
                    scan++;
                    if (scan >= owner.size())
                        return oldCount - count;
                    cur = splat_id(scan) << scanIdShift;
                    goto retry;
                }

                std::size_t n = std::min(splat_id(count), scanEnd - cur);
                std::size_t pos = cur & splatIdMask;
                for (std::size_t i = 0; i < n; i++)
                {
                    splats[i] = owner[scan][pos + i];
                    if (splatIds != NULL)
                        splatIds[i] = cur + i;
                }
                splats += n;
                if (splatIds != NULL)
                    splatIds += n;
                count -= n;
                cur += n;
            }
            return oldCount - count;
        }

        MySplatStream(const VectorsSet &owner, RangeIterator firstRange, RangeIterator lastRange)
            : owner(owner), curRange(firstRange), lastRange(lastRange)
        {
            if (curRange != lastRange)
                cur = curRange->first;
        }

    private:
        const VectorsSet &owner;
        splat_id cur;
        RangeIterator curRange, lastRange;
    };
};

} // namespace SplatSet

/**
 * Creates a sample set of splats for use in a test case. The resulting set of
 * splats is intended to be interesting when used with a grid spacing of 2.5
 * and an origin at the origin. Normally one would pass an instance of @ref
 * SplatSet::VectorsSet as @a splats.
 *
 * To make this easy to visualise, all splats are placed on a single Z plane.
 * This plane is along a major boundary, so when bucketing, each block can be
 * expected to appear twice (once on each side of the boundary).
 *
 * To see the splats graphically, save the following to a file and run gnuplot
 * over it. The coordinates are in grid space rather than world space:
 * <pre>
 * set xrange [0:16]
 * set yrange [0:20]
 * set size square
 * set xtics 4
 * set ytics 4
 * set grid
 * plot '-' with points
 * 4 8
 * 12 6.8
 * 12.8 4.8
 * 12.8 7.2
 * 14.8 7.2
 * 14 6.4
 * 4.8 14.8
 * 5.2 14.8
 * 4.8 15.2
 * 5.2 15.2
 * 6.8 12.8
 * 7.2 13.2
 * 10 18
 * e
 * pause -1
 * </pre>
 */
void createSplats(std::vector<std::vector<Splat> > &splats);

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
void createSplats2(std::vector<std::vector<Splat> > &splats);

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
                        const std::vector<SplatSet::splat_id> &ids);

    /// Check that retrieved blobs match what is expected
    void validateBlobs(const std::vector<Splat> &expected,
                       const std::vector<SplatSet::BlobInfo> &actual,
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

/// Tests for @ref SplatSet::SequenceSet
class TestSequenceSet : public TestSplatSubsettable<SplatSet::SequenceSet<const Splat *> >
{
    CPPUNIT_TEST_SUB_SUITE(TestSequenceSet, TestSplatSubsettable<SplatSet::SequenceSet<const Splat *> >);
    CPPUNIT_TEST_SUITE_END();
private:
    std::vector<Splat> store; ///< Backing data for the returned sets
protected:
    virtual Set *setFactory(const std::vector<std::vector<Splat> > &splatData,
                            float spacing, Grid::size_type bucketSize);
public:
    /// Adds all splats in @a splatData to the store and creates the set.
    static void populate(
        SplatSet::SequenceSet<const Splat *> &set,
        const std::vector<std::vector<Splat> > &splatData,
        std::vector<Splat> &store);
};

/// Tests for @ref SplatSet::FastBlobSet.
template<typename BaseType>
class TestFastBlobSet : public TestSplatSubsettable<SplatSet::FastBlobSet<BaseType> >
{
    typedef TestSplatSubsettable<SplatSet::FastBlobSet<BaseType> > BaseFixture;
    CPPUNIT_TEST_SUB_SUITE(TestFastBlobSet<BaseType>, BaseFixture);
    CPPUNIT_TEST(testBoundingGrid);
    CPPUNIT_TEST(testAddBlob);
    CPPUNIT_TEST_SUITE_END_ABSTRACT();
public:
    typedef typename BaseFixture::Set Set;

    void testBoundingGrid();         ///< Tests that the extracted bounding box is correct
    void testAddBlob();              ///< Tests the encoding of blobs
};

/// Tests for @ref SplatSet::FastBlobSet<SplatSet::SequenceSet<const Splat *> >.
class TestFastSequenceSet : public TestFastBlobSet<SplatSet::SequenceSet<const Splat *> >
{
    typedef TestFastBlobSet<SplatSet::SequenceSet<const Splat *> > BaseFixture;
    CPPUNIT_TEST_SUB_SUITE(TestFastSequenceSet, BaseFixture);
    CPPUNIT_TEST_SUITE_END();
private:
    std::vector<Splat> store; ///< Backing data for the returned sets
protected:
    virtual Set *setFactory(const std::vector<std::vector<Splat> > &splatData,
                            float spacing, Grid::size_type bucketSize);
};

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
        const SplatSet::splat_id rangesA[][2],
        std::size_t numB,
        const SplatSet::splat_id rangesB[][2],
        std::size_t numExpected,
        const SplatSet::splat_id rangesExpected[][2]);
public:
    void testMergeEmpty();     ///< Test @ref SplatSet::merge with two empty subsets
    void testMergeTail();      ///< Test @ref SplatSet::merge with tail elements in one set
    void testMergeGeneral();   ///< Miscellaneous tests for @ref SplatSet::merge.
};

/// Tests for @ref SplatSet::Subset
class TestSubset : public TestSplatSet<SplatSet::Subset<SplatSet::FastBlobSet<SplatSet::SequenceSet<const Splat *> > > >
{
    typedef SplatSet::FastBlobSet<SplatSet::SequenceSet<const Splat *> > Super;
    typedef TestSplatSet<SplatSet::Subset<SplatSet::FastBlobSet<SplatSet::SequenceSet<const Splat *> > > > BaseFixture;
    CPPUNIT_TEST_SUB_SUITE(TestSubset, BaseFixture);
    CPPUNIT_TEST_SUITE_END();
private:
    Super super;
    std::vector<Splat> store; ///< Backing data for the returned sets
protected:
    virtual Set *setFactory(const std::vector<std::vector<Splat> > &splatData,
                            float spacing, Grid::size_type bucketSize);
};

/// Tests for @ref SplatSet::FileSet
class TestFileSet : public TestSplatSubsettable<SplatSet::FileSet>
{
    CPPUNIT_TEST_SUB_SUITE(TestFileSet, TestSplatSubsettable<SplatSet::FileSet>);
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
    static void populate(SplatSet::FileSet &set, const std::vector<std::vector<Splat> > &splatData,
                         std::vector<std::string> &store);
};

/// Tests for @ref SplatSet::FastBlobSet<SplatSet::FileSet>.
class TestFastFileSet : public TestFastBlobSet<SplatSet::FileSet>
{
    typedef TestFastBlobSet<SplatSet::FileSet> BaseFixture;
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
    const std::vector<SplatSet::splat_id> &ids)
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
    const std::vector<SplatSet::BlobInfo> &actual,
    const Grid &grid,
    Grid::size_type bucketSize)
{
    std::size_t nextSplat = 0;
    for (std::size_t i = 0; i < actual.size(); i++)
    {
        CPPUNIT_ASSERT(i == 0 || actual[i].firstSplat >= actual[i - 1].lastSplat);
        const SplatSet::BlobInfo &cur = actual[i];
        CPPUNIT_ASSERT(cur.lastSplat > cur.firstSplat);
        SplatSet::splat_id numSplats = cur.lastSplat - cur.firstSplat;
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
        boost::scoped_ptr<SplatSet::SplatStream> stream(set->makeSplatStream());
        std::vector<Splat> actual;
        std::vector<SplatSet::splat_id> ids;
        const std::size_t count = 5; // there are files bigger, smaller and exactly this size
        Splat buffer[count];
        SplatSet::splat_id bufferIds[count];
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
        boost::scoped_ptr<SplatSet::BlobStream> stream(set->makeBlobStream(grid, bucketSize));
        std::vector<SplatSet::BlobInfo> actual;
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
        boost::scoped_ptr<SplatSet::BlobStream> stream;
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
    std::vector<Splat> expected, actual;
    std::vector<SplatSet::splat_id> expectedIds, actualIds;

    const unsigned int bucketSize = 5;
    boost::scoped_ptr<Set> set(this->setFactory(this->splatData, this->grid.getSpacing(), bucketSize));

    for (RangeIterator curRange = firstRange; curRange != lastRange; ++curRange)
    {
        boost::scoped_ptr<SplatSet::SplatStream> splatStream(set->makeSplatStream());
        Splat splat;
        SplatSet::splat_id id;
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
        boost::scoped_ptr<SplatSet::SplatStream> splatStream(set->makeSplatStream(firstRange, lastRange));
        const std::size_t count = 3;
        while (true)
        {
            Splat buffer[count];
            SplatSet::splat_id bufferIds[count];
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
    typedef std::pair<SplatSet::splat_id, SplatSet::splat_id> Range;
    std::vector<Range> ranges;

    ranges.push_back(Range(0, 1000000));
    ranges.push_back(Range(3, SplatSet::splat_id(3) << 40));
    ranges.push_back(Range(2, (SplatSet::splat_id(3) << 40) - 1));
    ranges.push_back(Range((SplatSet::splat_id(1) << 40) + 100, (SplatSet::splat_id(6) << 40) - 1));
    ranges.push_back(Range((SplatSet::splat_id(5) << 40), (SplatSet::splat_id(5) << 40) + 20000));
    ranges.push_back(Range((SplatSet::splat_id(4) << 40), (SplatSet::splat_id(50) << 40) - 1));
    testSplatStreamSeekHelper(ranges.begin(), ranges.end());
}

template<typename SetType>
void TestSplatSubsettable<SetType>::testSplatStreamSeekZeroRanges()
{
    typedef std::pair<SplatSet::splat_id, SplatSet::splat_id> Range;
    std::vector<Range> ranges;

    testSplatStreamSeekHelper(ranges.begin(), ranges.end());
}

template<typename SetType>
void TestSplatSubsettable<SetType>::testSplatStreamSeekEmptyRange()
{
    typedef std::pair<SplatSet::splat_id, SplatSet::splat_id> Range;
    std::vector<Range> ranges;

    ranges.push_back(Range(0, 0));
    ranges.push_back(Range(3, 3));
    ranges.push_back(Range(1000000000, 1000000000));
    testSplatStreamSeekHelper(ranges.begin(), ranges.end());
}

template<typename SetType>
void TestSplatSubsettable<SetType>::testSplatStreamSeekNegativeRange()
{
    typedef std::pair<SplatSet::splat_id, SplatSet::splat_id> Range;
    std::vector<Range> ranges;

    ranges.push_back(Range(1, 0));
    ranges.push_back(Range(SplatSet::splat_id(1) << 33, 1));
    testSplatStreamSeekHelper(ranges.begin(), ranges.end());
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
    Statistics::Container::vector<typename SplatSet::FastBlobSet<BaseType>::BlobData> blobData("mem.test.blobData");
    SplatSet::BlobInfo prevBlob, curBlob;

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
    Set set;
    set.blobFiles.push_back(typename SplatSet::FastBlobSet<BaseType>::BlobFile());
    {
        boost::filesystem::ofstream out;
        createTmpFile(set.blobFiles[0].path, out);
        out.exceptions(std::ios::failbit | std::ios::badbit);
        out.write(reinterpret_cast<const char *>(&blobData[0]), blobData.size() * sizeof(blobData[0]));
    }
    set.blobFiles[0].nBlobs = 2;
    set.internalBucketSize = 1;

    SplatSet::BlobInfo blob;
    boost::scoped_ptr<SplatSet::BlobStream> stream(set.makeBlobStream(set.boundingGrid, set.internalBucketSize));
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

#endif /* !TEST_SPLAT_SET_H */
