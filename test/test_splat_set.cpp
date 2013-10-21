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
 * Test code for @ref SplatSet. This file does not register any tests for execution,
 * because the code is reused by @ref test_splat_set_mpi.cpp and it is linked into
 * @c testmpi. The registrations are all in @ref test_splat_set_register.cpp.
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

void createSplats2(std::vector<std::vector<Splat> > &splats)
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

void TestFileSet::populate(
    SplatSet::FileSet &set,
    const std::vector<std::vector<Splat> > &splatData,
    std::vector<std::string> &store)
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
        set.addFile(new FastPly::Reader(
                MemoryReaderFactory(store.back()),
                "dummy",
                1.0f, std::numeric_limits<float>::infinity()));
    }
}

SplatSet::FileSet *TestFileSet::setFactory(
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
    SplatSet::SequenceSet<const Splat *> &set,
    const std::vector<std::vector<Splat> > &splatData,
    std::vector<Splat> &store)
{
    for (std::size_t i = 0; i < splatData.size(); i++)
        store.insert(store.end(), splatData[i].begin(), splatData[i].end());
    set.reset(&store[0], &store[0] + store.size());
}

SplatSet::SequenceSet<const Splat *> *TestSequenceSet::setFactory(
    const std::vector<std::vector<Splat> > &splatData,
    float spacing, Grid::size_type bucketSize)
{
    (void) spacing;
    (void) bucketSize;

    std::auto_ptr<Set> set(new Set);
    populate(*set, splatData, store);
    return set.release();
}

SplatSet::FastBlobSet<SplatSet::FileSet> *TestFastFileSet::setFactory(
    const std::vector<std::vector<Splat> > &splatData,
    float spacing, Grid::size_type bucketSize)
{
    if (splatData.empty())
        return NULL; // otherwise computeBlobs will throw
    std::auto_ptr<Set> set(new Set);
    TestFileSet::populate(*set, splatData, store);
    set->computeBlobs(spacing, bucketSize, NULL, false);
    return set.release();
}

void TestFastFileSet::testEmpty()
{
    boost::scoped_ptr<Set> set(new Set);
    CPPUNIT_ASSERT_THROW(set->computeBlobs(2.5f, 5, NULL, false), std::runtime_error);
}

void TestFastFileSet::testProgress()
{
    boost::scoped_ptr<Set> set(new Set);
    TestFileSet::populate(*set, splatData, store);

    boost::iostreams::null_sink nullSink;
    boost::iostreams::stream<boost::iostreams::null_sink> nullStream(nullSink);
    set->computeBlobs(2.5f, 5, &nullStream, false);
}

SplatSet::FastBlobSet<SplatSet::SequenceSet<const Splat *> > *TestFastSequenceSet::setFactory(
    const std::vector<std::vector<Splat> > &splatData,
    float spacing, Grid::size_type bucketSize)
{
    if (splatData.empty())
        return NULL;
    std::auto_ptr<Set> set(new Set);
    TestSequenceSet::populate(*set, splatData, store);
    set->computeBlobs(spacing, bucketSize, NULL, false);
    return set.release();
}

SplatSet::Subset<SplatSet::FastBlobSet<SplatSet::SequenceSet<const Splat *> > > *
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
    std::vector<Splat> flatSubset;
    unsigned int offset = 0;
    boost::scoped_ptr<SplatSet::BlobStream> superBlobs(super.makeBlobStream(super.getBoundingGrid(), bucketSize));
    while (!superBlobs->empty())
    {
        const SplatSet::BlobInfo blob = **superBlobs;
        SplatSet::splat_id numSplats = blob.lastSplat - blob.firstSplat;
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
    const SplatSet::splat_id rangesA[][2],
    std::size_t numB,
    const SplatSet::splat_id rangesB[][2],
    std::size_t numExpected,
    const SplatSet::splat_id rangesExpected[][2])
{
    SplatSet::SubsetBase a, b;
    for (std::size_t i = 0; i < numA; i++)
        a.addRange(rangesA[i][0], rangesA[i][1]);
    for (std::size_t i = 0; i < numB; i++)
        b.addRange(rangesB[i][0], rangesB[i][1]);
    a.flush();
    b.flush();

    SplatSet::SubsetBase ans;
    SplatSet::merge(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(ans));
    ans.flush();
    std::size_t pos = 0;
    for (SplatSet::SubsetBase::const_iterator i = ans.begin(); i != ans.end(); ++i)
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
    const SplatSet::splat_id rangesA[][2] =
    {
        { 1, 3 },
        { 20, 22 },
        { 25, 30 }
    };
    const SplatSet::splat_id rangesB[][2] =
    {
        { 3, 5 },
        { 7, 10 }
    };
    const SplatSet::splat_id rangesExpected[][2] =
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
    const SplatSet::splat_id rangesA[][2] =
    {
        { 1, 10 },
        { 20, 30 },
        { 40, 50 }
    };
    const SplatSet::splat_id rangesB[][2] =
    {
        { 3, 8 },
        { 9, 15 },
        { 18, 22 },
        { 30, 40 }
    };
    const SplatSet::splat_id rangesExpected[][2] =
    {
        { 1, 15 },
        { 18, 50 }
    };
    testMergeHelper(3, rangesA, 4, rangesB, 2, rangesExpected);
}
