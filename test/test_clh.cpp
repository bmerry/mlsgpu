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
 * Utilities for writing tests of OpenCL kernels.
 */

#ifndef __CL_ENABLE_EXCEPTIONS
# define __CL_ENABLE_EXCEPTIONS 1
#endif

#if HAVE_CONFIG_H
# include <config.h>
#endif

#include <cppunit/TestFixture.h>
#include <cppunit/extensions/HelperMacros.h>
#include <CL/cl.hpp>
#include <boost/program_options.hpp>
#include <boost/smart_ptr/scoped_array.hpp>
#include <cstdlib>
#include <algorithm>
#include <locale>
#include <sstream>
#include <string>
#include "../src/tr1_cstdint.h"
#include "testutil.h"
#include "test_clh.h"
#include "../src/clh.h"
#include "../src/misc.h"

using namespace std;
namespace po = boost::program_options;

namespace CLH
{
namespace Test
{

void Mixin::setUpCL()
{
    const po::variables_map &vm = testGetOptions();
    std::vector<cl::Device> devices = CLH::findDevices(vm);
    if (devices.empty())
    {
        cerr << "No suitable OpenCL device found!" << endl;
        exit(1);
    }
    device = devices[0];
    context = CLH::makeContext(device);
    queue = cl::CommandQueue(context, device);
}

void Mixin::tearDownCL()
{
    context = NULL;
    device = NULL;
    queue = NULL;
}

cl::Buffer Mixin::createBuffer(cl_mem_flags flags, ::size_t size)
{
    ::size_t words = divUp(size, 4);
    if (words == 0) words = 1;
    boost::scoped_array<cl_uint> data(new cl_uint[words]);
    fill(data.get(), data.get() + words, 0xDEADBEEF);
    return cl::Buffer(context, flags | CL_MEM_COPY_HOST_PTR, size, data.get());
}

void TestFixture::setUp()
{
    CppUnit::TestFixture::setUp();
    setUpCL();
}

void TestFixture::tearDown()
{
    tearDownCL();
    CppUnit::TestFixture::tearDown();
}

} // namespace Test
} // namespace CLH

/// Tests for @ref CLH::ResourceUsage
class TestResourceUsage : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestResourceUsage);
    CPPUNIT_TEST(testConstructor);
    CPPUNIT_TEST(testAddBuffer);
    CPPUNIT_TEST(testAddImage);
    CPPUNIT_TEST(testAdd);
    CPPUNIT_TEST(testAddEquals);
    CPPUNIT_TEST(testMultiply);
    CPPUNIT_TEST_SUITE_END();

private:
    ///< Fixture in initial state
    CLH::ResourceUsage empty;
    ///< Fixture with some things added to it
    CLH::ResourceUsage used;

    void testConstructor();    ///< Test initial state
    void testAddBuffer();      ///< Test @ref CLH::ResourceUsage::addBuffer
    void testAddImage();       ///< Test @ref CLH::ResourceUsage::addImage
    void testAdd();            ///< Test <code>operator+</code>
    void testAddEquals();      ///< Test <code>operator+=</code>
    void testMultiply();       ///< Test <code>operator*</code>

public:
    virtual void setUp();
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestResourceUsage, TestSet::perBuild());

void TestResourceUsage::setUp()
{
    used.addBuffer("buffer", 1234);
    used.addImage("image", 15, 20, 3);
    // Should now have 1234 + 15*20*3 bytes = 2134 bytes allocated
    // Biggest allocation is 1234
}

void TestResourceUsage::testConstructor()
{
    MLSGPU_ASSERT_EQUAL(0, empty.getMaxMemory());
    MLSGPU_ASSERT_EQUAL(0, empty.getTotalMemory());
    MLSGPU_ASSERT_EQUAL(0, empty.getImageWidth());
    MLSGPU_ASSERT_EQUAL(0, empty.getImageHeight());
}

void TestResourceUsage::testAddBuffer()
{
    empty.addBuffer("buffer1", 100);
    MLSGPU_ASSERT_EQUAL(100, empty.getMaxMemory());
    MLSGPU_ASSERT_EQUAL(100, empty.getTotalMemory());
    empty.addBuffer("buffer2", 50);
    MLSGPU_ASSERT_EQUAL(100, empty.getMaxMemory());
    MLSGPU_ASSERT_EQUAL(150, empty.getTotalMemory());
    empty.addBuffer("buffer2", 200);
    MLSGPU_ASSERT_EQUAL(200, empty.getMaxMemory());
    MLSGPU_ASSERT_EQUAL(350, empty.getTotalMemory());
    const std::tr1::uint64_t big = UINT64_C(0x1234567812345678);
    empty.addBuffer("buffer3", big);
    MLSGPU_ASSERT_EQUAL(big, empty.getMaxMemory());
    MLSGPU_ASSERT_EQUAL(big + 350, empty.getTotalMemory());

    MLSGPU_ASSERT_EQUAL(0, empty.getImageWidth());
    MLSGPU_ASSERT_EQUAL(0, empty.getImageHeight());

    Statistics::Registry registry;
    empty.addStatistics(registry, "test.");
    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream << registry;
    const std::string stats = stream.str();
    MLSGPU_ASSERT_EQUAL(
        "test.all: 1311768465173141462\n"
        "test.buffer1: 100\n"
        "test.buffer2: 250\n"
        "test.buffer3: 1311768465173141112\n", stats);
}

void TestResourceUsage::testAddImage()
{
    empty.addImage("image1", 8, 16, 4);
    MLSGPU_ASSERT_EQUAL(512, empty.getMaxMemory());
    MLSGPU_ASSERT_EQUAL(512, empty.getTotalMemory());
    MLSGPU_ASSERT_EQUAL(8, empty.getImageWidth());
    MLSGPU_ASSERT_EQUAL(16, empty.getImageHeight());

    empty.addImage("image2", 18, 8, 3);
    MLSGPU_ASSERT_EQUAL(512, empty.getMaxMemory());
    MLSGPU_ASSERT_EQUAL(944, empty.getTotalMemory());
    MLSGPU_ASSERT_EQUAL(18, empty.getImageWidth());
    MLSGPU_ASSERT_EQUAL(16, empty.getImageHeight());

    empty.addImage("image3", 0x1000000, 0x1000000, 1);
    const std::tr1::uint64_t big = UINT64_C(0x1000000000000);
    MLSGPU_ASSERT_EQUAL(big, empty.getMaxMemory());
    MLSGPU_ASSERT_EQUAL(big + 944, empty.getTotalMemory());
    MLSGPU_ASSERT_EQUAL(std::size_t(0x1000000), empty.getImageWidth());
    MLSGPU_ASSERT_EQUAL(std::size_t(0x1000000), empty.getImageHeight());

    Statistics::Registry registry;
    empty.addStatistics(registry, "test.");
    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream << registry;
    const std::string stats = stream.str();
    MLSGPU_ASSERT_EQUAL(
        "test.all: 281474976711600\n"
        "test.image1: 512\n"
        "test.image2: 432\n"
        "test.image3: 281474976710656\n", stats);
}

void TestResourceUsage::testAdd()
{
    const std::tr1::uint64_t big = 0x12345678;
    empty.addBuffer("big", big);
    CLH::ResourceUsage sum = empty + used;
    MLSGPU_ASSERT_EQUAL(big, sum.getMaxMemory());
    MLSGPU_ASSERT_EQUAL(big + 2134, sum.getTotalMemory());
    MLSGPU_ASSERT_EQUAL(15, sum.getImageWidth());
    MLSGPU_ASSERT_EQUAL(20, sum.getImageHeight());
}

void TestResourceUsage::testAddEquals()
{
    const std::tr1::uint64_t big = 0x1234567812345678;
    empty.addBuffer("big", big);
    used += empty;
    MLSGPU_ASSERT_EQUAL(big, used.getMaxMemory());
    MLSGPU_ASSERT_EQUAL(big + 2134, used.getTotalMemory());
    MLSGPU_ASSERT_EQUAL(15, used.getImageWidth());
    MLSGPU_ASSERT_EQUAL(20, used.getImageHeight());
}

void TestResourceUsage::testMultiply()
{
    const std::tr1::uint64_t big = 0x12345678;
    used.addBuffer("big", big);
    CLH::ResourceUsage prod = used * 10;
    MLSGPU_ASSERT_EQUAL(big, prod.getMaxMemory());
    MLSGPU_ASSERT_EQUAL((big + 2134) * 10, prod.getTotalMemory());
    MLSGPU_ASSERT_EQUAL(15, prod.getImageWidth());
    MLSGPU_ASSERT_EQUAL(20, prod.getImageHeight());
}
