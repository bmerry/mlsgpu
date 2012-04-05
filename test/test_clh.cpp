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
#include <tr1/cstdint>
#include "testmain.h"
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
    queue = cl::CommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
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
    used.addBuffer(1234);
    used.addImage(15, 20, 3);
    // Should now have 1234 + 15*20*3 bytes = 2134 bytes allocated
    // Biggest allocation is 1234
}

void TestResourceUsage::testConstructor()
{
    CPPUNIT_ASSERT_EQUAL(std::tr1::uint64_t(0), empty.getMaxMemory());
    CPPUNIT_ASSERT_EQUAL(std::tr1::uint64_t(0), empty.getTotalMemory());
    CPPUNIT_ASSERT_EQUAL(std::size_t(0), empty.getImageWidth());
    CPPUNIT_ASSERT_EQUAL(std::size_t(0), empty.getImageHeight());
}

void TestResourceUsage::testAddBuffer()
{
    empty.addBuffer(100);
    CPPUNIT_ASSERT_EQUAL(std::tr1::uint64_t(100), empty.getMaxMemory());
    CPPUNIT_ASSERT_EQUAL(std::tr1::uint64_t(100), empty.getTotalMemory());
    empty.addBuffer(50);
    CPPUNIT_ASSERT_EQUAL(std::tr1::uint64_t(100), empty.getMaxMemory());
    CPPUNIT_ASSERT_EQUAL(std::tr1::uint64_t(150), empty.getTotalMemory());
    empty.addBuffer(200);
    CPPUNIT_ASSERT_EQUAL(std::tr1::uint64_t(200), empty.getMaxMemory());
    CPPUNIT_ASSERT_EQUAL(std::tr1::uint64_t(350), empty.getTotalMemory());
    const std::tr1::uint64_t big = UINT64_C(0x1234567812345678);
    empty.addBuffer(big);
    CPPUNIT_ASSERT_EQUAL(big, empty.getMaxMemory());
    CPPUNIT_ASSERT_EQUAL(big + 350, empty.getTotalMemory());

    CPPUNIT_ASSERT_EQUAL(std::size_t(0), empty.getImageWidth());
    CPPUNIT_ASSERT_EQUAL(std::size_t(0), empty.getImageHeight());
}

void TestResourceUsage::testAddImage()
{
    empty.addImage(8, 16, 4);
    CPPUNIT_ASSERT_EQUAL(std::tr1::uint64_t(512), empty.getMaxMemory());
    CPPUNIT_ASSERT_EQUAL(std::tr1::uint64_t(512), empty.getTotalMemory());
    CPPUNIT_ASSERT_EQUAL(std::size_t(8), empty.getImageWidth());
    CPPUNIT_ASSERT_EQUAL(std::size_t(16), empty.getImageHeight());

    empty.addImage(18, 8, 3);
    CPPUNIT_ASSERT_EQUAL(std::tr1::uint64_t(512), empty.getMaxMemory());
    CPPUNIT_ASSERT_EQUAL(std::tr1::uint64_t(944), empty.getTotalMemory());
    CPPUNIT_ASSERT_EQUAL(std::size_t(18), empty.getImageWidth());
    CPPUNIT_ASSERT_EQUAL(std::size_t(16), empty.getImageHeight());

    empty.addImage(0x1000000, 0x1000000, 1);
    const std::tr1::uint64_t big = UINT64_C(0x1000000000000);
    CPPUNIT_ASSERT_EQUAL(big, empty.getMaxMemory());
    CPPUNIT_ASSERT_EQUAL(big + 944, empty.getTotalMemory());
    CPPUNIT_ASSERT_EQUAL(std::size_t(0x1000000), empty.getImageWidth());
    CPPUNIT_ASSERT_EQUAL(std::size_t(0x1000000), empty.getImageHeight());
}

void TestResourceUsage::testAdd()
{
    const std::tr1::uint64_t big = 0x12345678;
    empty.addBuffer(big);
    CLH::ResourceUsage sum = empty + used;
    CPPUNIT_ASSERT_EQUAL(big, sum.getMaxMemory());
    CPPUNIT_ASSERT_EQUAL(big + 2134, sum.getTotalMemory());
    CPPUNIT_ASSERT_EQUAL(std::size_t(15), sum.getImageWidth());
    CPPUNIT_ASSERT_EQUAL(std::size_t(20), sum.getImageHeight());
}

void TestResourceUsage::testAddEquals()
{
    const std::tr1::uint64_t big = 0x1234567812345678;
    empty.addBuffer(big);
    used += empty;
    CPPUNIT_ASSERT_EQUAL(big, used.getMaxMemory());
    CPPUNIT_ASSERT_EQUAL(big + 2134, used.getTotalMemory());
    CPPUNIT_ASSERT_EQUAL(std::size_t(15), used.getImageWidth());
    CPPUNIT_ASSERT_EQUAL(std::size_t(20), used.getImageHeight());
}

void TestResourceUsage::testMultiply()
{
    const std::tr1::uint64_t big = 0x12345678;
    used.addBuffer(big);
    CLH::ResourceUsage prod = used * 10;
    CPPUNIT_ASSERT_EQUAL(big, prod.getMaxMemory());
    CPPUNIT_ASSERT_EQUAL((big + 2134) * 10, prod.getTotalMemory());
    CPPUNIT_ASSERT_EQUAL(std::size_t(15), prod.getImageWidth());
    CPPUNIT_ASSERT_EQUAL(std::size_t(20), prod.getImageHeight());
}
