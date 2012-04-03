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
#include <CL/cl.hpp>
#include <boost/program_options.hpp>
#include <boost/smart_ptr/scoped_array.hpp>
#include <cstdlib>
#include <algorithm>
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
    context = CLH::makeContext(devices[0]);
    queue = cl::CommandQueue(context, devices[0], CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
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
