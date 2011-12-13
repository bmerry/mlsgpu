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
#include <cstdlib>
#include "testmain.h"
#include "test_clh.h"
#include "../src/clh.h"

using namespace std;
namespace po = boost::program_options;

namespace CLH
{
namespace Test
{

void TestFixture::setUp()
{
    CppUnit::TestFixture::setUp();
    const po::variables_map &vm = testGetOptions();
    device = CLH::findDevice(vm);
    if (device() == NULL)
    {
        cerr << "No suitable OpenCL device found!" << endl;
        exit(1);
    }
    context = CLH::makeContext(device);
    queue = cl::CommandQueue(context, device, 0);
}

void TestFixture::tearDown()
{
    context = NULL;
    device = NULL;
    queue = NULL;

    CppUnit::TestFixture::tearDown();
}

} // namespace Test
} // namespace CLH
