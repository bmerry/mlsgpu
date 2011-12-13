/**
 * @file
 *
 * Utilities for writing tests of OpenCL kernels.
 */

#ifndef TEST_CLH_H

#if HAVE_CONFIG_H
# include <config.h>
#endif

#include <cppunit/TestFixture.h>
#include <CL/cl.hpp>

namespace CLH
{

namespace Test
{


/**
 * Test fixture class that handles OpenCL setup.
 *
 * The provided command queue is guaranteed to be in-order. Create a
 * separate command queue if out-of-order execution is needed.
 */
class TestFixture : public CppUnit::TestFixture
{
protected:
    cl::Context context;           ///< OpenCL context
    cl::Device device;             ///< OpenCL device
    cl::CommandQueue queue;        ///< OpenCL command queue
public:
    virtual void setUp();          ///< Create context, etc.
    virtual void tearDown();       ///< Release context, etc.
};

} // namespace Test
} // namespace CLH

#endif /* !TEST_CLH_H */
