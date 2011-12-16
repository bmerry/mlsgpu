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
 * Mixin that can be included in another test fixture class to handle OpenCL
 * setup. In most cases one can just inherit from @ref TestFixture. This class
 * is mainly useful when already inheriting from a subclass of @c
 * CppUnit::TestFixture.
 */
class Mixin
{
protected:
    cl::Context context;           ///< OpenCL context
    cl::Device device;             ///< OpenCL device
    cl::CommandQueue queue;        ///< OpenCL command queue

    void setUpCL();                ///< Create context, etc.
    void tearDownCL();             ///< Release context, etc.
};

/**
 * Test fixture class that handles OpenCL setup.
 *
 * The provided command queue is guaranteed to be in-order. Create a
 * separate command queue if out-of-order execution is needed.
 */
class TestFixture : public CppUnit::TestFixture, Mixin
{
public:
    virtual void setUp();          ///< Create context, etc.
    virtual void tearDown();       ///< Release context, etc.
};

} // namespace Test
} // namespace CLH

#endif /* !TEST_CLH_H */
