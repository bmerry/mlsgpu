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

    /**
     * Create a buffer which contains 0xDEADBEEF repeated throughout.
     *
     * @param flags    Flags to pass to CL (do not specify @c CL_MEM_COPY_HOST_PTR or @c CL_MEM_USE_HOST_PTR).
     * @param size     The buffer size.
     * @return The initialized buffer.
     */
    cl::Buffer createBuffer(cl_mem_flags flags, ::size_t size);
};

/**
 * Test fixture class that handles OpenCL setup.
 *
 * The provided command queue is guaranteed to be in-order. Create a
 * separate command queue if out-of-order execution is needed.
 */
class TestFixture : public CppUnit::TestFixture, public Mixin
{
public:
    virtual void setUp();          ///< Create context, etc.
    virtual void tearDown();       ///< Release context, etc.
};

} // namespace Test
} // namespace CLH

#endif /* !TEST_CLH_H */
