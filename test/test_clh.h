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

#ifndef TEST_CLH_H
#define TEST_CLH_H

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
     *
     * @note The actual buffer size may be slightly larger than requested. Do not rely
     * on querying the buffer size.
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
