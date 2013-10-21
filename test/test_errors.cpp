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
 * Test code for @ref errors.h.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <stdexcept>
#include <cstdlib>
#include <iostream>
#include "../src/errors.h"

#if MLSGPU_ASSERT_ABORT
# error "Cannot set MLSGPU_ASSERT_ABORT when unit testing"
#endif

#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/extensions/HelperMacros.h>
#include "testutil.h"

class TestErrors : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestErrors);
    CPPUNIT_TEST(testAssertPass);
#if DEBUG
    CPPUNIT_TEST(testAssertFail);
#endif
    CPPUNIT_TEST_SUITE_END();

public:
    void testAssertPass();
    void testAssertFail();
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestErrors, TestSet::perBuild());

void TestErrors::testAssertPass()
{
    CPPUNIT_ASSERT_NO_THROW(MLSGPU_ASSERT(true, std::domain_error));
}

void TestErrors::testAssertFail()
{
    CPPUNIT_ASSERT_THROW(MLSGPU_ASSERT(false, std::domain_error), std::domain_error);
}
