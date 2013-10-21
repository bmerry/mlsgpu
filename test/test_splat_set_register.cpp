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
 * Registers the tests defined in test_splat_set.cpp for execution.
 */
#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/extensions/HelperMacros.h>
#include "test_splat_set.h"
#include "testutil.h"

CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestSplatToBuckets, TestSet::perBuild());
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestSplatToBucketsClass, TestSet::perBuild());
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestFileSet, TestSet::perBuild());
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestSequenceSet, TestSet::perBuild());
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestFastFileSet, TestSet::perBuild());
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestFastSequenceSet, TestSet::perBuild());
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestMerge, TestSet::perBuild());
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestSubset, TestSet::perBuild());

