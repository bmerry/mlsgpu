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

