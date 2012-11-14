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
    CPPUNIT_TEST(testAssertFail);
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
