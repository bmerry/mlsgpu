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
 * Common definitions for tests.
 */

#ifndef TESTUTIL_H
#define TESTUTIL_H

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <string>
#include <boost/program_options.hpp>
#include <boost/function.hpp>
#include <boost/ref.hpp>
#include <boost/exception/all.hpp>
#include <cppunit/TestCaller.h>
#include <cppunit/extensions/ExceptionTestCaseDecorator.h>

const boost::program_options::variables_map &testGetOptions();

namespace TestSet
{

std::string perBuild();
std::string perCommit();
std::string perNightly();

};

/**
 * Decorator that checks that an exception of a specific type is thrown that also
 * contains a specific filename encoded using @c boost::errinfo_file_name.
 */
template<class ExpectedException>
class FilenameExceptionTestCaseDecorator : public CppUnit::ExceptionTestCaseDecorator<ExpectedException>
{
public:
    FilenameExceptionTestCaseDecorator(CppUnit::TestCase *test, const std::string &filename)
        : CppUnit::ExceptionTestCaseDecorator<ExpectedException>(test), filename(filename) {}

private:
    const std::string filename;

    virtual void checkException(ExpectedException &e)
    {
        std::string *exceptionFilename = boost::get_error_info<boost::errinfo_file_name>(e);
        CPPUNIT_ASSERT(exceptionFilename != NULL);
        CPPUNIT_ASSERT_EQUAL(filename, *exceptionFilename);
    }
};

#define TEST_EXCEPTION_FILENAME(testMethod, ExceptionType, filename) \
    CPPUNIT_TEST_SUITE_ADD_TEST(                                     \
        (new FilenameExceptionTestCaseDecorator<ExceptionType>(      \
            new CppUnit::TestCaller<TestFixtureType>(                \
                context.getTestNameFor(#testMethod),                 \
                &TestFixtureType::testMethod,                        \
                context.makeFixture()), filename)))

/* Generalized test caller that takes any function object.
 */
template<class Fixture>
class GenericTestCaller : public CppUnit::TestCaller<Fixture>
{
public:
    typedef boost::function<void(Fixture *)> TestFunction;

    GenericTestCaller(const std::string &name, const TestFunction &function, Fixture &fixture) :
        CppUnit::TestCaller<Fixture>(name, NULL, fixture), fixture(&fixture), function(function) {}

    GenericTestCaller(const std::string &name, const TestFunction &function, Fixture *fixture) :
        CppUnit::TestCaller<Fixture>(name, NULL, fixture), fixture(fixture), function(function) {}

    void runTest()
    {
        function(fixture);
    }

private:
    Fixture *fixture;
    TestFunction function;
};

/**
 * Macro wrapper around @ref mlsgpuAssertDoublesEqual.
 */
#define MLSGPU_ASSERT_DOUBLES_EQUAL(expected, actual, eps) \
    mlsgpuAssertDoublesEqual( (expected), (actual), (eps), CPPUNIT_SOURCELINE())

/**
 * Variant of @c CppUnit::assertDoubleEquals that accepts relative or absolute error.
 *
 * - If @a expected or @a actual are both NaN, passes.
 * - If one of @a expected or @a actual is NaN, fails.
 * - If @a expected or @a actual is an infinity, fails unless they're equal.
 * - Otherwise, passes if |@a expected - @a actual| <= @a eps or
 *   if |@a expected - @a actual| <= @a eps * @a expected.
 *
 * @param expected     Expected value
 * @param actual       Actual value
 * @param eps          Error tolerance
 * @param sourceLine   Pass @c CPPUNIT_SOURCELINE
 *
 * @see @ref MLSGPU_ASSERT_DOUBLES_EQUAL
 */
void mlsgpuAssertDoublesEqual(double expected, double actual, double eps, const CppUnit::SourceLine &sourceLine);

/**
 * Macro wrapper around @ref mlsgpuAssertEqual. This is like @c CPPUNIT_ASSERT_EQUAL,
 * but allows the two values to have different types. The expected value is coerced to
 * the type of the actual value.
 */
#define MLSGPU_ASSERT_EQUAL(expected, actual) \
    (mlsgpuAssertEqual( (expected), (actual), CPPUNIT_SOURCELINE(), "" ) )

/**
 * Variant of @c CppUnit::assertEquals that accepts two different types.
 */
template<typename E, typename A>
void mlsgpuAssertEqual(const E &expected, const A &actual, CppUnit::SourceLine sourceLine, const std::string &msg)
{
    CppUnit::assertEquals(static_cast<A>(expected), actual, sourceLine, msg);
}

/**
 * Main program implementation.
 *
 * @param argc, argv  Command-line arguments
 * @param isMaster    Whether this is the master MPI process (true for single-process).
 * @return Status to return from @c main.
 */
int runTests(int argc, const char **argv, bool isMaster);

#endif /* !TESTMAIN_H */
