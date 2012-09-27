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
#include <cppunit/TestCaller.h>

const boost::program_options::variables_map &testGetOptions();

namespace TestSet
{

std::string perBuild();
std::string perCommit();
std::string perNightly();

};

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
#define MLSGPU_ASSERT_DOUBLES_EQUAL(actual, expected, eps) \
    mlsgpuAssertDoublesEqual(actual, expected, eps, CPPUNIT_SOURCELINE())

/**
 * Variant of @c CppUnit::assertDoublesEqual that accepts relative or absolute error.
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
 * Main program implementation.
 *
 * @param argc, argv  Command-line arguments
 * @param isMaster    Whether this is the master MPI process (true for single-process).
 * @return Status to return from @c main.
 */
int runTests(int argc, const char **argv, bool isMaster);

#endif /* !TESTMAIN_H */
