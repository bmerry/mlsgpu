/**
 * @file
 *
 * Common definitions for tests.
 */

#ifndef TESTMAIN_H
#define TESTMAIN_H

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

#endif /* !TESTMAIN_H */
