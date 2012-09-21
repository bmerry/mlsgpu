/**
 * @file
 * Main program for running unit tests.
 */

#ifndef __CL_ENABLE_EXCEPTIONS
# define __CL_ENABLE_EXCEPTIONS 1
#endif

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <iostream>
#include <cppunit/Test.h>
#include <cppunit/TestCase.h>
#include <cppunit/TextTestRunner.h>
#include <cppunit/CompilerOutputter.h>
#include <cppunit/BriefTestProgressListener.h>
#include <cppunit/TestResult.h>
#include <cppunit/extensions/TestFactoryRegistry.h>
#include <boost/program_options.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/tr1/cmath.hpp>
#include <string>
#include <stdexcept>
#include <typeinfo>
#include "../src/clh.h"

using namespace std;
namespace po = boost::program_options;

namespace TestSet
{
string perBuild()   { return "build"; }
string perCommit()  { return "commit"; }
string perNightly() { return "nightly"; }
};

void mlsgpuAssertDoublesEqual(double expected, double actual, double eps, const CppUnit::SourceLine &sourceLine)
{
    string expectedStr = boost::lexical_cast<string>(expected);
    string actualStr = boost::lexical_cast<string>(actual);
    if ((tr1::isnan)(expected) && (tr1::isnan)(actual))
        return;
    if ((tr1::isnan)(expected))
        CppUnit::Asserter::failNotEqual(expectedStr, actualStr, sourceLine, "expected is NaN");
    if ((tr1::isnan)(actual))
        CppUnit::Asserter::failNotEqual(expectedStr, actualStr, sourceLine, "actual is NaN");
    if (expected == actual)
        return;
    if ((tr1::isfinite)(expected) && (tr1::isfinite)(actual))
    {
        double err = abs(expected - actual);
        if (err > eps * abs(expected) && err > eps)
            CppUnit::Asserter::failNotEqual(expectedStr, actualStr, sourceLine,
                                            "Delta   : " + boost::lexical_cast<string>(err));
    }
    else
        CppUnit::Asserter::failNotEqual(expectedStr, actualStr, sourceLine, "");
}


static po::variables_map g_vm;

const po::variables_map &testGetOptions()
{
    return g_vm;
}

static void listTests(CppUnit::Test *root, string path)
{
    if (!path.empty())
        path += '/';
    path += root->getName();

    cout << path << '\n';
    for (int i = 0; i < root->getChildTestCount(); i++)
    {
        CppUnit::Test *sub = root->getChildTestAt(i);
        listTests(sub, path);
    }
}

static po::variables_map processOptions(int argc, const char **argv)
{
    po::options_description desc("Options");
    desc.add_options()
        ("help",                                      "Show help");

    po::options_description test("Test options");
    test.add_options()
        ("test", po::value<string>()->default_value("build"), "Choose test")
        ("list",                                      "List all tests")
        ("verbose,v",                                 "Show result of each test as it runs");
    desc.add(test);

    po::options_description cl("OpenCL options");
    CLH::addOptions(cl);
    desc.add(cl);

    try
    {
        po::variables_map vm;
        po::store(po::command_line_parser(argc, argv)
                  .style(po::command_line_style::default_style & ~po::command_line_style::allow_guessing)
                  .options(desc)
                  .run(), vm);
        po::notify(vm);

        if (vm.count("help"))
        {
            cout << desc << '\n';
            exit(0);
        }
        return vm;
    }
    catch (po::error &e)
    {
        cerr << e.what() << "\n\n" << desc << '\n';
        exit(1);
    }
}

int main(int argc, const char **argv)
{
    try
    {
        g_vm = processOptions(argc, argv);

        CppUnit::TestSuite *rootSuite = new CppUnit::TestSuite("All tests");
        CppUnit::TestSuite *buildSuite = new CppUnit::TestSuite("build");
        CppUnit::TestSuite *commitSuite = new CppUnit::TestSuite("commit");
        CppUnit::TestSuite *nightlySuite = new CppUnit::TestSuite("nightly");

        CppUnit::TestFactoryRegistry::getRegistry().addTestToSuite(rootSuite);
        CppUnit::TestFactoryRegistry::getRegistry(TestSet::perBuild()).addTestToSuite(buildSuite);
        CppUnit::TestFactoryRegistry::getRegistry(TestSet::perCommit()).addTestToSuite(commitSuite);
        CppUnit::TestFactoryRegistry::getRegistry(TestSet::perNightly()).addTestToSuite(nightlySuite);

        // Chain the subsuites, so that the bigger ones run the smaller ones too
        commitSuite->addTest(buildSuite);
        nightlySuite->addTest(commitSuite);
        rootSuite->addTest(nightlySuite);

        if (g_vm.count("list"))
        {
            listTests(rootSuite, "");
            return 0;
        }
        string path = g_vm["test"].as<string>();

        CppUnit::BriefTestProgressListener listener;
        CppUnit::TextTestRunner runner;
        runner.addTest(rootSuite);
        runner.setOutputter(new CppUnit::CompilerOutputter(&runner.result(), std::cerr));
        if (g_vm.count("verbose"))
            runner.eventManager().addListener(&listener);
        bool success = runner.run(path, false, true, false);
        return success ? 0 : 1;
    }
    catch (invalid_argument &e)
    {
        cerr << "\nERROR: " << e.what() << "\n";
        return 2;
    }
}
