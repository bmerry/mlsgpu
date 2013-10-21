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

#ifndef __CL_ENABLE_EXCEPTIONS
# define __CL_ENABLE_EXCEPTIONS 1
#endif

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <cppunit/Test.h>
#include <cppunit/TestCase.h>
#include <cppunit/TextTestRunner.h>
#include <cppunit/CompilerOutputter.h>
#include <cppunit/BriefTestProgressListener.h>
#include <cppunit/TestResult.h>
#include <cppunit/extensions/TestFactoryRegistry.h>
#include <boost/lexical_cast.hpp>
#include <boost/tr1/cmath.hpp>
#include <boost/program_options.hpp>
#include "../src/clh.h"


namespace po = boost::program_options;

namespace TestSet
{
    std::string perBuild()   { return "build"; }
    std::string perCommit()  { return "commit"; }
    std::string perNightly() { return "nightly"; }
};

void mlsgpuAssertDoublesEqual(double expected, double actual, double eps, const CppUnit::SourceLine &sourceLine)
{
    std::string expectedStr = boost::lexical_cast<std::string>(expected);
    std::string actualStr = boost::lexical_cast<std::string>(actual);
    if ((std::tr1::isnan)(expected) && (std::tr1::isnan)(actual))
        return;
    if ((std::tr1::isnan)(expected))
        CppUnit::Asserter::failNotEqual(expectedStr, actualStr, sourceLine, "expected is NaN");
    if ((std::tr1::isnan)(actual))
        CppUnit::Asserter::failNotEqual(expectedStr, actualStr, sourceLine, "actual is NaN");
    if (expected == actual)
        return;
    if ((std::tr1::isfinite)(expected) && (std::tr1::isfinite)(actual))
    {
        double err = abs(expected - actual);
        if (err > eps * abs(expected) && err > eps)
            CppUnit::Asserter::failNotEqual(expectedStr, actualStr, sourceLine,
                                            "Delta   : " + boost::lexical_cast<std::string>(err));
    }
    else
        CppUnit::Asserter::failNotEqual(expectedStr, actualStr, sourceLine, "");
}


static po::variables_map g_vm;

const po::variables_map &testGetOptions()
{
    return g_vm;
}

static void listTests(CppUnit::Test *root, std::string path)
{
    if (!path.empty())
        path += '/';
    path += root->getName();

    std::cout << path << '\n';
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
        ("test", po::value<std::string>()->default_value("build"), "Choose test")
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
            std::cout << desc << '\n';
            exit(0);
        }
        return vm;
    }
    catch (po::error &e)
    {
        std::cerr << e.what() << "\n\n" << desc << '\n';
        std::exit(1);
    }
}

int runTests(int argc, const char **argv, bool isMaster)
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
        std::string path = g_vm["test"].as<std::string>();

        CppUnit::BriefTestProgressListener listener;
        CppUnit::TextTestRunner runner;
        runner.addTest(rootSuite);
        runner.setOutputter(new CppUnit::CompilerOutputter(&runner.result(), std::cerr));
        if (g_vm.count("verbose"))
            runner.eventManager().addListener(&listener);
        bool success = runner.run(path, false, isMaster, false);
        return success ? 0 : 1;
    }
    catch (std::invalid_argument &e)
    {
        std::cerr << "\nERROR: " << e.what() << "\n";
        return 2;
    }
}
