#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <iostream>
#include <cppunit/Test.h>
#include <cppunit/TestCase.h>
#include <cppunit/TextTestRunner.h>
#include <cppunit/CompilerOutputter.h>
#include <cppunit/extensions/TestFactoryRegistry.h>
#include <string>
#include <stdexcept>
#include <typeinfo>

using namespace std;

namespace TestSet
{
string perBuild()   { return "build"; }
string perCommit()  { return "commit"; }
string perNightly() { return "nightly"; }
};

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

int main(int argc, const char * const * argv)
{
    string path = TestSet::perBuild();
    if (argc > 1)
        path = argv[1];

    try
    {
        CppUnit::TextTestRunner runner;
        runner.addTest(CppUnit::TestFactoryRegistry::getRegistry().makeTest());
        runner.addTest(CppUnit::TestFactoryRegistry::getRegistry(TestSet::perCommit()).makeTest());
        runner.addTest(CppUnit::TestFactoryRegistry::getRegistry(TestSet::perBuild()).makeTest());
        runner.addTest(CppUnit::TestFactoryRegistry::getRegistry(TestSet::perNightly()).makeTest());
        // listTests(rootTest, "");
        runner.setOutputter(new CppUnit::CompilerOutputter(&runner.result(), std::cerr));
        bool success = runner.run(path, false, true, false);
        return success ? 0 : 1;
    }
    catch (invalid_argument &e)
    {
        cerr << "\nERROR: " << e.what() << "\n";
        return 2;
    }
}
