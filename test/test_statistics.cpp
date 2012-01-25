/**
 * @file
 *
 * Test code for @ref statistics.h.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif

#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/extensions/HelperMacros.h>
#include <stdexcept>
#include <sstream>
#include <boost/foreach.hpp>
#include "../src/statistics.h"
#include "testmain.h"

/**
 * Test for the @ref Statistics::Statistic class.
 */
class TestStatistic : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestStatistic);
    CPPUNIT_TEST(testAdd);
    CPPUNIT_TEST(testGetMean);
    CPPUNIT_TEST(testGetStddev);
    CPPUNIT_TEST(testGetVariance);
    CPPUNIT_TEST(testGetNumSamples);
    CPPUNIT_TEST(testGetTotal);
    CPPUNIT_TEST(testGetName);
    CPPUNIT_TEST(testStream);
    CPPUNIT_TEST_SUITE_END();

    Statistics::Variable *stat0;          ///< Statistic with no samples
    Statistics::Variable *stat1;          ///< Statistic with one sample
    Statistics::Variable *stat2;          ///< Statistic with two samples
    Statistics::Variable *stat2s;         ///< Statistic with two identical samples
    Statistics::Counter *counter;         ///< Counter statistic

    void testAdd();            ///< Test @ref Statistics::Variable::add
    void testGetMean();        ///< Test @ref Statistics::Variable::getMean
    void testGetStddev();      ///< Test @ref Statistics::Variable::getStddev
    void testGetVariance();    ///< Test @ref Statistics::Variable::getVariance
    void testGetNumSamples();  ///< Test @ref Statistics::Variable::getNumSamples
    void testGetTotal();       ///< Test @ref Statistics::Counter::getTotal
    void testGetName();        ///< Test @ref Statistics::Statistic::getName
    void testStream();         ///< Test stream output of @ref Statistics::Statistic

public:
    void setUp();
    void tearDown();
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestStatistic, TestSet::perBuild());

void TestStatistic::setUp()
{
    stat0 = NULL; stat1 = NULL; stat2 = NULL; stat2s = NULL;
    stat0 = new Statistics::Variable("stat0");
    stat1 = new Statistics::Variable("stat1");
    stat2 = new Statistics::Variable("stat2");
    stat2s = new Statistics::Variable("stat2s");
    counter = new Statistics::Counter("counter");

    stat1->add(1.0);
    stat2->add(2.0);
    stat2->add(3.0);
    stat2s->add(4.5);
    stat2s->add(4.5);
    counter->add(100);
}

void TestStatistic::tearDown()
{
    delete stat0;
    delete stat1;
    delete stat2;
    delete stat2s;
    delete counter;
}

void TestStatistic::testAdd()
{
    // We test the add function by looking at the internal state of the fixtures
    CPPUNIT_ASSERT_EQUAL(1.0, stat1->sum);
    CPPUNIT_ASSERT_EQUAL(1.0, stat1->sum2);
    CPPUNIT_ASSERT_EQUAL(1ULL, stat1->n);

    CPPUNIT_ASSERT_EQUAL(5.0, stat2->sum);
    CPPUNIT_ASSERT_EQUAL(13.0, stat2->sum2);
    CPPUNIT_ASSERT_EQUAL(2ULL, stat2->n);

    CPPUNIT_ASSERT_EQUAL(9.0, stat2s->sum);
    CPPUNIT_ASSERT_EQUAL(40.5, stat2s->sum2);
    CPPUNIT_ASSERT_EQUAL(2ULL, stat2s->n);

    CPPUNIT_ASSERT_EQUAL(100ULL, counter->total);
}

void TestStatistic::testGetMean()
{
    CPPUNIT_ASSERT_THROW(stat0->getMean(), std::length_error);
    CPPUNIT_ASSERT_EQUAL(1.0, stat1->getMean());
    CPPUNIT_ASSERT_EQUAL(2.5, stat2->getMean());
    CPPUNIT_ASSERT_EQUAL(4.5, stat2s->getMean());
}

void TestStatistic::testGetVariance()
{
    CPPUNIT_ASSERT_THROW(stat0->getVariance(), std::length_error);
    CPPUNIT_ASSERT_THROW(stat1->getVariance(), std::length_error);
    CPPUNIT_ASSERT_EQUAL(0.5, stat2->getVariance());
    CPPUNIT_ASSERT_EQUAL(0.0, stat2s->getVariance());
}

void TestStatistic::testGetStddev()
{
    CPPUNIT_ASSERT_THROW(stat0->getStddev(), std::length_error);
    CPPUNIT_ASSERT_THROW(stat1->getStddev(), std::length_error);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.707106781186548, stat2->getStddev(), 1e-12);
    CPPUNIT_ASSERT_EQUAL(0.0, stat2s->getStddev());
}

void TestStatistic::testGetNumSamples()
{
    CPPUNIT_ASSERT_EQUAL(0ULL, stat0->getNumSamples());
    CPPUNIT_ASSERT_EQUAL(1ULL, stat1->getNumSamples());
    CPPUNIT_ASSERT_EQUAL(2ULL, stat2->getNumSamples());
    CPPUNIT_ASSERT_EQUAL(2ULL, stat2s->getNumSamples());
}

void TestStatistic::testGetTotal()
{
    CPPUNIT_ASSERT_EQUAL(100ULL, counter->getTotal());
}

void TestStatistic::testGetName()
{
    CPPUNIT_ASSERT_EQUAL(std::string("stat0"), stat0->getName());
    CPPUNIT_ASSERT_EQUAL(std::string("stat1"), stat1->getName());
    CPPUNIT_ASSERT_EQUAL(std::string("stat2"), stat2->getName());
    CPPUNIT_ASSERT_EQUAL(std::string("stat2s"), stat2s->getName());
    CPPUNIT_ASSERT_EQUAL(std::string("counter"), counter->getName());
}

void TestStatistic::testStream()
{
    {
        std::ostringstream s;
        s << *stat0;
        CPPUNIT_ASSERT_EQUAL(std::string("stat0: [0]"), s.str());
    }
    {
        std::ostringstream s;
        s << *stat1;
        CPPUNIT_ASSERT_EQUAL(std::string("stat1: 1 [1]"), s.str());
    }
    {
        std::ostringstream s;
        s.precision(6);
        s << *stat2;
        CPPUNIT_ASSERT_EQUAL(std::string("stat2: 2.5 +/- 0.707107 [2]"), s.str());
    }
    {
        std::ostringstream s;
        s.precision(6);
        s << *stat2s;
        CPPUNIT_ASSERT_EQUAL(std::string("stat2s: 4.5 +/- 0 [2]"), s.str());
    }
    {
        std::ostringstream s;
        s << *counter;
        CPPUNIT_ASSERT_EQUAL(std::string("counter: 100"), s.str());
    }
}


class TestStatisticRegistry : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestStatisticRegistry);
    CPPUNIT_TEST(testGetInstance);
    CPPUNIT_TEST(testGetStatistic);
    CPPUNIT_TEST(testStream);
    CPPUNIT_TEST(testIterate);
    CPPUNIT_TEST(testConstIterate);
    CPPUNIT_TEST_SUITE_END();
private:
    Statistics::Registry registry;

    void testGetInstance();       ///< Test @ref Statistics::Registry::getInstance
    void testGetStatistic();      ///< Test @ref Statistics::Registry::getStatistic
    void testStream();            ///< Test ostream output
    void testIterate();           ///< Test iteration over a non-const registry
    void testConstIterate();      ///< Test iteration over a const registry

public:
    void setUp();
    void tearDown();
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestStatisticRegistry, TestSet::perBuild());

void TestStatisticRegistry::setUp()
{
    registry.getStatistic<Statistics::Variable>("stat0");

    Statistics::Variable &stat1 = registry.getStatistic<Statistics::Variable>("stat1");
    stat1.add(1.0);

    Statistics::Variable &stat3 = registry.getStatistic<Statistics::Variable>("stat3");
    stat3.add(2.0);
    stat3.add(4.0);
    stat3.add(6.0);

    Statistics::Counter &counter = registry.getStatistic<Statistics::Counter>("counter");
    counter.add(100);
}

void TestStatisticRegistry::tearDown()
{
}

void TestStatisticRegistry::testGetInstance()
{
    Statistics::Registry &reg = Statistics::Registry::getInstance();
    CPPUNIT_ASSERT(&reg != NULL);

    Statistics::Registry &reg2 = Statistics::Registry::getInstance();
    CPPUNIT_ASSERT_MESSAGE("Singleton reference should not move", &reg == &reg2);
}

void TestStatisticRegistry::testGetStatistic()
{
    // Get a known statistic
    Statistics::Variable &stat1 = registry.getStatistic<Statistics::Variable>("stat1");
    CPPUNIT_ASSERT_EQUAL(1ULL, stat1.getNumSamples());

    // Get a new statistic
    Statistics::Variable &n = registry.getStatistic<Statistics::Variable>("new");
    CPPUNIT_ASSERT_EQUAL(0ULL, n.getNumSamples());

    // Type mismatch on known statistic
    CPPUNIT_ASSERT_THROW(registry.getStatistic<Statistics::Variable>("counter"), std::bad_cast);
}

void TestStatisticRegistry::testStream()
{
    std::ostringstream s;
    s << registry;
    CPPUNIT_ASSERT_EQUAL(std::string(
            "counter: 100\n"
            "stat0: [0]\n"
            "stat1: 1 [1]\n"
            "stat3: 4 +/- 2 [3]\n"), s.str());
}

void TestStatisticRegistry::testIterate()
{
    BOOST_FOREACH(Statistics::Statistic &s, registry)
    {
        if (typeid(s) == typeid(Statistics::Variable))
        {
            dynamic_cast<Statistics::Variable &>(s).add(1.0);
        }
        else if (typeid(s) == typeid(Statistics::Counter))
        {
            dynamic_cast<Statistics::Counter &>(s).add(1);
        }
    }
    // Check that each statistic had 1 added to it

    Statistics::Variable &stat0 = registry.getStatistic<Statistics::Variable>("stat0");
    Statistics::Variable &stat1 = registry.getStatistic<Statistics::Variable>("stat1");
    Statistics::Variable &stat3 = registry.getStatistic<Statistics::Variable>("stat3");
    Statistics::Counter &counter = registry.getStatistic<Statistics::Counter>("counter");
    CPPUNIT_ASSERT_EQUAL(1ULL, stat0.getNumSamples());
    CPPUNIT_ASSERT_EQUAL(2ULL, stat1.getNumSamples());
    CPPUNIT_ASSERT_EQUAL(4ULL, stat3.getNumSamples());
    CPPUNIT_ASSERT_EQUAL(101ULL, counter.getTotal());
}

void TestStatisticRegistry::testConstIterate()
{
    const Statistics::Statistic &counter = registry.getStatistic<Statistics::Counter>("counter");
    const Statistics::Statistic &stat0 = registry.getStatistic<Statistics::Variable>("stat0");
    const Statistics::Statistic &stat1 = registry.getStatistic<Statistics::Variable>("stat1");
    const Statistics::Statistic &stat3 = registry.getStatistic<Statistics::Variable>("stat3");

    Statistics::Registry::const_iterator cur = registry.begin();
    CPPUNIT_ASSERT_EQUAL(&*cur, &counter);
    cur++;
    CPPUNIT_ASSERT_EQUAL(&*cur, &stat0);
    cur++;
    CPPUNIT_ASSERT_EQUAL(&*cur, &stat1);
    cur++;
    CPPUNIT_ASSERT_EQUAL(&*cur, &stat3);
    cur++;
    CPPUNIT_ASSERT(cur == registry.end());
}
