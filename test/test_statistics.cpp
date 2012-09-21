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
#include <string>
#include <boost/foreach.hpp>
#include <boost/smart_ptr/scoped_ptr.hpp>
#include "../src/statistics.h"
#include "testutil.h"

/**
 * Test for the @ref Statistics::Statistic base class.
 */
class TestStatistic : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestStatistic);
    CPPUNIT_TEST(testGetName);
    CPPUNIT_TEST_SUITE_END_ABSTRACT();

private:
    void testGetName();        ///< Test @ref Statistics::Statistic::getName

protected:
    /// Create a statistic with the given name
    virtual Statistics::Statistic *createStatistic(const std::string &name) const = 0;
};

void TestStatistic::testGetName()
{
    boost::scoped_ptr<Statistics::Statistic> stat(createStatistic("myname"));
    CPPUNIT_ASSERT_EQUAL(std::string("myname"), stat->getName());
}


/// Tests for @ref Statistics::Variable
class TestVariable : public TestStatistic
{
    CPPUNIT_TEST_SUB_SUITE(TestVariable, TestStatistic);
    CPPUNIT_TEST(testAdd);
    CPPUNIT_TEST(testGetMean);
    CPPUNIT_TEST(testGetStddev);
    CPPUNIT_TEST(testGetVariance);
    CPPUNIT_TEST(testGetNumSamples);
    CPPUNIT_TEST_SUITE_END();

private:
    boost::scoped_ptr<Statistics::Variable> stat0;   ///< Statistic with no samples
    boost::scoped_ptr<Statistics::Variable> stat1;   ///< Statistic with one sample
    boost::scoped_ptr<Statistics::Variable> stat2;   ///< Statistic with two samples
    boost::scoped_ptr<Statistics::Variable> stat2s;  ///< Statistic with two identical samples

    void testAdd();            ///< Test @ref Statistics::Variable::add
    void testGetMean();        ///< Test @ref Statistics::Variable::getMean
    void testGetStddev();      ///< Test @ref Statistics::Variable::getStddev
    void testGetVariance();    ///< Test @ref Statistics::Variable::getVariance
    void testGetNumSamples();  ///< Test @ref Statistics::Variable::getNumSamples
    void testStream();         ///< Test stream output of @ref Statistics::Variable

protected:
    virtual Statistics::Statistic *createStatistic(const std::string &name) const;

public:
    virtual void setUp();
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestVariable, TestSet::perBuild());

/// Tests for @ref Statistics::Counter
class TestCounter : public TestStatistic
{
    CPPUNIT_TEST_SUB_SUITE(TestCounter, TestStatistic);
    CPPUNIT_TEST(testAdd);
    CPPUNIT_TEST(testGetTotal);
    CPPUNIT_TEST(testStream);
    CPPUNIT_TEST_SUITE_END();

private:
    boost::scoped_ptr<Statistics::Counter> counter;   ///< Counter statistic

    void testAdd();            ///< Test @ref Statistics::Counter::add
    void testGetTotal();       ///< Test @ref Statistics::Counter::getTotal
    void testStream();         ///< Test stream output of @ref Statistics::Counter

protected:
    virtual Statistics::Statistic *createStatistic(const std::string &name) const;

public:
    virtual void setUp();
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestCounter, TestSet::perBuild());

/// Tests for the @ref Statistics::Peak
class TestPeak : public TestStatistic
{
    CPPUNIT_TEST_SUB_SUITE(TestPeak, TestStatistic);
    CPPUNIT_TEST(testAdd);
    CPPUNIT_TEST(testSub);
    CPPUNIT_TEST(testSet);
    CPPUNIT_TEST(testGet);
    CPPUNIT_TEST(testGetMax);
    CPPUNIT_TEST(testStream);
    CPPUNIT_TEST(testEmpty);
    CPPUNIT_TEST_SUITE_END();

private:
    /**
     * Peak statistic fixture. It is initialized to the value -100, with the
     * maximum being left at 0.
     */
    boost::scoped_ptr<Statistics::Peak<long long> > peak;

    void testAdd();      ///< Test Statistics::Peak::operator+=()
    void testSub();      ///< Test Statistics::Peak::operator-=()
    void testSet();      ///< Test Statistics::Peak::operator=()
    void testGet();      ///< Test @ref Statistics::Peak::get
    void testGetMax();   ///< Test @ref Statistics::Peak::getMax
    void testStream();   ///< Test streaming a @ref Statistics::Peak to an @c ostream
    void testEmpty();    ///< Test initial state

protected:
    virtual Statistics::Statistic *createStatistic(const std::string &name) const;

public:
    virtual void setUp();
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestPeak, TestSet::perBuild());

void TestVariable::setUp()
{
    stat0.reset(new Statistics::Variable("stat0"));
    stat1.reset(new Statistics::Variable("stat1"));
    stat2.reset(new Statistics::Variable("stat2"));
    stat2s.reset(new Statistics::Variable("stat2s"));

    stat1->add(1.0);
    stat2->add(2.0);
    stat2->add(3.0);
    stat2s->add(4.5);
    stat2s->add(4.5);
}

void TestVariable::testAdd()
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
}

void TestVariable::testGetMean()
{
    CPPUNIT_ASSERT_THROW(stat0->getMean(), std::length_error);
    CPPUNIT_ASSERT_EQUAL(1.0, stat1->getMean());
    CPPUNIT_ASSERT_EQUAL(2.5, stat2->getMean());
    CPPUNIT_ASSERT_EQUAL(4.5, stat2s->getMean());
}

void TestVariable::testGetVariance()
{
    CPPUNIT_ASSERT_THROW(stat0->getVariance(), std::length_error);
    CPPUNIT_ASSERT_THROW(stat1->getVariance(), std::length_error);
    CPPUNIT_ASSERT_EQUAL(0.5, stat2->getVariance());
    CPPUNIT_ASSERT_EQUAL(0.0, stat2s->getVariance());
}

void TestVariable::testGetStddev()
{
    CPPUNIT_ASSERT_THROW(stat0->getStddev(), std::length_error);
    CPPUNIT_ASSERT_THROW(stat1->getStddev(), std::length_error);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.707106781186548, stat2->getStddev(), 1e-12);
    CPPUNIT_ASSERT_EQUAL(0.0, stat2s->getStddev());
}

void TestVariable::testGetNumSamples()
{
    CPPUNIT_ASSERT_EQUAL(0ULL, stat0->getNumSamples());
    CPPUNIT_ASSERT_EQUAL(1ULL, stat1->getNumSamples());
    CPPUNIT_ASSERT_EQUAL(2ULL, stat2->getNumSamples());
    CPPUNIT_ASSERT_EQUAL(2ULL, stat2s->getNumSamples());
}

void TestVariable::testStream()
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
}

Statistics::Statistic *TestVariable::createStatistic(const std::string &name) const
{
    return new Statistics::Variable(name);
}

void TestCounter::setUp()
{
    counter.reset(new Statistics::Counter("counter"));
    counter->add(100);
}

void TestCounter::testAdd()
{
    CPPUNIT_ASSERT_EQUAL(100ULL, counter->total);
    counter->add(50);
    CPPUNIT_ASSERT_EQUAL(150ULL, counter->total);
}

void TestCounter::testGetTotal()
{
    CPPUNIT_ASSERT_EQUAL(100ULL, counter->getTotal());
}

void TestCounter::testStream()
{
    std::ostringstream s;
    s << *counter;
    CPPUNIT_ASSERT_EQUAL(std::string("counter: 100"), s.str());
}

Statistics::Statistic *TestCounter::createStatistic(const std::string &name) const
{
    return new Statistics::Counter(name);
}

void TestPeak::setUp()
{
    peak.reset(new Statistics::Peak<long long>("peak"));
    *peak = -100LL;
}

void TestPeak::testSet()
{
    // Test initial state
    CPPUNIT_ASSERT_EQUAL(-100LL, peak->current);
    CPPUNIT_ASSERT_EQUAL(0LL, peak->peak);

    // Test setting a maximal value
    *peak = 1234567890LL;
    CPPUNIT_ASSERT_EQUAL(1234567890LL, peak->current);
    CPPUNIT_ASSERT_EQUAL(1234567890LL, peak->peak);

    // Test setting a non-maximal value
    *peak = 123456LL;
    CPPUNIT_ASSERT_EQUAL(123456LL, peak->current);
    CPPUNIT_ASSERT_EQUAL(1234567890LL, peak->peak);
}

void TestPeak::testAdd()
{
    // Go up
    *peak += 250LL;
    CPPUNIT_ASSERT_EQUAL(150LL, peak->current);
    CPPUNIT_ASSERT_EQUAL(150LL, peak->peak);

    // Go down
    *peak += -200LL;
    CPPUNIT_ASSERT_EQUAL(-50LL, peak->current);
    CPPUNIT_ASSERT_EQUAL(150LL, peak->peak);
}

void TestPeak::testSub()
{
    // Go up
    *peak -= -250LL;
    CPPUNIT_ASSERT_EQUAL(150LL, peak->current);
    CPPUNIT_ASSERT_EQUAL(150LL, peak->peak);

    // Go down
    *peak -= 200LL;
    CPPUNIT_ASSERT_EQUAL(-50LL, peak->current);
    CPPUNIT_ASSERT_EQUAL(150LL, peak->peak);
}

void TestPeak::testGet()
{
    CPPUNIT_ASSERT_EQUAL(-100LL, peak->get());
    peak->set(-200LL);
    CPPUNIT_ASSERT_EQUAL(-200LL, peak->get());
}

void TestPeak::testGetMax()
{
    CPPUNIT_ASSERT_EQUAL(0LL, peak->getMax());
    peak->set(-200LL);
    CPPUNIT_ASSERT_EQUAL(0LL, peak->getMax());
    peak->set(200LL);
    CPPUNIT_ASSERT_EQUAL(200LL, peak->getMax());
}

void TestPeak::testStream()
{
    std::ostringstream o;
    *peak = 123LL;
    o << *peak;
    CPPUNIT_ASSERT_EQUAL(std::string("peak: 123"), o.str());

    Statistics::Peak<int> empty("empty");
    o.str("");
    o << empty;
    CPPUNIT_ASSERT_EQUAL(std::string("empty: 0"), o.str());
}

void TestPeak::testEmpty()
{
    Statistics::Peak<int> empty("empty");
    CPPUNIT_ASSERT_EQUAL(0, empty.get());
    CPPUNIT_ASSERT_EQUAL(0, empty.getMax());
}

Statistics::Statistic *TestPeak::createStatistic(const std::string &name) const
{
    return new Statistics::Peak<int>(name);
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
            "stat1: 1 : 1 [1]\n"
            "stat3: 12 : 4 +/- 2 [3]\n"), s.str());
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
