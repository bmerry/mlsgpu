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
#include <typeinfo>
#include <boost/foreach.hpp>
#include <boost/smart_ptr/scoped_ptr.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/scoped_ptr.hpp>
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
    CPPUNIT_TEST(testStream);
    CPPUNIT_TEST(testSerialize);
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
    void testSerialize();      ///< Test that serialization and deserialization works

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
    CPPUNIT_TEST(testSerialize);
    CPPUNIT_TEST_SUITE_END();

private:
    boost::scoped_ptr<Statistics::Counter> counter;   ///< Counter statistic

    void testAdd();            ///< Test @ref Statistics::Counter::add
    void testGetTotal();       ///< Test @ref Statistics::Counter::getTotal
    void testStream();         ///< Test stream output of @ref Statistics::Counter
    void testSerialize();      ///< Test that serialization works

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
    CPPUNIT_TEST(testSerialize);
    CPPUNIT_TEST_SUITE_END();

private:
    /**
     * Peak statistic fixture. It is initialized to the value -100, with the
     * maximum being left at 0.
     */
    boost::scoped_ptr<Statistics::Peak> peak;

    void testAdd();      ///< Test Statistics::Peak::operator+=()
    void testSub();      ///< Test Statistics::Peak::operator-=()
    void testSet();      ///< Test Statistics::Peak::operator=()
    void testGet();      ///< Test @ref Statistics::Peak::get
    void testGetMax();   ///< Test @ref Statistics::Peak::getMax
    void testStream();   ///< Test streaming a @ref Statistics::Peak to an @c ostream
    void testEmpty();    ///< Test initial state
    void testSerialize(); ///< Test that serialization works

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
#if DEBUG
    CPPUNIT_ASSERT_THROW(stat0->getMean(), std::length_error);
#endif
    CPPUNIT_ASSERT_EQUAL(1.0, stat1->getMean());
    CPPUNIT_ASSERT_EQUAL(2.5, stat2->getMean());
    CPPUNIT_ASSERT_EQUAL(4.5, stat2s->getMean());
}

void TestVariable::testGetVariance()
{
#if DEBUG
    CPPUNIT_ASSERT_THROW(stat0->getVariance(), std::length_error);
    CPPUNIT_ASSERT_THROW(stat1->getVariance(), std::length_error);
#endif
    CPPUNIT_ASSERT_EQUAL(0.5, stat2->getVariance());
    CPPUNIT_ASSERT_EQUAL(0.0, stat2s->getVariance());
}

void TestVariable::testGetStddev()
{
#if DEBUG
    CPPUNIT_ASSERT_THROW(stat0->getStddev(), std::length_error);
    CPPUNIT_ASSERT_THROW(stat1->getStddev(), std::length_error);
#endif
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
        CPPUNIT_ASSERT_EQUAL(std::string("stat1: 1 : 1 [1]"), s.str());
    }
    {
        std::ostringstream s;
        s.precision(6);
        s << *stat2;
        CPPUNIT_ASSERT_EQUAL(std::string("stat2: 5 : 2.5 +/- 0.707107 [2]"), s.str());
    }
    {
        std::ostringstream s;
        s.precision(6);
        s << *stat2s;
        CPPUNIT_ASSERT_EQUAL(std::string("stat2s: 9 : 4.5 +/- 0 [2]"), s.str());
    }
}

void TestVariable::testSerialize()
{
    std::stringstream s;
    boost::archive::text_oarchive oa(s);
    Statistics::Statistic *oldPtr = stat2.get();
    oa << oldPtr;

    boost::archive::text_iarchive ia(s);
    Statistics::Statistic *newPtr;
    ia >> newPtr;
    boost::scoped_ptr<Statistics::Statistic> save(newPtr);

    Statistics::Variable *newStat = dynamic_cast<Statistics::Variable *>(newPtr);
    CPPUNIT_ASSERT(newPtr != NULL);
    CPPUNIT_ASSERT_EQUAL(stat2->sum, newStat->sum);
    CPPUNIT_ASSERT_EQUAL(stat2->sum2, newStat->sum2);
    CPPUNIT_ASSERT_EQUAL(stat2->n, newStat->n);
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
    MLSGPU_ASSERT_EQUAL(100, counter->total);
    counter->add(50);
    MLSGPU_ASSERT_EQUAL(150, counter->total);
}

void TestCounter::testGetTotal()
{
    MLSGPU_ASSERT_EQUAL(100, counter->getTotal());
}

void TestCounter::testStream()
{
    std::ostringstream s;
    s << *counter;
    CPPUNIT_ASSERT_EQUAL(std::string("counter: 100"), s.str());
}

void TestCounter::testSerialize()
{
    std::stringstream s;
    boost::archive::text_oarchive oa(s);
    Statistics::Statistic *oldPtr = counter.get();
    oa << oldPtr;

    boost::archive::text_iarchive ia(s);
    Statistics::Statistic *newPtr;
    ia >> newPtr;
    boost::scoped_ptr<Statistics::Statistic> save(newPtr);

    Statistics::Counter *newStat = dynamic_cast<Statistics::Counter *>(newPtr);
    CPPUNIT_ASSERT(newStat != NULL);
    CPPUNIT_ASSERT_EQUAL(counter->total, newStat->total);
}

Statistics::Statistic *TestCounter::createStatistic(const std::string &name) const
{
    return new Statistics::Counter(name);
}

void TestPeak::setUp()
{
    peak.reset(new Statistics::Peak("peak"));
    *peak = -100LL;
}

void TestPeak::testSet()
{
    // Test initial state
    MLSGPU_ASSERT_EQUAL(-100, peak->current);
    MLSGPU_ASSERT_EQUAL(0, peak->peak);

    // Test setting a maximal value
    *peak = 1234567890;
    MLSGPU_ASSERT_EQUAL(1234567890, peak->current);
    MLSGPU_ASSERT_EQUAL(1234567890, peak->peak);

    // Test setting a non-maximal value
    *peak = 123456;
    MLSGPU_ASSERT_EQUAL(123456, peak->current);
    MLSGPU_ASSERT_EQUAL(1234567890, peak->peak);
}

void TestPeak::testAdd()
{
    // Go up
    *peak += 250;
    MLSGPU_ASSERT_EQUAL(150, peak->current);
    MLSGPU_ASSERT_EQUAL(150, peak->peak);

    // Go down
    *peak += -200;
    MLSGPU_ASSERT_EQUAL(-50, peak->current);
    MLSGPU_ASSERT_EQUAL(150, peak->peak);
}

void TestPeak::testSub()
{
    // Go up
    *peak -= -250;
    MLSGPU_ASSERT_EQUAL(150, peak->current);
    MLSGPU_ASSERT_EQUAL(150, peak->peak);

    // Go down
    *peak -= 200;
    MLSGPU_ASSERT_EQUAL(-50, peak->current);
    MLSGPU_ASSERT_EQUAL(150, peak->peak);
}

void TestPeak::testGet()
{
    MLSGPU_ASSERT_EQUAL(-100, peak->get());
    *peak = -200;
    MLSGPU_ASSERT_EQUAL(-200, peak->get());
}

void TestPeak::testGetMax()
{
    MLSGPU_ASSERT_EQUAL(0, peak->getMax());
    *peak = -200;
    MLSGPU_ASSERT_EQUAL(0, peak->getMax());
    *peak = 200;
    MLSGPU_ASSERT_EQUAL(200, peak->getMax());
}

void TestPeak::testStream()
{
    std::ostringstream o;
    *peak = 123;
    o << *peak;
    CPPUNIT_ASSERT_EQUAL(std::string("peak: 123"), o.str());

    Statistics::Peak empty("empty");
    o.str("");
    o << empty;
    CPPUNIT_ASSERT_EQUAL(std::string("empty: 0"), o.str());
}

void TestPeak::testEmpty()
{
    Statistics::Peak empty("empty");
    MLSGPU_ASSERT_EQUAL(0, empty.get());
    MLSGPU_ASSERT_EQUAL(0, empty.getMax());
}

void TestPeak::testSerialize()
{
    *peak = 234;

    std::stringstream s;
    boost::archive::text_oarchive oa(s);
    Statistics::Statistic *oldPtr = peak.get();
    oa << oldPtr;

    boost::archive::text_iarchive ia(s);
    Statistics::Statistic *newPtr;
    ia >> newPtr;
    boost::scoped_ptr<Statistics::Statistic> save(newPtr);

    Statistics::Peak *newStat = dynamic_cast<Statistics::Peak *>(newPtr);
    CPPUNIT_ASSERT(newStat != NULL);
    CPPUNIT_ASSERT_EQUAL(peak->peak, newStat->peak);
}

Statistics::Statistic *TestPeak::createStatistic(const std::string &name) const
{
    return new Statistics::Peak(name);
}

class TestStatisticsRegistry : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestStatisticsRegistry);
    CPPUNIT_TEST(testGetInstance);
    CPPUNIT_TEST(testGetStatistic);
    CPPUNIT_TEST(testStream);
    CPPUNIT_TEST(testIterate);
    CPPUNIT_TEST(testConstIterate);
    CPPUNIT_TEST(testSerialize);
    CPPUNIT_TEST(testMerge);
    CPPUNIT_TEST_SUITE_END();
private:
    Statistics::Registry registry;

    void testGetInstance();       ///< Test @ref Statistics::Registry::getInstance
    void testGetStatistic();      ///< Test @ref Statistics::Registry::getStatistic
    void testStream();            ///< Test ostream output
    void testIterate();           ///< Test iteration over a non-const registry
    void testConstIterate();      ///< Test iteration over a const registry
    void testSerialize();         ///< Test serialization and deserialization
    void testMerge();             ///< Test @ref Statistics::Registry::merge

public:
    void setUp();
    void tearDown();
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestStatisticsRegistry, TestSet::perBuild());

void TestStatisticsRegistry::setUp()
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

void TestStatisticsRegistry::tearDown()
{
}

void TestStatisticsRegistry::testGetInstance()
{
    Statistics::Registry &reg = Statistics::Registry::getInstance();
    CPPUNIT_ASSERT(&reg != NULL);

    Statistics::Registry &reg2 = Statistics::Registry::getInstance();
    CPPUNIT_ASSERT_MESSAGE("Singleton reference should not move", &reg == &reg2);
}

void TestStatisticsRegistry::testGetStatistic()
{
    // Get a known statistic
    Statistics::Variable &stat1 = registry.getStatistic<Statistics::Variable>("stat1");
    CPPUNIT_ASSERT_EQUAL(1ULL, stat1.getNumSamples());

    // Get a new statistic
    Statistics::Variable &n = registry.getStatistic<Statistics::Variable>("new");
    CPPUNIT_ASSERT_EQUAL(0ULL, n.getNumSamples());

#if DEBUG
    // Type mismatch on known statistic
    CPPUNIT_ASSERT_THROW(registry.getStatistic<Statistics::Variable>("counter"), std::bad_cast);
#endif
}

void TestStatisticsRegistry::testStream()
{
    std::ostringstream s;
    s << registry;
    CPPUNIT_ASSERT_EQUAL(std::string(
            "counter: 100\n"
            "stat0: [0]\n"
            "stat1: 1 : 1 [1]\n"
            "stat3: 12 : 4 +/- 2 [3]\n"), s.str());
}

void TestStatisticsRegistry::testIterate()
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

void TestStatisticsRegistry::testConstIterate()
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

void TestStatisticsRegistry::testSerialize()
{
    std::stringstream s;
    boost::archive::text_oarchive oa(s);
    oa << registry;

    boost::archive::text_iarchive ia(s);
    Statistics::Registry newRegistry;
    ia >> newRegistry;

    std::ostringstream oldStr, newStr;
    oldStr << registry;
    newStr << newRegistry;

    CPPUNIT_ASSERT_EQUAL(oldStr.str(), newStr.str());
}

void TestStatisticsRegistry::testMerge()
{
    Statistics::Registry other;

    Statistics::Variable &stat0 = registry.getStatistic<Statistics::Variable>("stat0");
    stat0.add(12.0);
    Statistics::Counter &counter = registry.getStatistic<Statistics::Counter>("counter");
    counter.add(17);
    Statistics::Variable &extra = registry.getStatistic<Statistics::Variable>("extra");
    extra.add(3.0);

    registry.merge(other);
    std::ostringstream s;
    s << registry;
    CPPUNIT_ASSERT_EQUAL(std::string(
            "counter: 117\n"
            "extra: 3 : 3 [1]\n"
            "stat0: 12 : 12 [1]\n"
            "stat1: 1 : 1 [1]\n"
            "stat3: 12 : 4 +/- 2 [3]\n"), s.str());
}
