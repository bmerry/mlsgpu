/**
 * @file
 *
 * Classes for gathering and reporting statistics.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif

/* Do not remove these! There are needed to ensure that BOOST_CLASS_EXPORT_IMPLEMENT
 * does the right things.
 */
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <string>
#include <stdexcept>
#include <cmath>
#include <vector>
#include <utility>
#include <queue>
#include <sstream>
#include <boost/foreach.hpp>
#include <boost/thread/locks.hpp>
#include <boost/ptr_container/serialize_ptr_map.hpp>
#include "statistics.h"

namespace Statistics
{

Statistic::Statistic(const std::string &name) : name(name)
{
}

Statistic::~Statistic()
{
}

const std::string &Statistic::getName() const
{
    return name;
}

Statistic *Statistic::clone() const
{
    boost::lock_guard<boost::mutex> lock(mutex);

    std::stringstream s;
    boost::archive::text_oarchive oa(s);
    const Statistic *thisPtr = this; // clang wants an lvalue
    oa << thisPtr;

    Statistic *cloned = NULL;
    boost::archive::text_iarchive ia(s);
    ia >> cloned;
    return cloned;
}

std::ostream &operator<<(std::ostream &o, const Statistic &stat)
{
    boost::lock_guard<boost::mutex> lock(stat.mutex);
    o << stat.getName() << ": ";
    stat.write(o);
    return o;
}

template<typename Archive>
void Statistic::serialize(Archive &ar, const unsigned int)
{
    ar & name;
}


Counter::Counter(const std::string &name) : Statistic(name), total(0)
{
}

void Counter::write(std::ostream &o) const
{
    o << total;
}

void Counter::add(unsigned long long incr)
{
    boost::lock_guard<boost::mutex> lock(mutex);
    total += incr;
}

unsigned long long Counter::getTotal() const
{
    return total;
}

void Counter::merge(const Statistic &other)
{
    const Counter &stat = dynamic_cast<const Counter &>(other);
    total += stat.total;
}

template<typename Archive>
void Counter::serialize(Archive &ar, const unsigned int)
{
    ar & boost::serialization::base_object<Statistic>(*this);
    ar & total;
}


Variable::Variable(const std::string &name) : Statistic(name), sum(0.0), sum2(0.0), n(0)
{
}

void Variable::add(double value)
{
    boost::lock_guard<boost::mutex> lock(mutex);
    sum += value;
    sum2 += value * value;
    n++;
}

unsigned long long Variable::getNumSamples() const
{
    boost::lock_guard<boost::mutex> lock(mutex);
    return n;
}

double Variable::getMean() const
{
    boost::lock_guard<boost::mutex> lock(mutex);
    if (n < 1)
        throw std::length_error("Cannot compute mean without at least 1 sample");
    return sum / n;
}

double Variable::getStddev() const
{
    return std::sqrt(getVariance());
}

double Variable::getVarianceUnlocked() const
{
    if (n < 2)
        throw std::length_error("Cannot compute variance without at least 2 samples");
    // Theoretically the variable must be non-negative, but rounding errors
    // could make it negative, leading to problems when computing stddev.
    return std::max((sum2 - sum * sum / n) / (n - 1), 0.0);
}

double Variable::getVariance() const
{
    boost::lock_guard<boost::mutex> lock(mutex);
    return getVarianceUnlocked();
}

void Variable::write(std::ostream &o) const
{
    if (n >= 1)
        o << sum << " : " << sum / n << ' ';
    if (n >= 2)
        o << "+/- " << std::sqrt(getVarianceUnlocked()) << ' ';
    o << "[" << n << "]";
}

void Variable::merge(const Statistic &other)
{
    const Variable &stat = dynamic_cast<const Variable &>(other);
    sum += stat.sum;
    sum2 += stat.sum2;
    n += stat.n;
}

template<typename Archive>
void Variable::serialize(Archive &ar, const unsigned int)
{
    ar & boost::serialization::base_object<Statistic>(*this);
    ar & sum;
    ar & sum2;
    ar & n;
}


Peak::Peak(const std::string &name) : Statistic(name), current(0), peak (0)
{
}

void Peak::write(std::ostream &o) const
{
    o << peak;
}

void Peak::set(value_type x)
{
    current = x;
    if (peak < current)
        peak = current;
}

Peak &Peak::operator+=(value_type x)
{
    boost::lock_guard<boost::mutex> lock(mutex);
    set(current + x);
    return *this;
}

Peak &Peak::operator-=(value_type x)
{
    boost::lock_guard<boost::mutex> lock(mutex);
    set(current - x);
    return *this;
}

Peak &Peak::operator=(value_type x)
{
    boost::lock_guard<boost::mutex> lock(mutex);
    set(x);
    return *this;
}

Peak::value_type Peak::get() const
{
    boost::lock_guard<boost::mutex> lock(mutex);
    return current;
}

/// Retrieves the highest value that has been set.
Peak::value_type Peak::getMax() const
{
    boost::lock_guard<boost::mutex> lock(mutex);
    return peak;
}

void Peak::merge(const Statistic &other)
{
    const Peak &stat = dynamic_cast<const Peak &>(other);
    if (stat.peak > peak)
        peak = stat.peak;
}

template<typename Archive>
void Peak::serialize(Archive &ar, const unsigned int)
{
    ar & boost::serialization::base_object<Statistic>(*this);
    ar & current;
    ar & peak;
}


Timer::Timer(const std::string &name)
    : stat(getStatistic<Variable>(name))
{
}

Timer::Timer(Variable &stat)
    : stat(stat)
{
}

Timer::~Timer()
{
    stat.add(getElapsed());
}


Registry::Registry() : mutex()
{
}

Registry::~Registry()
{
}

Registry &Registry::getInstance()
{
    static Registry singleton;
    return singleton;
}

Registry::iterator Registry::begin()
{
    return iterator(statistics.begin());
}

Registry::iterator Registry::end()
{
    return iterator(statistics.end());
}

Registry::const_iterator Registry::begin() const
{
    return const_iterator(statistics.begin());
}

Registry::const_iterator Registry::end() const
{
    return const_iterator(statistics.end());
}

template<typename Archive>
void Registry::serialize(Archive &ar, const unsigned int)
{
    ar & statistics;
}

void Registry::merge(const Registry &other)
{
    for (const_iterator i = other.begin(); i != other.end(); i++)
    {
        boost::ptr_map<std::string, Statistic>::iterator pos = statistics.find(i->getName());
        if (pos == statistics.end())
        {
            Statistic *clone = i->clone();
            std::string name = clone->getName();
            statistics.insert(name, clone);
        }
        else
        {
            pos->second->merge(*i);
        }
    }
}

std::ostream &operator<<(std::ostream &o, const Registry &reg)
{
    boost::lock_guard<boost::mutex> _(reg.mutex);
    for (boost::ptr_map<std::string, Statistic>::const_iterator i = reg.statistics.begin(); i != reg.statistics.end(); ++i)
    {
        o << *i->second << '\n';
    }
    return o;
}

/* Explicitly instantiate the template member functions for text archives.
 * This is necessary to ensure that they exist at runtime, because they're only
 * implemented inside this file.
 */
template void Statistic::serialize(boost::archive::text_oarchive &ar, const unsigned int version);
template void Statistic::serialize(boost::archive::text_iarchive &ar, const unsigned int version);
template void Counter::serialize(boost::archive::text_oarchive &ar, const unsigned int version);
template void Counter::serialize(boost::archive::text_iarchive &ar, const unsigned int version);
template void Variable::serialize(boost::archive::text_oarchive &ar, const unsigned int version);
template void Variable::serialize(boost::archive::text_iarchive &ar, const unsigned int version);
template void Peak::serialize(boost::archive::text_oarchive &ar, const unsigned int version);
template void Peak::serialize(boost::archive::text_iarchive &ar, const unsigned int version);
template void Registry::serialize(boost::archive::text_oarchive &ar, const unsigned int version);
template void Registry::serialize(boost::archive::text_iarchive &ar, const unsigned int version);

} // namespace Statistics

BOOST_CLASS_EXPORT_IMPLEMENT(Statistics::Statistic)
BOOST_CLASS_EXPORT_IMPLEMENT(Statistics::Variable)
BOOST_CLASS_EXPORT_IMPLEMENT(Statistics::Counter)
BOOST_CLASS_EXPORT_IMPLEMENT(Statistics::Peak)
BOOST_CLASS_EXPORT_IMPLEMENT(Statistics::Registry)
