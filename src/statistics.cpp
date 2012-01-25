/**
 * @file
 *
 * Classes for gathering and reporting statistics.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif

#include <string>
#include <stdexcept>
#include <cmath>
#include <boost/foreach.hpp>
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

std::ostream &operator<<(std::ostream &o, const Statistic &stat)
{
    boost::lock_guard<boost::mutex> _(stat.mutex);
    o << stat.getName() << ": ";
    stat.write(o);
    return o;
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
    boost::lock_guard<boost::mutex> _(mutex);
    total += incr;
}

unsigned long long Counter::getTotal() const
{
    return total;
}

Variable::Variable(const std::string &name) : Statistic(name), sum(0.0), sum2(0.0), n(0)
{
}

void Variable::add(double value)
{
    boost::lock_guard<boost::mutex> _(mutex);
    sum += value;
    sum2 += value * value;
    n++;
}

unsigned long long Variable::getNumSamples() const
{
    boost::lock_guard<boost::mutex> _(mutex);
    return n;
}

double Variable::getMean() const
{
    boost::lock_guard<boost::mutex> _(mutex);
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
    boost::lock_guard<boost::mutex> _(mutex);
    return getVarianceUnlocked();
}

void Variable::write(std::ostream &o) const
{
    if (n >= 1)
        o << sum / n << ' ';
    if (n >= 2)
        o << "+/- " << std::sqrt(getVarianceUnlocked()) << ' ';
    o << "[" << n << "]";
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

std::ostream &operator<<(std::ostream &o, const Registry &reg)
{
    boost::lock_guard<boost::mutex> _(reg.mutex);
    for (boost::ptr_map<std::string, Statistic>::const_iterator i = reg.statistics.begin(); i != reg.statistics.end(); ++i)
    {
        o << *i->second << '\n';
    }
    return o;
}

} // namespace Statistics
