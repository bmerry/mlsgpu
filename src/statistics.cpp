/**
 * @file
 *
 * Classes for gathering and reporting statistics.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif

#ifndef __CL_ENABLE_EXCEPTIONS
# define __CL_ENABLE_EXCEPTIONS
#endif

#include <string>
#include <stdexcept>
#include <cmath>
#include <boost/foreach.hpp>
#include <CL/cl.hpp>
#include "statistics.h"
#include "logging.h"

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
        o << sum << " : " << sum / n << ' ';
    if (n >= 2)
        o << "+/- " << std::sqrt(getVarianceUnlocked()) << ' ';
    o << "[" << n << "]";
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

static void CL_CALLBACK timeEventCallback(cl_event event, cl_int event_command_exec_status, void *user_data)
{
    cl_int status;
    const cl_profiling_info fields[2] =
    {
        CL_PROFILING_COMMAND_START,
        CL_PROFILING_COMMAND_END
    };
    cl_ulong values[2];
    Statistics::Variable &stat = *static_cast<Statistics::Variable *>(user_data);

    if (event_command_exec_status != CL_COMPLETE)
    {
        Log::log[Log::warn] << "Warning: Event for " << stat.getName() << " did not complete successfully\n";
        return;
    }

    for (unsigned int i = 0; i < 2; i++)
    {
        status = clGetEventProfilingInfo(event, fields[i], sizeof(values[i]), &values[i], NULL);
        switch (status)
        {
        case CL_PROFILING_INFO_NOT_AVAILABLE:
            return;
        case CL_SUCCESS:
            break;
        default:
            Log::log[Log::warn] << "Warning: Could not extract profiling information for " << stat.getName() << '\n';
            return;
        }
    }

    double duration = 1e-9 * (values[1] - values[0]);
    if (duration > 1.0)
        Log::log[Log::debug] << values[0] << ' ' << values[1] << ' ' << duration << '\n';
    stat.add(duration);
}

void timeEvent(cl::Event event, Variable &stat)
{
    event.setCallback(CL_COMPLETE, timeEventCallback, &stat);
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
