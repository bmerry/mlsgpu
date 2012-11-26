/**
 * @file
 *
 * Record timing information.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif

#include <string>
#include <iostream>
#include <fstream>
#include <cerrno>
#include <cassert>
#include <boost/thread/mutex.hpp>
#include <boost/thread/locks.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/exception/all.hpp>
#include "timeplot.h"
#include "statistics.h"
#include "timer.h"
#include "errors.h"

namespace Timeplot
{

static bool hasFile = false;
static boost::mutex outputMutex;
static Timer::timestamp startTime = Timer::currentTime();
static std::ofstream log;

void init(const std::string &filename)
{
    MLSGPU_ASSERT(!hasFile, state_error);
    startTime = Timer::currentTime();
    try
    {
        log.open(filename.c_str());
        if (!log)
            throw std::ios::failure("Could not open timeplot file");
        log << std::fixed;
        log.precision(9);
        hasFile = true;
    }
    catch (std::ios::failure &e)
    {
        throw boost::enable_error_info(e)
            << boost::errinfo_file_name(filename)
            << boost::errinfo_errno(errno);
    }
}

Worker::Worker(const std::string &name) : name(name), currentAction(NULL)
{
}

Worker::Worker(const std::string &name, int idx)
    : name(name + "." + boost::lexical_cast<std::string>(idx)),
    currentAction(NULL)
{
}

Action *Worker::start(Action *current, Timer::timestamp time)
{
    Action *ret = currentAction;
    currentAction = current;
    if (ret != NULL)
        ret->pause(time);
    return ret;
}

void Worker::stop(Action *current, Action *prev, Timer::timestamp time)
{
    (void) current; // prevent warning in release builds
    MLSGPU_ASSERT(current == currentAction, state_error);
    currentAction = prev;
    if (currentAction != NULL)
        currentAction->resume(time);
}

const std::string &Worker::getName() const
{
    return name;
}

void Action::init()
{
    start = Timer::currentTime();
    running = true;
    elapsed = 0.0;
    oldAction = worker.start(this, start);
}

Action::Action(const std::string &name, Worker &worker)
    : name(name), worker(worker), stat(NULL)
{
    init();
}

Action::Action(const std::string &name, Worker &worker, Statistics::Variable &stat)
    : name(name), worker(worker), stat(&stat)
{
    init();
}

Action::Action(const std::string &name, Worker &worker, const std::string &statName)
    : name(name), worker(worker),
    stat(&Statistics::getStatistic<Statistics::Variable>(statName))
{
    init();
}

void Action::pause(Timer::timestamp time)
{
    MLSGPU_ASSERT(running, state_error);
    running = false;
    elapsed += Timer::getElapsed(start, time);

    if (hasFile)
    {
        boost::lock_guard<boost::mutex> lock(outputMutex);
        log << "EVENT " << worker.getName() << ' ' << name << ' '
            << Timer::getElapsed(startTime, start) << ' '
            << Timer::getElapsed(startTime, time) << '\n';
        if (value)
            log << "VALUE " << *value << '\n';
    }
}

void Action::resume(Timer::timestamp time)
{
    MLSGPU_ASSERT(!running, state_error);
    running = true;
    start = time;
}

void Action::setValue(std::size_t value)
{
    this->value = value;
}

Action::~Action()
{
    assert(running); // Can't MLSGPU_ASSERT it because destructors must not throw
    Timer::timestamp stop = Timer::currentTime();
    pause(stop);
    if (stat != NULL)
        stat->add(elapsed);
    worker.stop(this, oldAction, stop);
}

} // namespace Timeplot
