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

Action::Action(const std::string &name, Worker &worker, Statistics::Variable *stat)
    : name(name), worker(worker), stat(stat)
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

void recordEvent(const std::string &name, Worker &worker)
{
    if (hasFile)
    {
        boost::lock_guard<boost::mutex> lock(outputMutex);
        Timer::timestamp now = Timer::currentTime();
        double t = Timer::getElapsed(startTime, now);
        log << "EVENT " << worker.getName() << ' ' << name << ' '
            << t << ' '
            << t << '\n';
    }
}

} // namespace Timeplot
