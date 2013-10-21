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
 * Statistics collection specific to OpenCL.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif

#ifndef __CL_ENABLE_EXCEPTIONS
# define __CL_ENABLE_EXCEPTIONS
#endif

#include <vector>
#include <queue>
#include <utility>
#include <boost/ref.hpp>
#include <boost/thread/locks.hpp>
#include <CL/cl.hpp>
#include "statistics.h"
#include "statistics_cl.h"
#include "logging.h"

namespace Statistics
{

static bool eventsEnabled = false;
static std::queue<std::pair<std::vector<cl::Event>, boost::reference_wrapper<Variable> > > savedEvents;
static boost::mutex savedEventsMutex;

void enableEventTiming(bool enable)
{
    eventsEnabled = enable;
}

bool isEventTimingEnabled()
{
    return eventsEnabled;
}

static void flushEventTimes(bool finalize)
{
    const cl_profiling_info fields[2] =
    {
        CL_PROFILING_COMMAND_START,
        CL_PROFILING_COMMAND_END
    };

    while (!savedEvents.empty())
    {
        const std::vector<cl::Event> &events = savedEvents.front().first;
        Variable &stat = boost::unwrap_ref(savedEvents.front().second);
        double total = 0.0;
        bool good = true;

        for (std::size_t j = 0; j < events.size() && good; j++)
        {
            const cl::Event &event = events[j];
            if (event.getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>() != CL_COMPLETE)
            {
                if (finalize)
                {
                    Log::log[Log::warn] << "Warning: Event for " << stat.getName() << " did not complete successfully\n";
                    good = false;
                    break;
                }
                else
                {
                    // The front item is not ready to be reaped yet
                    return;
                }
            }

            cl_int status;
            cl_ulong values[2];
            for (unsigned int i = 0; i < 2 && good; i++)
            {
                status = clGetEventProfilingInfo(event(), fields[i], sizeof(values[i]), &values[i], NULL);
                switch (status)
                {
                case CL_PROFILING_INFO_NOT_AVAILABLE:
                    good = false;
                    break;
                case CL_SUCCESS:
                    break;
                default:
                    Log::log[Log::warn] << "Warning: Could not extract profiling information for " << stat.getName() << '\n';
                    good = false;
                    break;
                }
            }

            if (good)
            {
                double duration = 1e-9 * (cl_long(values[1]) - cl_long(values[0]));
                if (duration >= 0.0 && duration < 100.0)
                    total += duration;
                else
                    Log::log[Log::debug] << "Warning: nonsense event times: " << values[1] << " - " << values[0] << " = " << duration << "s\n";
            }
        }

        if (good)
            stat.add(total);
        savedEvents.pop();
        getStatistic<Peak>("events.peak") -= 1;
    }
}

void timeEvents(const std::vector<cl::Event> &events, Variable &stat)
{
    if (eventsEnabled && !events.empty())
    {
        boost::lock_guard<boost::mutex> lock(savedEventsMutex);
        savedEvents.push(std::make_pair(events, boost::ref(stat)));
        getStatistic<Peak>("events.peak") += 1;
        flushEventTimes(false);
    }
}

void timeEvent(const cl::Event &event, Variable &stat)
{
    if (eventsEnabled)
    {
        std::vector<cl::Event> events(1, event);
        timeEvents(events, stat);
    }
}

void CL_CALLBACK timeEventCallback(const cl::Event &event, void *stat)
{
    timeEvent(event, *static_cast<Variable *>(stat));
}

void finalizeEventTimes()
{
    boost::lock_guard<boost::mutex> lock(savedEventsMutex);
    flushEventTimes(true);
}

} // namespace Statistics
