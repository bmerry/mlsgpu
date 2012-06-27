/**
 * @file
 *
 * Statistics collection specific to OpenCL.
 */

#ifndef MLSGPU_STATISTICS_CL_H
#define MLSGPU_STATISTICS_CL_H

#if HAVE_CONFIG_H
# include <config.h>
#endif

#include <vector>
#include <CL/cl.hpp>
#include "statistics.h"

namespace Statistics
{

/**
 * Requests that the timing statistics for an event be added to a timer statistic.
 * This function operates asynchronously, so the statistic must not be deleted
 * until after @ref finalizeEventTimes has been called. However, the event does
 * not need to be retained by the caller.
 *
 * If profiling was not enabled on the corresponding queue, no statistics are
 * recorded. If some other error occurs, a warning is printed to the log, but
 * no exception is thrown.
 *
 * If the associated command did not complete successfully, a warning is printed
 * to the log and the statistic is not updated.
 *
 * @param event   An enqueued (but not necessarily complete) event
 * @param stat    Statistic to which the time will be added.
 */
void timeEvent(cl::Event event, Variable &stat);

/**
 * Similar to @ref timeEvent, but requests that the combined time from multiple
 * events is added to the statistic as a single data point.
 *
 * If there were problems retrieving the time for any of the events in the list,
 * the entire list will be skipped.
 *
 * @param events  Enqueued (but not necessarily complete) events (can be empty)
 * @param stat    Statistic to which the total time will be added.
 */
void timeEvents(const std::vector<cl::Event> &events, Variable &stat);

/**
 * Ensure that the events registered using @ref timeEvent have had their
 * times extracted and recorded. This must only be called after the events
 * are guaranteed to have completed (e.g. by calling @c clFinish on the
 * corresponding queues), but before the contexts are destroyed.
 */
void finalizeEventTimes();

} // namespace Statistics

#endif /* !MLSGPU_STATISTICS_CL_H */
