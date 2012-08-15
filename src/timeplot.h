/**
 * @file
 *
 * Record timing information.
 */

#ifndef TIMEPLOT_H
#define TIMEPLOT_H

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <boost/noncopyable.hpp>
#include <string>
#include "timer.h"
#include "statistics.h"

/**
 * Record timing information.
 */
namespace Timeplot
{

/**
 * Initialize the timeplot subsystem. This function is optional; if it is
 * not called, no timeplot data will be written, but statistics will still
 * be updated as normal.
 *
 * @param filename          File to which the data are written.
 * @throw std::ios::failure if the file could not be opened.
 * @pre @ref init has not already been called.
 */
void init(const std::string &filename);

class Action;

/**
 * Encapsulates a worker. Workers perform actions, which must start and stop in
 * LIFO order i.e. it emulates a call stack. However, only the leaf call is shown
 * in the resulting plot.
 */
class Worker : public boost::noncopyable
{
    friend class Action;
private:
    const std::string name;
    Action *currentAction;

    /**
     * Called by @ref Action to push itself onto the stack. It returns the previous
     * action, which it must save and pass back to @ref stop. That is, the stack is
     * maintained as a linked list of each action pointing to the previous one.
     */
    Action *start(Action *current, Timer::timestamp time);

    /**
     * Called by @ref Action to pop itself from the stack. It must pass itself and
     * the previous action in the stack.
     *
     * @pre @a current is the current action.
     */
    void stop(Action *current, Action *prev, Timer::timestamp time);
public:
    /**
     * Construct the worker, in the stopped state.
     *
     * @param name Name for the worker
     */
    explicit Worker(const std::string &name);

    /**
     * Construct the worker, in the stopped state.
     *
     * @param name Base name for the worker
     * @param idx  Worker number that is appended to the name
     */
    Worker(const std::string &name, int idx);

    const std::string &getName() const;
};

/**
 * A single action taken by a worker. This is similar to @ref Statistics::Timer,
 * in that it records a start time on creation and a stop time on destruction.
 * It can optionally record to a statistic in the same way.
 */
class Action : public boost::noncopyable
{
    friend class Worker;
private:
    const std::string name;
    Worker &worker;
    Statistics::Variable *stat;
    Action *oldAction;       ///< Action below this in the stack

    bool running;            ///< Whether time is being attributed to this action
    double elapsed;          ///< Time taken up to the last time it was paused
    Timer::timestamp start;  ///< Time of the last resume, or of construction

    void init();
    void pause(Timer::timestamp time);
    void resume(Timer::timestamp time);

public:
    Action(const std::string &name, Worker &worker);
    Action(const std::string &name, Worker &worker, Statistics::Variable &stat);
    Action(const std::string &name, Worker &worker, const std::string &statName);

    ~Action();
};

} // namespace Timeplot

#endif /* !TIMEPLOT_H */
