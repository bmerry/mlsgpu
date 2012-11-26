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
#include <boost/optional.hpp>
#include <string>
#include "timer.h"
#include "statistics.h"

/**
 * Record timing information. The start and end times for various events are
 * recorded to a text file where each line has the format
 * <pre>
 * EVENT <em>worker</em> <em>action</em> <em>start</em> <em>stop</em>
 * </pre>
 * The @a worker and @a action are strings (containing no spaces) provided by
 * the caller. The @a start and @a end times are floating-point numbers in
 * seconds since the program was started.
 *
 * The file format may evolve in future, so tools should ignore lines that do
 * not start with EVENT, and should also ignore extra fields.
 *
 * A @a worker is intended to model a thread: a worker can be working on only
 * one thing at a time. Provided that workers are given unique names (which is
 * not enforced), it is guaranteed that two events for the same worker will not
 * overlap in time. An action is something currently being undertaken by a
 * worker.
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
 * LIFO order i.e. it emulates a call stack. However, only the leaf action is
 * considered active.
 */
class Worker : public boost::noncopyable
{
    friend class Action;
private:
    /// Name of the worker
    const std::string name;

    /// The top of the stack, or @c NULL if there is no current action
    Action *currentAction;

    /**
     * Called by @ref Action to push itself onto the stack. It returns the previous
     * action, which it must save and pass back to @ref stop. That is, the stack is
     * maintained as a linked list of each action pointing to the previous one.
     *
     * @param current       The new current action.
     * @param time          The current time.
     * @return The previous action (possibly @c NULL).
     */
    Action *start(Action *current, Timer::timestamp time);

    /**
     * Called by @ref Action to pop itself from the stack.
     *
     * @param current       The current action.
     * @param prev          The action returned by the corresponding call to @ref start.
     * @param time          The current time.
     *
     * @pre @a current must equal @ref currentAction
     */
    void stop(Action *current, Action *prev, Timer::timestamp time);
public:
    /**
     * Construct the worker.
     *
     * @param name Name for the worker
     */
    explicit Worker(const std::string &name);

    /**
     * Construct the worker, using a number to augment the name.
     *
     * @param name Base name for the worker
     * @param idx  Worker number that is appended to the name
     */
    Worker(const std::string &name, int idx);

    /// Get the name of the worker
    const std::string &getName() const;
};

/**
 * A single action taken by a worker. This is similar to @ref Statistics::Timer,
 * in that it records a start time on creation and a stop time on destruction.
 * For safety it should always be allocated as an automatic variable.
 * It can optionally record the accumulated time to a statistic on destruction.
 *
 * If another action is started on the same worker, this action is paused and
 * does not accumulate time towards the statistic.
 */
class Action : public boost::noncopyable
{
    friend class Worker;
private:
    const std::string name;  ///< Name of the action
    Worker &worker;          ///< Owning worker
    Statistics::Variable *stat; ///< Statistic to update with time (possibly @c NULL).
    Action *oldAction;       ///< Action below this in the stack

    bool running;            ///< Whether time is being attributed to this action
    double elapsed;          ///< Time taken up to the last time it was paused
    Timer::timestamp start;  ///< Time of the last resume, or of construction

    boost::optional<std::size_t> value;  ///< User-supplied value

    /// Second-phase initialization, shared by several constructors
    void init();

    /**
     * Called by @ref Worker to stop the clock. This will cause an @c EVENT
     * record to be emitted.
     *
     * @param time   The current time.
     * @pre The action is not already paused.
     */
    void pause(Timer::timestamp time);

    /**
     * Called by @ref Worker to restart the clock.
     *
     * @param time   The current time.
     * @pre The action is paused.
     */
    void resume(Timer::timestamp time);

public:
    /**
     * Constructor with no statistic to update.
     * @param name   Name for the action.
     * @param worker Owning worker.
     */
    Action(const std::string &name, Worker &worker);

    /**
     * Constructor with statistic to update.
     * @param name   Name for the action.
     * @param worker Owning worker.
     * @param stat   Statistic that will be updated with the total time.
     */
    Action(const std::string &name, Worker &worker, Statistics::Variable &stat);

    /**
     * Constructor with statistic given by name.
     * @param name      Name for the action.
     * @param worker    Owning worker.
     * @param statName  Name of statistic to be found in the default registry.
     */
    Action(const std::string &name, Worker &worker, const std::string &statName);

    /**
     * Set a custom value e.g. the number of bytes consumed on a queue.
     */
    void setValue(std::size_t value);

    /**
     * Destructor. This causes a final @c EVENT record to be emitted.
     * @pre The action is not paused.
     */
    ~Action();
};

} // namespace Timeplot

#endif /* !TIMEPLOT_H */
