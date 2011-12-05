/**
 * @file
 *
 * Simple timer functions.
 */

#ifndef TIMER_H
#define TIMER_H

#include <string>
#include <time.h>

/**
 * Simple timer that measures elapsed time. Under POSIX, it uses the realtime
 * monotonic timer, and so it may be necessary to pass @c -lrt when linking.
 */
class Timer
{
private:
    struct timespec start;

public:
    /// Constructor. Starts the timer.
    Timer();

    /// Get elapsed time since the timer was constructor
    double getElapsed() const;
};

#endif /* TIMER_H */
