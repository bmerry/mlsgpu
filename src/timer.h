/**
 * @file
 *
 * Simple timer functions.
 */

#ifndef TIMER_H
#define TIMER_H

#if HAVE_CONFIG_H
# include <config.h>
#endif

#if HAVE_CLOCK_GETTIME
# ifndef _POSIX_C_SOURCE
#  define _POSIX_C_SOURCE 200112L
# endif
# include <time.h>
# define TIMER_TYPE_POSIX 1
#elif HAVE_QUERY_PERFORMANCE_COUNTER
# ifndef WIN32_LEAN_AND_MEAN
#  define WIN32_LEAN_AND_MEAN
# endif
# include <windows.h>
# define TIMER_TYPE_WINDOWS 1
#else
# error "No timer implementation found"
#endif


#include <time.h>

/**
 * Simple timer that measures elapsed time. Under POSIX, it uses the realtime
 * monotonic timer, and so it may be necessary to pass @c -lrt when linking.
 * Under Windows it uses QueryPerformanceCounter.
 */
class Timer
{
private:
#if TIMER_TYPE_POSIX
    struct timespec start;
#endif
#if TIMER_TYPE_WINDOWS
    LARGE_INTEGER start;
#endif

public:
    /**
     * Constructor. Starts the timer.
     */
    Timer();

    /// Get elapsed time since the timer was constructed
    double getElapsed() const;
};

#endif /* TIMER_H */
