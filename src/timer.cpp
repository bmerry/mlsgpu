/**
 * @file
 *
 * Simple timer functions.
 */

#include <stdexcept>
#include "timer.h"

#if TIMER_TYPE_POSIX

Timer::timestamp Timer::currentTime()
{
    Timer::timestamp now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return now;
}

double Timer::getElapsed(const timestamp &start, const timestamp &end)
{
    double elapsed = end.tv_sec - start.tv_sec + 1e-9 * (end.tv_nsec - start.tv_nsec);
    return elapsed;
}

#endif // TIMER_TYPE_POSIX

#if TIMER_TYPE_WINDOWS

Timer::timestamp Timer::currentTime()
{
    timestamp start;
    BOOL ret = QueryPerformanceCounter(&start);
    if (!ret)
        throw std::runtime_error("QueryPerformanceCounter failed");
    return start;
}

double Timer::getElapsed(const timestamp &start, const timestamp &end)
{
    LARGE_INTEGER freq;
    BOOL ret;
    ret = QueryPerformanceFrequency(&freq);
    if (!ret)
        throw std::runtime_error("QueryPerformanceFrequency failed");
    return (double) (end.QuadPart - start.QuadPart) / freq.QuadPart;
}

#endif // TIMER_TYPE_WINDOWS

Timer::Timer()
{
    start = currentTime();
}

double Timer::getElapsed() const
{
    timestamp end = currentTime();
    return getElapsed(start, end);
}
