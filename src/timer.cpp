#include <stdexcept>
#include "timer.h"

#if TIMER_TYPE_POSIX

Timer::Timer()
{
    clock_gettime(CLOCK_MONOTONIC, &start);
}

double Timer::getElapsed() const
{
    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = end.tv_sec - start.tv_sec + 1e-9 * (end.tv_nsec - start.tv_nsec);
    return elapsed;
}

#endif // TIMER_TYPE_POSIX

#if TIMER_TYPE_WINDOWS

Timer::Timer()
{
    BOOL ret = QueryPerformanceCounter(&start);
    if (!ret)
        throw std::runtime_error("QueryPerformanceCounter failed");
}

double Timer::getElapsed() const
{
    LARGE_INTEGER end;
    LARGE_INTEGER freq;
    BOOL ret;
    ret = QueryPerformanceCounter(&end);
    if (!ret)
        throw std::runtime_error("QueryPerformanceCounter failed");
    ret = QueryPerformanceFrequency(&freq);
    if (!ret)
        throw std::runtime_error("QueryPerformanceFrequency failed");
    return (double) (end.QuadPart - start.QuadPart) / freq.QuadPart;
}

#endif // TIMER_TYPE_WINDOWS
