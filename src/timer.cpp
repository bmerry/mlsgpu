#include "timer.h"
#include <string>
#include <cstdio>
#include <time.h>

using namespace std;

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
