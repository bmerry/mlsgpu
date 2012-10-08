/**
 * @file
 *
 * A thread-safe progress meter, modelled on boost::progress_display.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif

#include <ostream>
#include <string>
#include <boost/thread/mutex.hpp>
#include <boost/thread/locks.hpp>
#include "progress.h"

void ProgressMeter::operator++()
{
    *this += 1;
}

ProgressDisplay::ProgressDisplay(size_type total,
                                 std::ostream &os,
                                 const std::string &s1,
                                 const std::string &s2,
                                 const std::string &s3)
: os(os), s1(s1), s2(s2), s3(s3)
{
    restart(total);
}

void ProgressDisplay::updateNextTic()
{
    size_type t = ticsShown + 1;
    /* We want to compute total * t / totalTics (rounded down), but the
     * intermediate result may overflow. So instead, we represent total
     * as totalQ * totalTics + totalR. Then the value we want is
     * totalQ * t + (totalR * t / totalTics), and this will not lead to
     * overflow provided that totalTics^2 is small enough.
     */
    nextTic = totalQ * t + (totalR * t / totalTics);
}

void ProgressDisplay::restart(size_type total)
{
    current = 0;
    ticsShown = 0;
    this->total = total;
    totalQ = total / totalTics;
    totalR = total % totalTics;
    os  << s1 << "0%   10   20   30   40   50   60   70   80   90   100%\n"
        << s2 << "|----|----|----|----|----|----|----|----|----|----|\n"
        << s3;
    os.flush();

    updateNextTic();
}

void ProgressDisplay::operator+=(size_type increment)
{
    boost::lock_guard<boost::mutex> lock(mutex);
    this->current += increment;

    while (ticsShown < totalTics && this->current >= nextTic)
    {
        os << '*'; os.flush();
        ticsShown++;
        updateNextTic();
        if (ticsShown == totalTics)
            os << std::endl;
    }
}

ProgressDisplay::size_type ProgressDisplay::count() const
{
    boost::lock_guard<boost::mutex> lock(mutex);
    return current;
}

ProgressDisplay::size_type ProgressDisplay::expected_count() const
{
    return total;
}
