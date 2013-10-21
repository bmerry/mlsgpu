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
#include "misc.h"

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
    unsigned int t = ticsShown + 1;
    if (t <= totalTics)
        nextTic = mulDiv(total, t, (unsigned int) totalTics);
}

void ProgressDisplay::restart(size_type total)
{
    current = 0;
    ticsShown = 0;
    this->total = total;
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
