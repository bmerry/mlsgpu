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

#ifndef PROGRESS_H
#define PROGRESS_H

#if HAVE_CONFIG_H
# include <config.h>
#endif

#include <ostream>
#include <iostream>
#include <string>
#include <boost/thread/mutex.hpp>
#include "tr1_cstdint.h"
#include <boost/noncopyable.hpp>

/**
 * An abstraction of a progress meter. It supports large integral progress values.
 * The display of the progress is left to concrete subclasses.
 */
class ProgressMeter
{
public:
    /// Type to store progress amounts
    typedef std::tr1::uintmax_t size_type;

    virtual ~ProgressMeter() {}

    /// Add 1 to the progress
    virtual void operator++();

    /// Add a given amount to the progress
    virtual void operator+=(size_type increment) = 0;
};

/**
 * A thread-safe progress meter which displays ASCII-art progress.
 */
class ProgressDisplay : public ProgressMeter, public boost::noncopyable
{
public:
    /**
     * Constructor.
     *
     * @param total     Amount of progress on completion
     * @param os        Output stream to show the progress bar
     * @param s1,s2,s3  Prefix to apply to each line of the progress bar
     */
    explicit ProgressDisplay(size_type total,
                             std::ostream &os = std::cout,
                             const std::string &s1 = "\n",
                             const std::string &s2 = "",
                             const std::string &s3 = "");

    virtual void operator+=(std::tr1::uintmax_t increment);

    size_type count() const;     ///< Current value
    size_type expected_count() const;  ///< Value at completion

private:
    size_type current;
    unsigned int ticsShown;      ///< Number of tick marks already displayed
    size_type nextTic;           ///< Progress amount at which the next tick will be shown

    size_type total;             ///< Total amount of progress expected

    mutable boost::mutex mutex;  ///< Lock protecting the count and stream
    std::ostream &os;            ///< Output stream
    const std::string s1, s2, s3;

    enum
    {
        totalTics = 51           ///< Width of the ASCII art
    };

    /// Print the header and initialize state
    void restart(size_type total);

    /**
     * Recompute @ref nextTic. Call this at the start and after drawing a tic.
     */
    void updateNextTic();
};

#endif /* !PROGRESS_H */
