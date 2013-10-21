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
 * Miscellaneous small functions.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif

#define __STDC_LIMIT_MACROS 1
#include "misc.h"
#include "errors.h"
#include <stdexcept>
#include <string>
#include <errno.h>
#include <climits>
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/exception/all.hpp>

static boost::filesystem::path tmpFileDir;

DownDivider::DownDivider(std::tr1::uint32_t d)
{
    MLSGPU_ASSERT(d > 0, std::invalid_argument);
    shift = 0;
    std::tr1::int64_t k2 = 1; // 2^shift
    negAdd = INT32_MIN; // never matches
    posAdd = INT32_MAX; // never matches
    while (k2 / d <= INT32_MAX / 2)
    {
        shift++;
        k2 <<= 1;
    }
    if (k2 % d == 0)
    {
        // d is a power of 2
        inverse = 1;
        shift = 0;
        while (1U << shift != d)
            shift++;
    }
    else if (k2 % d <= d / 2)
    {
        inverse = k2 / d;
        posAdd = -1;
    }
    else
    {
        inverse = k2 / d + 1;
        negAdd = -1;
    }
}

void createTmpFile(boost::filesystem::path &path, boost::filesystem::ofstream &out)
{
    path = tmpFileDir;
    if (path.empty())
        path = boost::filesystem::temp_directory_path();
    boost::filesystem::path name = boost::filesystem::unique_path("mlsgpu-tmp-%%%%-%%%%-%%%%-%%%%");
    path /= name; // appends
    out.open(path, std::ios::binary);
    if (!out)
    {
        int e = errno;
        throw boost::enable_error_info(std::ios::failure("Could not open temporary file"))
            << boost::errinfo_file_name(path.string())
            << boost::errinfo_errno(e);
    }
}

void setTmpFileDir(const boost::filesystem::path &path)
{
    tmpFileDir = path;
}
