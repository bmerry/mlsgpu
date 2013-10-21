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
 * OS-specific utilities to remove a file from the OS cache.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif

#include <string>
#include <stdexcept>
#include <boost/exception/all.hpp>
#include "decache.h"

#if HAVE_POSIX_FADVISE

#ifndef _POSIX_C_SOURCE
# define _POSIX_C_SOURCE 200809L
#endif
#include <fcntl.h>
#include <sys/types.h>
#include <errno.h>

bool decacheSupported()
{
    return true;
}

void decache(const std::string &filename)
{
    int fd = open(filename.c_str(), O_RDONLY);
    if (fd < 0)
    {
        throw boost::enable_error_info(std::runtime_error("could not open file"))
            << boost::errinfo_file_name(filename)
            << boost::errinfo_errno(errno);
    }
    int status = posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    if (status != 0)
    {
        int e = errno;
        close(fd);
        throw boost::enable_error_info(std::runtime_error("posix_fadvise failed"))
            << boost::errinfo_file_name(filename)
            << boost::errinfo_errno(e);
    }

    status = close(fd);
    if (status != 0)
    {
        throw boost::enable_error_info(std::runtime_error("close failed"))
            << boost::errinfo_file_name(filename)
            << boost::errinfo_errno(errno);
    }
}

#else /* !HAVE_POSIX_FADVISE */

#include <fstream>

bool decacheSupported()
{
    return false;
}

void decache(const std::string &filename)
{
    // Try to open the file, just so that semantics are the same
    std::filebuf f;
    if (!f.open(filename.c_str(), std::ios::in))
    {
        throw boost::enable_error_info(std::runtime_error("could not open file"))
            << boost::errinfo_file_name(filename);
    }
}

#endif
