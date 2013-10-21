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
 * Debug utility to set the name for the current thread.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif

#include <string>
#include "thread_name.h"

#if HAVE_PTHREAD_SETNAME_NP
# ifndef _GNU_SOURCE
#  define _GNU_SOURCE 1
# endif
# include <pthread.h>
# include <errno.h>

void thread_set_name(const std::string &name)
{
    char oldName[1024];
    if (pthread_getname_np(pthread_self(), oldName, sizeof(oldName)) == 0)
    {
        std::string newName = name + " [" + oldName + "]";
        int status = pthread_setname_np(pthread_self(), newName.c_str());
        if (status == ERANGE)
        {
            // Under Linux with glibc there is a limit of 15 characters, which
            // we might have overflowed.
            pthread_setname_np(pthread_self(), name.c_str());
        }
    }
}

#else // HAVE_PTHREAD_SETNAME_NP

void thread_set_name(const std::string &name)
{
    (void) name;
}

#endif
