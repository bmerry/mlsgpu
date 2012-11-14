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
