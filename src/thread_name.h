/**
 * @file
 *
 * Debug utility to set the name for the current thread.
 */

#ifndef THREAD_NAME_H
#define THREAD_NAME_H

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <string>

/**
 * Sets the thread name to @a name. This may preserve the existing (process)
 * name, so it should not be used more than once on a thread.
 *
 * The effects are platform-dependent and it is not guaranteed that anything
 * will happen at all.
 */
void thread_set_name(const std::string &name);

#endif /* !THREAD_NAME_H */
