/**
 * OS-specific utilities to remove a file from the OS cache.
 */

#ifndef DECACHE_H
#define DECACHE_H

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <string>

/**
 * Indicates whether this build supports the decaching functionality. If it is
 * not supported, calling @ref decache is still legal but will be a no-op.
 */
bool decacheSupported();

/**
 * Attempt to remove a file from the filesystem cache.
 *
 * @throw std::runtime_error if the file could not be accessed (with @c boost::errinfo_file_name).
 */
void decache(const std::string &filename);

#endif /* !DECACHE_H */
