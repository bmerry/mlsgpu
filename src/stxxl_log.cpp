/**
 * @file
 *
 * Replacement for STXXL @c print_msg internal function.
 */

#include <stxxl.h>
#include "logging.h"

namespace stxxl
{

/**
 * Replacement for STXXL @c print_msg internal function. This is only intended
 * to be used by @ref stxxl_log.h. The message does not need to have a terminating
 * newline: it will be added by this function.
 *
 * @param label Subsystem generating the message (may be @c NULL)
 * @param msg   The message to display
 * @param level Log level
 *
 * @warning This is a nasty hack. It could be broken at any time by STXXL changes,
 * and probably will not work on all operating systems.
 */
void print_msg(const char *label, const std::string &msg, unsigned int flags)
{
    Log::Level level;
    if (flags & _STXXL_PRNT_CERR)
        level = Log::warn;
    else if (flags & _STXXL_PRNT_COUT)
        level = Log::info;
    else
        level = Log::debug;

    std::ostream &o = Log::log[level];
    if (label != NULL)
        o << '[' << label << "] ";
    o << msg;
    if (flags & _STXXL_PRNT_ADDNEWLINE)
        o << '\n';
    o << std::flush;
}

}
