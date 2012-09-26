/**
 * @file
 *
 * Enumeration of tags for MPI point-to-point communications.
 */

#ifndef TAGS_H
#define TAGS_H

#if HAVE_CONFIG_H
# include <config.h>
#endif

enum
{
    MLSGPU_TAG_SCATTER_NEED_WORK = 0,   ///< Requester wants work to do
    MLSGPU_TAG_SCATTER_HAS_WORK = 1,    ///< Tells requester to either retrieve work or shut down
    MLSGPU_TAG_GATHER_HAS_WORK = 2,     ///< Tells the receiver to either receive work or decrement refcount
    MLSGPU_TAG_WORK = 3                 ///< Generic tag for transmitting a work item
};

#endif /* !TAGS_H */
