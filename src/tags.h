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
    MLSGPU_TAG_WORK = 3,                ///< Generic tag for transmitting a work item
    MLSGPU_TAG_PROGRESS = 4             ///< A report of progress
};

#endif /* !TAGS_H */
