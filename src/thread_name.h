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
