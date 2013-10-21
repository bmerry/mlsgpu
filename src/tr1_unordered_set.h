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
 * Wrapper header to include either <tt>&lt;tr1/unordered_set&gt;</tt> or
 * <tt>&lt;unordered_set&gt;</tt> depending on what the compiler provides.
 */

#ifndef TR1_UNORDERED_SET_H
#define TR1_UNORDERED_SET_H

#if HAVE_CONFIG_H
# include <config.h>
#endif

#if HAVE_TR1_UNORDERED_SET
# include <tr1/unordered_set>
#else
# include <unordered_set>
#endif

#endif /* TR1_UNORDERED_SET_H */
