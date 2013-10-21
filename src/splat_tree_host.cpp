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
 * Implementation of @ref SplatTreeHost.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <vector>
#include "grid.h"
#include "splat.h"
#include "splat_tree_host.h"

SplatTree::command_type *SplatTreeHost::allocateCommands(std::size_t size)
{
    commands.resize(size);
    return &commands[0];
}

SplatTree::command_type *SplatTreeHost::allocateStart(std::size_t size)
{
    start.resize(size);
    return &start[0];
}

SplatTreeHost::SplatTreeHost(const std::vector<Splat> &splats,
                             const Grid::size_type size[3],
                             const Grid::difference_type offset[3])
: SplatTree(splats, size, offset)
{
    initialize();
}
