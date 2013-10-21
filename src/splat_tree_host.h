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
 * Declaration of @ref SplatTreeHost.
 */

#ifndef SPLAT_TREE_HOST_H
#define SPLAT_TREE_HOST_H

#if HAVE_CONFIG_H
# include <config.h>
#endif

#include <vector>
#include "splat_tree.h"

/**
 * Concrete implementation of @ref SplatTree that stores data in arrays on the host.
 */
class SplatTreeHost : public SplatTree
{
private:
    /**
     * @name
     * @{
     * Backing storage for the splat tree. @see @ref SplatTree.
     */
    std::vector<command_type> commands;
    std::vector<command_type> start;
    /**
     * @}
     */
protected:
    virtual command_type *allocateCommands(std::size_t size);
    virtual command_type *allocateStart(std::size_t size);

public:
    /**
     * Constructor.
     * @see @ref SplatTree::SplatTree.
     */
    SplatTreeHost(const std::vector<Splat> &splats,
                  const Grid::size_type size[3],
                  const Grid::difference_type offset[3]);

    const std::vector<command_type> &getCommands() { return commands; }
    const std::vector<command_type> &getStart() { return start; }
};

#endif /* !SPLAT_TREE_HOST_H */
