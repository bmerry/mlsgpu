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

SplatTreeHost::SplatTreeHost(const std::vector<Splat> &splats, const Grid &grid)
    : SplatTree(splats, grid)
{
    initialize();
}
