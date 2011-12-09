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

SplatTree::command_type *SplatTreeHost::allocateStart(
    std::size_t width, std::size_t height, std::size_t depth,
    std::size_t &rowPitch, std::size_t &slicePitch)
{
    start.resize(width * height * depth);
    rowPitch = width;
    slicePitch = width * height;
    return &start[0];
}

SplatTreeHost::SplatTreeHost(const std::vector<Splat> &splats, const Grid &grid)
    : SplatTree(splats, grid)
{
    initialize();
}
