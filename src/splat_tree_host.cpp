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

SplatTree::size_type *SplatTreeHost::allocateIds(size_type size)
{
    ids.resize(size);
    return &ids[0];
}

SplatTree::size_type *SplatTreeHost::allocateStart(size_type size)
{
    start.resize(size);
    return &start[0];
}

SplatTree::size_type *SplatTreeHost::allocateLevelStart(size_type size)
{
    levelStart.resize(size);
    return &levelStart[0];
}

SplatTreeHost::SplatTreeHost(const std::vector<Splat> &splats, const Grid &grid)
    : SplatTree(splats, grid)
{
    initialize();
}
