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

class TestSplatTree;

/**
 * Concrete implementation of @ref SplatTree that stores data in arrays on the host.
 */
class SplatTreeHost : public SplatTree
{
    friend class TestSplatTree;
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
    virtual command_type *allocateStart(std::size_t width, std::size_t height, std::size_t depth,
                                        std::size_t &rowPitch, std::size_t &slicePitch);
public:
    /**
     * Constructor.
     * @see @ref SplatTree::SplatTree.
     */
    SplatTreeHost(const std::vector<Splat> &splats, const Grid &grid);
};

#endif /* !SPLAT_TREE_HOST_H */
