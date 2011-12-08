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
     * {
     * Backing storage for the splat tree. See @ref SplatTree.
     */
    std::vector<size_type> ids;
    std::vector<size_type> start;
    std::vector<size_type> levelStart;
    /**
     * @}
     */
protected:
    virtual size_type *allocateIds(size_type size);
    virtual size_type *allocateStart(size_type size);
    virtual size_type *allocateLevelStart(size_type size);
public:
    SplatTreeHost(const std::vector<Splat> &splats, const Grid &grid);
};

#endif /* !SPLAT_TREE_HOST_H */
