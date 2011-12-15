/**
 * @file
 *
 * Implementation of @ref SplatTree using OpenCL buffers for the backing store.
 */

#ifndef SPLATTREE_CL_H
#define SPLATTREE_CL_H

#if HAVE_CONFIG_H
# include <config.h>
#endif

#include <CL/cl.hpp>
#include <boost/noncopyable.hpp>
#include <boost/smart_ptr/scoped_ptr.hpp>
#include "splat_tree.h"
#include "src/clh.h"
#include "grid.h"

/**
 * Concrete implementation of @ref SplatTree that stores the data
 * in OpenCL buffers.
 *
 * To ease implementation, levels are numbered backwards i.e. level 0 is the
 * largest, finest-grained level, and the last level is 1x1x1.
 */
class SplatTreeCL
{
private:
    Grid grid;

    /// OpenCL context used to create buffers.
    cl::Context context;
    /// OpenCL device used for enqueuing the building operations.
    cl::Device device;

    /**
     * @name
     * @{
     * Backing storage for the octree.
     * @see SplatTree.
     */
    cl::Buffer splats;
    cl::Buffer start;
    cl::Buffer commands;
    /** @} */

    /**
     * @name
     * @{
     * Intermediate data structures used while building the octree.
     *
     * These are never deleted, so that the memory can be recycled each
     * time the octree is regenerated.
     */
    cl::Buffer entryKeys;
    cl::Buffer entryValues;
    /** @} */

    std::size_t maxSplats; ///< Maximum splats for which memory has been allocated

    std::size_t numSplats; ///< Number of splats in the octree
    std::vector<std::size_t> levelOffsets; ///< Start of each level in compacted arrays

    void enqueueUpload(const cl::CommandQueue &queue,
                       const Splat *splats, std::size_t numSplats,
                       const std::vector<cl::Event> *events,
                       cl::Event *event);

    void enqueueWriteEntries(const cl::CommandQueue &queue,
                             const cl::Buffer &keys,
                             const cl::Buffer &values,
                             const cl::Buffer &splats,
                             const Grid &grid);

    void enqueueCountLevel(const cl::CommandQueue &queue,
                           const cl::Buffer &sizes,
                           const cl::Buffer &ranges,
                           const cl::Buffer &keys,
                           std::size_t keysLen,
                           std::size_t keyOffset);

public:
    /**
     * Constructor. This allocates the maximum supported sizes for all the
     * buffers necessary, but does not populate them.
     *
     * @param context   OpenCL context used to create buffers, images etc.
     */
    SplatTreeCL(const cl::Context &context, std::size_t maxLevels, std::size_t maxSplats);

    /**
     * Asynchronously builds the octree, discarding any previous contents.
     *
     * @param queue         The command queue for the building operations.
     * @param splats        The splats to put in the octree.
     * @param numSplats     The size of the @a splats array.
     * @param grid          The octree sampling grid.
     * @param blockingCopy  If true, the @a splats array can be reused on return.
     *                      Otherwise, one must wait for @a uploadEvent.
     * @param events        Events to wait for (or @c NULL).
     * @param[out] uploadEvent   Event that fires when @a splats may be reused (or @c NULL).
     * @param[out] event         Event that fires when the octree is ready to use (or @c NULL).
     *
     * @pre
     * - @a grid has no more than 2^(maxLevels - 1) elements in any direction.
     * - @a numSplats is less than @a maxSplats.
     * - @a splats is not @c NULL.
     */
    void enqueueBuild(const cl::CommandQueue &queue,
                      const Splat *splats, std::size_t numSplats,
                      const Grid &grid, bool blockingCopy,
                      const std::vector<cl::Event> *events,
                      cl::Event *uploadEvent, cl::Event *event);

    /**
     * @name Getters for the buffers and images needed to use the octree.
     * @see @ref processCorners.
     * @{
     */
    const cl::Buffer &getSplats() const { return splats; }
    const cl::Buffer &getCommands() const { return commands; }
    const cl::Buffer &getStart() const { return start; }
    /**
     * @}
     */
};

#endif /* !SPLATTREE_CL_H */
