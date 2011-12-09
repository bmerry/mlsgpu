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

/**
 * Concrete implementation of @ref SplatTree that stores the data
 * in OpenCL buffers.
 */
class SplatTreeCL : public SplatTree
{
private:
    struct Buffer
    {
        cl::Buffer buffer;
        boost::scoped_ptr<CLH::BufferMapping> mapping;

        size_type *allocate(const cl::Context &context, const cl::Device &device, size_type size);
    };

    /// OpenCL context used to create buffers.
    const cl::Context &context;
    /// OpenCL device used for enqueuing map and unmap operations.
    const cl::Device &device;
    /**
     * @name
     * @{
     * Backing storage for the octree. We don't store a levelStart array, because
     * the kernel computes it internally in shared memory.
     * @see SplatTree.
     */
    Buffer ids, start;
    /** @} */

    /**
     * A lookup table for computing codes.
     * @see @ref processCorners.
     */
    cl::Image2D shuffle;

    virtual size_type *allocateIds(size_type size);
    virtual size_type *allocateStart(size_type size);
    virtual size_type *allocateLevelStart(size_type size);

public:
    /**
     * Constructor. At present the constructor will synchronously create and populate
     * all the OpenCL objects, but in future it may change to be asynchronous and force
     * completion only on the first call to one of the get functions.
     *
     * @param context   OpenCL context used to create buffers, images etc.
     * @param device    OpenCL device used only to enqueue map and unmap operations internally.
     * @param splats    The splats to put in the octree.
     * @param grid      The octree sampling grid.
     */
    SplatTreeCL(const cl::Context &context, const cl::Device &device, const std::vector<Splat> &splats, const Grid &grid);

    /**
     * @name Getters for the buffers and images needed to use the octree.
     * @see @ref processCorners.
     * @{
     */
    const cl::Buffer &getIds() const { return ids.buffer; }
    const cl::Buffer &getStart() const { return start.buffer; }
    const cl::Image2D &getShuffle() const { return shuffle; }
    /**
     * @}
     */
};

#endif /* !SPLATTREE_CL_H */
