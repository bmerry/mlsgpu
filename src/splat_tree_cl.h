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

        command_type *allocate(const cl::Context &context, const cl::Device &device, std::size_t size);
    };

    struct Image3D
    {
        cl::Image3D image;
        boost::scoped_ptr<CLH::ImageMapping> mapping;

        command_type *allocate(const cl::Context &context, const cl::Device &device,
                               std::size_t width, std::size_t height, std::size_t depth,
                               std::size_t &rowPitch, std::size_t &slicePitch);
    };

    /// OpenCL context used to create buffers.
    const cl::Context &context;
    /// OpenCL device used for enqueuing map and unmap operations.
    const cl::Device &device;
    /**
     * @name
     * @{
     * Backing storage for the octree.
     * @see SplatTree.
     */
    Buffer commands;
    Image3D start;
    /** @} */

    virtual command_type *allocateCommands(std::size_t size);
    virtual command_type *allocateStart(std::size_t width, std::size_t height, std::size_t depth,
                                        std::size_t &rowPitch, std::size_t &slicePitch);

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
    const cl::Buffer &getCommands() const { return commands.buffer; }
    const cl::Image3D &getStart() const { return start.image; }
    /**
     * @}
     */
};

#endif /* !SPLATTREE_CL_H */
