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

    const cl::Context &context;
    const cl::Device &device;
    Buffer ids, start, levelStart;

    virtual size_type *allocateIds(size_type size);
    virtual size_type *allocateStart(size_type size);
    virtual size_type *allocateLevelStart(size_type size);

public:
    SplatTreeCL(const cl::Context &context, const cl::Device &device, const std::vector<Splat> &splats, const Grid &grid);

    const cl::Buffer &getIds() const;
    const cl::Buffer &getStart() const;
    const cl::Buffer &getLevelStart() const;
};

#endif /* !SPLATTREE_CL_H */
