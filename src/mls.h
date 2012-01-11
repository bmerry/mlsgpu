/**
 * @file
 *
 * Moving least squares implementation.
 */

#ifndef MLS_H
#define MLS_H

#if HAVE_CONFIG_H
# include <config.h>
#endif

#include <CL/cl.hpp>
#include <cstddef>
#include "grid.h"
#include "splat_tree_cl.h"

/**
 * Generates the signed distance from an MLS surface for a single slice.
 * It is designed to be usable with @ref Marching::Functor.
 */
class MlsFunctor
{
private:
    cl::Program program;
    mutable cl::Kernel kernel; // has to be mutable to allow arguments to be set

    float zScale, zBias;
    std::size_t dims[2];

    static const std::size_t wgs[2];

public:
    MlsFunctor(const cl::Context &context);

    void set(const Grid &grid, const SplatTreeCL &tree, unsigned int subsamplingShift);

    void operator()(const cl::CommandQueue &queue,
                    const cl::Image2D &slice,
                    cl_uint z,
                    const std::vector<cl::Event> *events,
                    cl::Event *event) const;
};

#endif /* !MLS_H */
