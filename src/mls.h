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
 * It is designed to be usable with @ref Marching::InputFunctor.
 *
 * After constructing the object, the user must call @ref set to specify
 * the parameters. The parameters can be changed again later, and doing so
 * is more efficient than creating a new object (since it avoids recompiling
 * the code).
 *
 * This object is @em not thread-safe. Two calls to the () operator cannot be
 * made at the same time, as they will clobber the kernel arguments.
 *
 * @bug The grid positions use just a scale-and-bias relative to the local origin,
 * and so changing the grid extents while keeping the reference fixed may cause
 * perturbations.
 */
class MlsFunctor
{
private:
    /// Program compiled from @ref mls.cl.
    cl::Program program;
    /**
     * Kernel generated from @ref processCorners.
     * It has to be mutable to allow arguments to be set.
     */
    mutable cl::Kernel kernel;

    /// Horizontal and vertical vertex count of the grid passed to @ref set
    std::size_t dims[2];

public:
    /**
     * Work group size for @ref kernel.
     */
    static const std::size_t wgs[2];

    /**
     * Constructor. It compiles the kernel, so it can throw a compilation error.
     * @param context   The context in which the function operates.
     */
    MlsFunctor(const cl::Context &context);

    /**
     * Specify the parameters. This must be called before using this object as a functor.
     * The vertices that will be sampled by the functor are from
     * offset (inclusive) to offset + size (exclusive). Note that this means that the
     * number of cells that will be processed by marching cubes will be one @em less
     * than @a size in each dimension.
     *
     * @param size, offset     Region of interest used to construct @a tree.
     * @param tree             Octree containing input splats.
     * @param subsamplingShift Subsampling shift passed when building @a tree.
     *
     * @pre
     * - @a tree was constructed with the same @a size, @a offset and @a subsamplingShift.
     */
    void set(const Grid::size_type size[3], const Grid::difference_type offset[3],
             const SplatTreeCL &tree, unsigned int subsamplingShift);

    /**
     * Function object callback for use with @ref Marching.
     *
     * @pre The memory allocated for slice must be padded up to the multiple of
     * the work group size.
     *
     * @bug This isn't currently queryable.
     */
    void operator()(const cl::CommandQueue &queue,
                    const cl::Image2D &slice,
                    cl_uint z,
                    const std::vector<cl::Event> *events,
                    cl::Event *event) const;
};

#endif /* !MLS_H */
