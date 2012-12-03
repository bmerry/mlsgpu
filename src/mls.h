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
#include <map>
#include <string>
#include "grid.h"
#include "splat_tree_cl.h"
#include "marching.h"
#include "clh.h"
#include "statistics.h"

class TestMls;

/**
 * Shape to fit through a local set of splats.
 */
enum MlsShape
{
    MLS_SHAPE_SPHERE,
    MLS_SHAPE_PLANE
};

/**
 * Wrapper around @ref MlsShape for use with @ref Choice.
 */
class MlsShapeWrapper
{
public:
    typedef MlsShape type;
    static std::map<std::string, MlsShape> getNameMap();
};

/**
 * Generates the signed distance from an MLS surface for a single slice.
 * It is designed to be usable with @ref Marching.
 *
 * After constructing the object, the user must call @ref set to specify
 * the parameters. The parameters can be changed again later, and doing so
 * is more efficient than creating a new object (since it avoids recompiling
 * the code).
 *
 * This object is @em not thread-safe. Two calls to the () operator cannot be
 * made at the same time, as they will clobber the kernel arguments. However,
 * it is safe for back-to-back calls to the operator() without synchronization
 * i.e. there is no internal device state.
 */
class MlsFunctor : public Marching::Generator
{
private:
    friend class TestMls;

    /**
     * Context used by @ref allocateSlices.
     */
    cl::Context context;

    /**
     * Kernel generated from @ref processCorners.
     */
    cl::Kernel kernel;

    /**
     * Measures device time spent in @ref kernel.
     */
    Statistics::Variable &kernelTime;

    /**
     * Specify the parameters. This is a private variant that
     * does not require the buffers to be stored in a @ref SplatTreeCL, and
     * is used for testing.
     */
    void set(const Grid::difference_type offset[3],
             const cl::Buffer &splats,
             const cl::Buffer &commands,
             const cl::Buffer &start,
             unsigned int subsamplingShift);
public:
    /**
     * Work group size for @ref kernel.
     */
    static const std::size_t wgs[3];

    /**
     * Minimum subsampling for corresponding octree.
     */
    static const int subsamplingMin;

    /**
     * Constructor. It compiles the kernel, so it can throw a compilation error.
     * @param context   The context in which the function operates.
     * @param shape     The shape to fit to the data.
     */
    MlsFunctor(const cl::Context &context, MlsShape shape);

    /**
     * Determines the resource usage of calling @ref allocateSlices, assuming a
     * depth of @ref maxSlices().
     */
    static CLH::ResourceUsage sliceResourceUsage(Grid::size_type width, Grid::size_type height);

    /**
     * Specify the parameters. This must be called before using this object as a functor.
     * The vertices that will be sampled by the functor are from
     * offset (inclusive) to offset + size (exclusive). Note that this means that the
     * number of cells that will be processed by marching cubes will be one @em less
     * than @a size in each dimension.
     *
     * @param offset           Offset between world coordinates and region-relative coordinates.
     * @param tree             Octree containing input splats.
     * @param subsamplingShift Subsampling shift passed when building @a tree.
     *
     * @pre
     * - @a tree was constructed with the same @a offset and @a subsamplingShift.
     */
    void set(const Grid::difference_type offset[3],
             const SplatTreeCL &tree, unsigned int subsamplingShift);

    virtual Grid::size_type maxSlices() const { return wgs[2]; }

    /**
     * @copydoc Marching::Generator::allocateSlices
     *
     * @note This implementation will round the allocation up to multiples of
     * @ref wgs.
     */
    virtual cl::Image2D allocateSlices(
        Grid::size_type width, Grid::size_type height, Grid::size_type depth,
        Grid::size_type &zStride) const;

    /**
     * @pre The tree passed to @ref set was constructed with dimensions at least
     * equal to @a size rounded up to multiples of @ref wgs.
     */
    virtual void enqueue(
        const cl::CommandQueue &queue,
        const cl::Image2D &distance,
        const Grid::size_type size[3],
        Grid::size_type zFirst, Grid::size_type zLast,
        Grid::size_type zStride, Grid::size_type zOffset,
        const std::vector<cl::Event> *events,
        cl::Event *event);

    /**
     * Function object callback for use with @ref Marching.
     *
     * @pre The memory allocated for slice must be padded up to the multiple of
     * the work group size.
     */
    void operator()(const cl::CommandQueue &queue,
                    const cl::Image2D &slice,
                    cl_uint z,
                    const std::vector<cl::Event> *events,
                    cl::Event *event) const;

    /**
     * Sets the tuning factor for boundary clipping.
     * A value of 1 is theoretically "correct" and is the default, but in
     * reality tends to cause holes to open.
     */
    void setBoundaryLimit(float limit);
};

#endif /* !MLS_H */
