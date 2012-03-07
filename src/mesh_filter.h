/**
 * @file
 *
 * Filter chains for post-processing on-device meshes.
 */

#ifndef MESH_FILTER_H
#define MESH_FILTER_H

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <CL/cl.hpp>
#include <boost/function.hpp>
#include <boost/bind.hpp>
#include <boost/ref.hpp>
#include <vector>
#include "mesh.h"
#include "marching.h"

/**
 * Function for accepting a mesh and transforming it in some way (on the
 * device). The output mesh will usually reference internal state of the
 * function object rather than allocating storage on the fly, so these
 * functions are not expected to be reentrant.
 *
 * The parameters are
 *  -# A command queue to use for enqueuing work
 *  -# The input mesh (not expected to be modified).
 *  -# Events to wait for before reading from the input mesh (may be @c NULL).
 *  -# An event the caller must wait for before reading the output mesh (may be @c NULL).
 *  -# The output mesh.
 *
 * A filter may assume that the input is non-empty, and may output an empty mesh.
 */
typedef boost::function<
    void(const cl::CommandQueue &queue,
         const DeviceKeyMesh &inMesh,
         const std::vector<cl::Event> *events,
         cl::Event *event,
         DeviceKeyMesh &outMesh)> MeshFilter;


/**
 * Combine several filters with an output functor to create a composite output
 * functor that can be used with @ref Marching.
 */
class MeshFilterChain
{
private:
    std::vector<MeshFilter> filters;
    Marching::OutputFunctor output;

public:
    typedef void result_type;

    /**
     * Default constructor. The object cannot be used in this form, and this constructor
     * is provided only to make the object default-constructible. @ref setOutput must be
     * called before the object is usable.
     */
    MeshFilterChain() {}

    /**
     * Constructor. The provided arguments are copied.
     */
    MeshFilterChain(const std::vector<MeshFilter> &filters, const Marching::OutputFunctor &output)
        : filters(filters), output(output) {}

    /**
     * Update the output functor dynamically. The output functor is copied; use
     * @c boost::ref if it should be held by reference.
     */
    void setOutput(const Marching::OutputFunctor &output)
    {
        this->output = output;
    }

    /**
     * Append a filter to the end. The filter is copied; use @c boost::ref if it should be
     * held by reference.
     */
    void addFilter(const MeshFilter &filter)
    {
        filters.push_back(filter);
    }

    /**
     * Output functor suitable for use as @ref Marching::OutputFunctor. It applies each of
     * the filters in turn, passing the last output to the output functor provided to the
     * constructor.
     *
     * If any of the filters outputs an empty mesh (zero triangles), processing is terminated
     * and the event returned is for completion of the filters that have already executed. The
     * output functor will not be called in this case.
     *
     * @pre This object was not default constructed.
     */
    void operator()(
        const cl::CommandQueue &queue,
        const DeviceKeyMesh &mesh,
        const std::vector<cl::Event> *events,
        cl::Event *event) const;
};

#endif /* !MESH_FILTER_H */
