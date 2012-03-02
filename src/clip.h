/**
 * @file
 *
 * Marching output functor for boundary clipping.
 */

#ifndef CLIP_H
#define CLIP_H

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <CL/cl.hpp>
#include <cstddef>
#include <vector>
#include <clogs/scan.h>
#include <boost/function.hpp>
#include <boost/noncopyable.hpp>
#include "marching.h"

/**
 * Marching output functor for boundary clipping.
 *
 * This is a proxy functor that satisfies the requirements for @ref
 * Marching::OutputFunctor. It
 * -# Receives a collection of vertices and triangles.
 * -# Evaluates a user-provided function at the vertices.
 * -# Clips the geometry to the portion when the user-provided function is
 *   negative.
 * -# Compacts the resulting geometry.
 * -# Forwards the results to another output functor.
 *
 * Newly introduced vertices will created along existing edges. They will be
 * classified as external if both vertices on the original edge were external,
 * and the vertex key will be set to the average of the original keys.
 *
 * @note This obtain maintains state, so it cannot be used directly as a
 * functor. It must be wrapped in @c boost::ref.
 */
class Clip : public boost::noncopyable
{
public:
    /**
     * Type for a functor used to determine whether each vertex is inside
     * or outside the region to keep (and how far).
     *
     * -# A command queue to use.
     * -# Output buffer of @c cl_float, containing signed distances (negative
     *   means inside/keep, positive means outside/clip).
     * -# Input buffer of vertices, of type @c cl_float3.
     * -# Events to wait for. If this is non-NULL, the named events must be
     *    waited for before reading the input vertices.
     * -# Completion event. If this is non-NULL, it must be written with an
     *    event that will be signaled once the output buffer is ready.
     */
    typedef boost::function<
        void(const cl::CommandQueue &,
             const cl::Buffer &,
             const cl::Buffer &,
             const std::vector<cl::Event> *,
             const cl::Event *)> DistanceFunctor;

    Clip(const cl::Context &context, const cl::Device &device,
         std::size_t maxVertices, std::size_t maxTriangles);

    void setDistance(const DistanceFunctor &distance);
    void setOutput(const Marching::OutputFunctor &output);

    void operator()(
        const cl::CommandQueue &queue,
        const cl::Buffer &vertices,
        const cl::Buffer &vertexKeys,
        const cl::Buffer &indices,
        std::size_t numVertices,
        std::size_t numInternalVertices,
        std::size_t numIndices,
        cl::Event *event);

private:
    std::size_t maxVertices, maxTriangles;
    DistanceFunctor distance;
    Marching::OutputFunctor output;

    cl::Buffer discriminant;
    cl::Buffer compact;
    clogs::Scan compactScan;
    cl::Buffer outVertices;
    cl::Buffer outVertexKeys;
    cl::Buffer outIndices;

    cl::Program program;
    cl::Kernel vertexClassifyKernel;
    cl::Kernel vertexCompactKernel;
    cl::Kernel triangleClassifyKernel;
    cl::Kernel triangleCompactKernel;
};

#endif /* !CLIP_H */
