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
#include "mesh.h"
#include "clh.h"

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
     * -# Input buffer of vertices, of tightly-packed triplets of @c cl_float.
     * -# The number of vertices.
     * -# Events to wait for. If this is non-NULL, the named events must be
     *    waited for before reading the input vertices.
     * -# Completion event. If this is non-NULL, it must be written with an
     *    event that will be signaled once the output buffer is ready.
     */
    typedef boost::function<
        void(const cl::CommandQueue &,
             const cl::Buffer &,
             const cl::Buffer &,
             std::size_t,
             const std::vector<cl::Event> *,
             cl::Event *)> DistanceFunctor;

    static CLH::ResourceUsage resourceUsage(
        const cl::Device &device,
        std::size_t maxVertices, std::size_t maxTriangles);

    Clip(const cl::Context &context, const cl::Device &device,
         std::size_t maxVertices, std::size_t maxTriangles);

    void setDistanceFunctor(const DistanceFunctor &distanceFunctor);
    void setOutput(const Marching::OutputFunctor &output);

    void operator()(
        const cl::CommandQueue &queue,
        const DeviceKeyMesh &mesh,
        const std::vector<cl::Event> *events,
        cl::Event *event);

private:
    std::size_t maxVertices, maxTriangles;
    DistanceFunctor distanceFunctor;
    Marching::OutputFunctor output;

    cl::Buffer distances;       ///< Signed distances from the clip boundary
    /**
     * Booleans indicating whether vertices should be kept; later scanned.
     * The elements are of type @c cl_uint.
     */
    cl::Buffer vertexCompact;
    /**
     * Booleans indicating whether triangles should be kept; later scanned.
     * The elements are of type @c cl_uint.
     */
    cl::Buffer triangleCompact;
    clogs::Scan compactScan;    ///< Scanner object for scanning the compaction arrays
    DeviceKeyMesh outMesh;      ///< Compacted clipped mesh.

    cl::Program program;
    cl::Kernel vertexInitKernel;
    cl::Kernel classifyKernel;
    cl::Kernel triangleCompactKernel;
    cl::Kernel vertexCompactKernel;
};

#endif /* !CLIP_H */
