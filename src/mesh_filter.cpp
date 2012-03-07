/**
 * @file
 *
 * Filter chains for post-processing on-device meshes.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif

#ifndef __CL_ENABLE_EXCEPTIONS
# define __CL_ENABLE_EXCEPTIONS
#endif

#include <CL/cl.hpp>
#include <vector>
#include <algorithm>
#include <boost/function.hpp>
#include <boost/foreach.hpp>
#include "mesh_filter.h"
#include "errors.h"

void MeshFilterChain::operator()(
    const cl::CommandQueue &queue,
    const DeviceKeyMesh &mesh,
    const std::vector<cl::Event> *events,
    cl::Event *event) const
{
    MLSGPU_ASSERT(mesh.numTriangles > 0 && mesh.numVertices > 0, std::invalid_argument);

    DeviceKeyMesh meshes[2]; // for ping-pong
    DeviceKeyMesh *inMesh = &meshes[0];
    DeviceKeyMesh *outMesh = &meshes[1];
    *inMesh = mesh;

    std::vector<cl::Event> wait(1);
    cl::Event last;
    BOOST_FOREACH(const MeshFilter &filter, filters)
    {
        filter(queue, *inMesh, events, &last, *outMesh);
        wait[0] = last;
        events = &wait;
        std::swap(inMesh, outMesh);
        if (inMesh->numTriangles == 0)
        {
            // Filter completed eliminated the mesh.
            if (event != NULL)
                *event = last;
            return;
        }
    }
    output(queue, *inMesh, events, event);
}
