/*
 * mlsgpu: surface reconstruction from point clouds
 * Copyright (C) 2013  University of Cape Town
 *
 * This file is part of mlsgpu.
 *
 * mlsgpu is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/**
 * @file
 *
 * Implementation of @ref mesh.h.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif

#ifndef __CL_ENABLE_EXCEPTIONS
# define __CL_ENABLE_EXCEPTIONS
#endif

#include <CL/cl.hpp>
#include <vector>
#include "mesh.h"
#include "clh.h"
#include "errors.h"
#include "tr1_cstdint.h"

DeviceKeyMesh::DeviceKeyMesh(
    const cl::Context &context, cl_mem_flags flags, const MeshSizes &sizes)
    : MeshSizes(sizes),
    vertices(context, flags, sizes.numVertices() ? sizes.numVertices() * (3 * sizeof(cl_float)) : 1),
    triangles(context, flags, sizes.numTriangles() ? sizes.numTriangles() * (3 * sizeof(cl_uint)) : 1),
    vertexKeys(context, flags, sizes.numVertices() ? sizes.numVertices() * sizeof(cl_ulong) : 1)
{
}

HostKeyMesh::HostKeyMesh(void *ptr, const MeshSizes &sizes)
    : MeshSizes(sizes)
{
    std::tr1::uintptr_t ptrInt = reinterpret_cast<std::tr1::uintptr_t>(ptr);
    MLSGPU_ASSERT(ptrInt % sizeof(cl_ulong) == 0, std::invalid_argument);

    vertexKeys = reinterpret_cast<cl_ulong *>(ptr);
    vertices = reinterpret_cast<boost::array<cl_float, 3> *>(vertexKeys + numExternalVertices());
    triangles = reinterpret_cast<boost::array<cl_uint, 3> *>(vertices + numVertices());
}

void enqueueReadMesh(const cl::CommandQueue &queue,
                     const DeviceKeyMesh &dMesh, HostKeyMesh &hMesh,
                     const std::vector<cl::Event> *events,
                     cl::Event *verticesEvent,
                     cl::Event *vertexKeysEvent,
                     cl::Event *trianglesEvent)
{
    MLSGPU_ASSERT(dMesh.numInternalVertices() <= dMesh.numVertices(), std::invalid_argument);
    MLSGPU_ASSERT(static_cast<const MeshSizes &>(dMesh) == hMesh, std::invalid_argument);

    if (trianglesEvent != NULL)
    {
        CLH::enqueueReadBuffer(queue,
                               dMesh.triangles, CL_FALSE,
                               0, dMesh.numTriangles() * (3 * sizeof(cl_uint)),
                               hMesh.triangles,
                               events, trianglesEvent);
        queue.flush();
    }

    if (vertexKeysEvent != NULL)
    {
        CLH::enqueueReadBuffer(queue,
                               dMesh.vertexKeys, CL_FALSE,
                               dMesh.numInternalVertices() * sizeof(cl_ulong),
                               dMesh.numExternalVertices() * sizeof(cl_ulong),
                               hMesh.vertexKeys,
                               events, vertexKeysEvent);
        queue.flush();
    }

    if (verticesEvent != NULL)
    {
        CLH::enqueueReadBuffer(queue,
                               dMesh.vertices, CL_FALSE,
                               0, dMesh.numVertices() * (3 * sizeof(cl_float)),
                               hMesh.vertices,
                               events, verticesEvent);
        queue.flush();
    }
}
