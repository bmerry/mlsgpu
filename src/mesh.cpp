/**
 * @file
 *
 * Implementation of @ref MeshBase subclasses.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif

#ifndef __CL_ENABLE_EXCEPTIONS
# define __CL_ENABLE_EXCEPTIONS
#endif

#include <CL/cl.h>
#include <vector>
#include <boost/array.hpp>
#include <boost/function.hpp>
#include <boost/bind/bind.hpp>
#include <tr1/unordered_map>
#include <cassert>
#include "mesh.h"
#include "fast_ply.h"

std::size_t SimpleMesh::numVertices() const
{
    return vertices.size();
}

std::size_t SimpleMesh::numTriangles() const
{
    return triangles.size();
}

void SimpleMesh::add(const cl::CommandQueue &queue,
                     const cl::Buffer &vertices,
                     const cl::Buffer &vertexKeys,
                     const cl::Buffer &indices,
                     std::size_t numVertices,
                     std::size_t numInternalVertices,
                     std::size_t numIndices,
                     cl::Event *event)
{
    /* Unused parameters */
    (void) numInternalVertices;
    (void) vertexKeys;

    std::size_t oldVertices = this->vertices.size();
    std::size_t oldTriangles = this->triangles.size();
    std::size_t numTriangles = numIndices / 3;
    this->vertices.resize(oldVertices + numVertices);
    this->triangles.resize(oldTriangles + numTriangles);

    std::vector<cl::Event> wait(1);
    cl::Event last;
    queue.enqueueReadBuffer(vertices, CL_FALSE,
                            0, numVertices * (3 * sizeof(cl_float)),
                            &this->vertices[oldVertices][0],
                            NULL, &last);
    wait[0] = last;
    queue.enqueueReadBuffer(indices, CL_FALSE,
                            0, numTriangles * (3 * sizeof(cl_uint)),
                            &this->triangles[oldTriangles][0],
                            &wait, &last);
    if (event != NULL)
        *event = last;
}

void SimpleMesh::write(const std::string &filename) const
{
    FastPly::StreamWriter writer;
    writer.setNumVertices(vertices.size());
    writer.setNumTriangles(triangles.size());
    writer.open(filename);
    writer.writeVertices(0, vertices.size(), &vertices[0][0]);
    writer.writeTriangles(0, triangles.size(), &triangles[0][0]);
}

Marching::OutputFunctor SimpleMesh::outputFunctor(unsigned int pass)
{
    /* only one pass, so ignore it */
    (void) pass;
    assert(pass == 0);

    return Marching::OutputFunctor(boost::bind(&SimpleMesh::add, this, _1, _2, _3, _4, _5, _6, _7, _8));
}

std::size_t WeldMesh::numVertices() const
{
    return internalVertices.size() + externalVertices.size();
}

std::size_t WeldMesh::numTriangles() const
{
    return triangles.size();
}

void WeldMesh::add(const cl::CommandQueue &queue,
                   const cl::Buffer &vertices,
                   const cl::Buffer &vertexKeys,
                   const cl::Buffer &indices,
                   std::size_t numVertices,
                   std::size_t numInternal,
                   std::size_t numIndices,
                   cl::Event *event)
{
    std::size_t oldInternal = internalVertices.size();
    std::size_t oldExternal = externalVertices.size();
    std::size_t oldVertices = oldInternal + oldExternal;
    std::size_t oldTriangles = triangles.size();
    std::size_t numExternal = numVertices - numInternal;
    std::size_t numTriangles = numIndices / 3;

    internalVertices.resize(oldInternal + numInternal);
    externalVertices.resize(oldExternal + numExternal);
    externalKeys.resize(externalVertices.size());
    triangles.resize(oldTriangles + numTriangles);

    cl::Event indicesEvent, last;
    std::vector<cl::Event> wait(1);

    queue.enqueueReadBuffer(indices, CL_FALSE, 0, numTriangles * (3 * sizeof(cl_uint)),
                            &triangles[oldTriangles][0], NULL, &indicesEvent);
    queue.flush(); // Kick off this read-back in the background while we queue more.

    /* Read back the vertex and key data. We don't need it now, so we just return
     * an event for it.
     * TODO: allow them to proceed in parallel.
     */
    if (numInternal > 0)
    {
        queue.enqueueReadBuffer(vertices, CL_FALSE,
                                0,
                                numInternal * (3 * sizeof(cl_float)),
                                &internalVertices[oldInternal][0],
                                NULL, &last);
        wait[0] = last;
    }
    if (numExternal > 0)
    {
        queue.enqueueReadBuffer(vertices, CL_FALSE,
                                numInternal * (3 * sizeof(cl_float)),
                                numExternal * (3 * sizeof(cl_float)),
                                &externalVertices[oldExternal][0],
                                &wait, &last);
        wait[0] = last;
        queue.enqueueReadBuffer(vertexKeys, CL_FALSE,
                                numInternal * sizeof(cl_ulong),
                                numExternal * sizeof(cl_ulong),
                                &externalKeys[oldExternal],
                                &wait, &last);
        wait[0] = last;
    }

    /* Rewrite indices to refer to the two separate arrays, at the same time
     * applying ~ to the external indices to disambiguate them.  Note that
     * these offsets will wrap around, but that is well-defined for unsigned
     * values.
     *
     * Internal vertices are currently indexed starting at oldVertices,
     * and external vertices are indexed starting at oldVertices + numInternal.
     */
    indicesEvent.wait();
    cl_uint offsetInternal = oldInternal - oldVertices;
    cl_uint offsetExternal = oldExternal - (oldVertices + numInternal);
    for (std::size_t i = oldTriangles; i < oldTriangles + numTriangles; i++)
        for (unsigned int j = 0; j < 3; j++)
        {
            cl_uint &index = triangles[i][j];
            if (index < oldVertices + numInternal)
                index = (index + offsetInternal);
            else
                index = ~(index + offsetExternal);
        }
    if (event != NULL)
        *event = last; /* Waits for vertices to be transferred */
}

void WeldMesh::finalize()
{
    std::size_t welded = 0;
    std::tr1::unordered_map<cl_ulong, cl_uint> place; // maps keys to new positions

    /* Maps original external indices to new ones. It includes a bias of
     * |internalVertices| so that we can index the concatenation of
     * internal and external vertices.
     */
    std::vector<cl_uint> remap(externalVertices.size());

    /* Weld the external vertices in place */
    for (size_t i = 0; i < externalVertices.size(); i++)
    {
        cl_ulong key = externalKeys[i];
        std::tr1::unordered_map<cl_ulong, cl_uint>::const_iterator pos = place.find(key);
        if (pos == place.end())
        {
            // New key, not seen before
            place[key] = welded;
            remap[i] = welded + internalVertices.size();
            // Shuffle down the vertex data in-place
            externalVertices[welded] = externalVertices[i];
            welded++;
        }
        else
        {
            remap[i] = pos->second + internalVertices.size();
        }
    }

    /* Rewrite the indices that refer to external vertices
     * (TODO: is it possible to partition these as well, to
     * reduce the work?)
     */
    for (std::size_t i = 0; i < triangles.size(); i++)
        for (unsigned int j = 0; j < 3; j++)
        {
            cl_uint &index = triangles[i][j];
            if (index >= internalVertices.size())
            {
                assert(~index < externalVertices.size());
                index = remap[~index];
            }
        }

    /* Throw away unneeded data. */
    std::vector<cl_ulong>().swap(externalKeys);
    externalVertices.resize(welded);
}

void WeldMesh::write(const std::string &filename) const
{
    FastPly::StreamWriter writer;
    writer.setNumVertices(internalVertices.size() + externalVertices.size());
    writer.setNumTriangles(triangles.size());
    writer.open(filename);
    writer.writeVertices(0, internalVertices.size(), &internalVertices[0][0]);
    writer.writeVertices(internalVertices.size(), externalVertices.size(), &externalVertices[0][0]);
    writer.writeTriangles(0, triangles.size(), &triangles[0][0]);
}

Marching::OutputFunctor WeldMesh::outputFunctor(unsigned int pass)
{
    /* only one pass, so ignore it */
    (void) pass;
    assert(pass == 0);

    return Marching::OutputFunctor(boost::bind(&WeldMesh::add, this, _1, _2, _3, _4, _5, _6, _7, _8));
}
