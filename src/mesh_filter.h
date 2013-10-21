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
#include "statistics.h"

class Grid;

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
     * @pre
     * - The output functor has been set.
     */
    void operator()(
        const cl::CommandQueue &queue,
        const DeviceKeyMesh &mesh,
        const std::vector<cl::Event> *events,
        cl::Event *event) const;
};

/**
 * Mesh filter that applies a scale-and-bias. This class is not reentrant i.e.
 * the @c operator() must not be called from multiple threads simultaneously.
 * However, the queued work can proceed in parallel as there is no internal
 * device state.
 */
class ScaleBiasFilter
{
private:
    /**
     * Kernel generated from @ref scaleBiasVertices. It is mutable so that
     * the arguments can be set.
     */
    mutable cl::Kernel kernel;

    /**
     * Statistic for time spent in the kernel
     */
    Statistics::Variable &kernelTime;

    cl_float4 scaleBias;          ///< Scale in w, bias in xyz

public:
    /**
     * Default constructor. Sets a scale of 1 and a bias of 0.
     */
    ScaleBiasFilter(const cl::Context &context);

    /// Set the scale and bias.
    void setScaleBias(float scale, float x, float y, float z);

    /**
     * Set the scale and bias from a grid. The scale and bias are set such that
     * grid coordinates are transformed to world coordinates.
     */
    void setScaleBias(const Grid &grid);

    /// Filter operation (see @ref MeshFilter).
    void operator()(
        const cl::CommandQueue &queue,
        const DeviceKeyMesh &inMesh,
        const std::vector<cl::Event> *events,
        cl::Event *event,
        DeviceKeyMesh &outMesh) const;
};

#endif /* !MESH_FILTER_H */
