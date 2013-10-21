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
 * Simple filter to apply a scale+bias to vertex coordinates in place.
 */

/**
 * Replace each vertex @a v with @a v * @a scaleBias.w + @a scaleBias.xyz.
 * The vertices are tightly packed xyz triplets.
 *
 * There is one workitem per vertex.
 */
__kernel void scaleBiasVertices(
    __global float *vertices,
    float4 scaleBias)
{
    uint gid = get_global_id(0);
    float3 vertex = vload3(gid, vertices);
    vertex = fma(vertex, scaleBias.w, scaleBias.xyz);
    vstore3(vertex, gid, vertices);
}
