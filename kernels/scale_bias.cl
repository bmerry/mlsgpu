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
