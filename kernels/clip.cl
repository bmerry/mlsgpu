/**
 * @file
 *
 * Kernels for clipping geometry against an implicit boundary. At present there
 * is no actual clipping; triangles that hit the boundary are discarded.
 */

/**
 * Marks all vertices as rejected (@ref classify will mark the relevant ones as accepted).
 */
__kernel void vertexInit(
    __global uint * restrict vertexKeep)
{
    vertexKeep[get_global_id(0)] = 0;
}

/**
 * Mark triangles and vertices that are inside the boundary as accepted.
 * Triangles are retained if all three vertices are inside the boundary.
 * Vertices are retained if incident on at least one retained triangle.
 * Note that this means that vertices inside the boundary can nevertheless
 * be rejected.
 *
 * @param[out] triangleKeep      Set to 1 for retained triangles, 0 for rejected ones.
 * @param[out] vertexKeep        Set to 1 for retained vertices.
 * @param      indices           Triangle indices.
 * @param      vertexDist        Signed distances of vertices from the boundary (negative for inside).
 *
 * @pre @a vertexKeep has been set to all zeros.
 */
__kernel void classify(
    __global uint * restrict triangleKeep,
    __global uint * restrict vertexKeep,
    __global const uint * restrict indices,
    __global const float * restrict vertexDist)
{
    uint gid = get_global_id(0);
    uint3 idx = vload3(gid, indices);
    float3 dist;
    dist.s0 = vertexDist[idx.s0];
    dist.s1 = vertexDist[idx.s1];
    dist.s2 = vertexDist[idx.s2];
    bool keep = all(dist <= 0.0f);
    triangleKeep[gid] = keep;
    if (keep)
    {
        vertexKeep[idx.s0] = 1;
        vertexKeep[idx.s1] = 1;
        vertexKeep[idx.s2] = 1;
    }
}

/**
 * Compact the triangles to keep only the accepted ones and rewrite
 * the indices to reference the accepted vertices.
 *
 * @param[out] outIndices        The output triangles.
 * @param      remap             Scan of the @a triangleKeep values written by @ref classify.
 * @param      inIndices         The input triangles.
 * @param      indexRemap        Remapping table from original vertex IDs to new ones.
 *
 * @note @a remap must contain one more element than the number of triangles.
 */
__kernel void triangleCompact(
    __global uint * restrict outIndices,
    __global const uint * restrict remap,
    __global const uint * restrict inIndices,
    __global const uint * restrict indexRemap)
{
    uint gid = get_global_id(0);
    uint offset = remap[gid];
    if (offset != remap[gid + 1])   // check whether triangleKeep was true
    {
        uint3 idx = vload3(gid, inIndices);
        idx.s0 = indexRemap[idx.s0];
        idx.s1 = indexRemap[idx.s1];
        idx.s2 = indexRemap[idx.s2];
        vstore3(idx, offset, outIndices);
    }
}

/**
 * Rewrite the vertices and vertex keys to contain only the accepted ones.
 *
 * @param[out] outVertices       Accepted vertices.
 * @param[out] outKeys           Keys for accepted vertices.
 * @param      remap             Scan of the @a vertexKeep values written by @ref classify.
 * @param      inVertices        Original vertices.
 * @param      inKeys            Original vertex keys.
 *
 * @note @a remap must contain one more element than the number of original vertices.
 */
__kernel void vertexCompact(
    __global float * restrict outVertices,
    __global ulong * restrict outKeys,
    __global const uint * restrict remap,
    __global const float * restrict inVertices,
    __global const ulong * restrict inKeys)
{
    uint gid = get_global_id(0);
    uint offset = remap[gid];
    if (offset != remap[gid + 1]) // check whether vertexKeep was true
    {
        float3 vertex = vload3(gid, inVertices);
        vstore3(vertex, offset, outVertices);
        outKeys[offset] = inKeys[gid];
    }
}
