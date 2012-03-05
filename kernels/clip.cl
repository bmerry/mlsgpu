/**
 * Marks all vertices as rejected (@ref classify will mark the relevant ones as accepted).
 */
__kernel void vertexInit(
    __global uint * restrict vertexKeep)
{
    vertexKeep[get_global_id(0)] = 0;
}

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

__kernel void triangleCompact(
    __global uint * restrict outIndices,
    __global const uint * restrict remap,
    __global const uint * restrict inIndices,
    __global const uint * restrict indexRemap)
{
    uint gid = get_global_id(0);
    uint offset = remap[gid];
    if (offset != remap[gid + 1])
    {
        uint3 idx = vload3(gid, inIndices);
        idx.s0 = indexRemap[idx.s0];
        idx.s1 = indexRemap[idx.s1];
        idx.s2 = indexRemap[idx.s2];
        vstore3(idx, offset, outIndices);
    }
}

__kernel void vertexCompact(
    __global float3 * restrict outVertices,
    __global ulong * restrict outKeys,
    __global const uint * restrict remap,
    __global const float3 * restrict inVertices,
    __global const ulong * restrict inKeys)
{
    uint gid = get_global_id(0);
    uint offset = remap[gid];
    if (offset != remap[gid + 1])
    {
        outVertices[offset] = inVertices[gid];
        outKeys[offset] = inKeys[gid];
    }
}
