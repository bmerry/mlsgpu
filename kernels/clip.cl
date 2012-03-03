__kernel void triangleClassifyKernel(
    __global uint * restrict triangleKeep,
    __global uint * restrict vertexKeep,
    __global const uint * restrict indices,
    __global const float * restrict vertexDist)
{
    uint gid = get_global_id(0);
    uint3 idx = vload3(gid, indices);
    bool keep = vertexDist[idx.s0] <= 0.0f || vertexDist[idx.s1] <= 0.0f || vertexDist[idx.s2] <= 0.0f;
    triangleKeep[gid] = keep;
    if (keep)
    {
        vertexKeep[idx.s0] = 1;
        vertexKeep[idx.s1] = 1;
        vertexKeep[idx.s2] = 1;
    }
}

__kernel void triangleCompactKernel(
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

__kernel void vertexCompactKernel(
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
