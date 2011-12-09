/**
 * @file
 *
 * Required defines:
 * - OCTREE_LEVELS
 */

/**
 * Shorthand for defining a kernel with a fixed work group size.
 * This is needed to unconfuse Doxygen's parser.
 */
#define KERNEL(xsize, ysize, zsize) __kernel __attribute__((reqd_work_group_size(xsize, ysize, zsize)))

typedef struct
{
    float4 positionRadius;   // position in xyz, inverse-squared radius in w
    float4 normalQuality;    // normal in xyz, quality metric in w
} Splat;

typedef struct
{
    __global const Splat *splats;
    __global const uint *splatIds;
    __global const uint *startOffsets[OCTREE_LEVELS];
} Octree;

typedef struct
{
    uint hits;
    float sumW;
} Corner;

__constant sampler_t nearest = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE;

void processCorner(float3 coord, uint lcoord, __global Corner *out, __local const Octree *ot)
{
    int level = OCTREE_LEVELS - 1;
    uint start = ot->startOffsets[level][lcoord];
    uint end = ot->startOffsets[level][lcoord + 1];
    uint pos = start;

    Corner corner = {0, 0.0f};
    while (true)
    {
        while (pos == end)
        {
            if (level == 0)
            {
                goto end;
            }
            level--;
            lcoord >>= 3;
            start = ot->startOffsets[level][lcoord];
            end = ot->startOffsets[level][lcoord + 1];
            pos = start;
        }

        uint splatId = ot->splatIds[pos++];
        Splat splat = ot->splats[splatId];
        float3 offset = splat.positionRadius.xyz - coord;
        float d = dot(offset, offset) * splat.positionRadius.w;
        if (d < 1.0f)
        {
            float w = 1.0f - d;
            w *= w;
            w *= w;
            w *= splat.normalQuality.w;
            corner.hits++;
            corner.sumW += w;
        }
    }
end:
    *out = corner;
}


/**
 * The @a shuffleBits image is an Nx3 image, with rows for x, y, z. Each element
 * is a single uint.
 *
 * Row 0, column i contains the bits of i spread out to every third bit e.g.
 * ABCDb maps to A00B00C00Db. The values for row 1 and row 2 are the same but
 * shifted one and two bits left.
 */
KERNEL(4, 4, 8)
void processCorners(
    __global Corner *corners,
    __global const Splat *splats,
    __global const uint *splatIds,
    __global const uint *startOffsets,
    __read_only image2d_t shuffleBits,
    float3 gridScale,
    float3 gridBias)
{
    __local Octree ot;

    uint gx = get_global_id(0);
    uint gy = get_global_id(1);
    uint gz = get_global_id(2);
    uint lcoord =
        read_imageui(shuffleBits, nearest, (int2) (gx, 0) ).x
        | read_imageui(shuffleBits, nearest, (int2) (gy, 1) ).x
        | read_imageui(shuffleBits, nearest, (int2) (gz, 2) ).x;
    bool master = get_local_id(0) == 0 && get_local_id(1) == 0 && get_local_id(2) == 0;

    if (master)
    {
        ot.splats = splats;
        ot.splatIds = splatIds;
        uint levelOffset = 0;
        for (uint i = 0; i < OCTREE_LEVELS; i++)
        {
            ot.startOffsets[i] = startOffsets + levelOffset;
            levelOffset += 1U << (3 * i);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float3 coord = (float3) (gx, gy, gz);
    coord = coord * gridScale + gridBias;
    processCorner(coord, lcoord, &corners[lcoord], &ot);
}
