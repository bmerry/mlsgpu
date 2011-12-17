/**
 * @file
 *
 * Required defines:
 * - WGS_X, WGS_Y, WGS_Z
 * - USE_IMAGES
 */

/**
 * Shorthand for defining a kernel with a fixed work group size.
 * This is needed to unconfuse Doxygen's parser.
 */
#define KERNEL(xsize, ysize, zsize) __kernel __attribute__((reqd_work_group_size(xsize, ysize, zsize)))

typedef int command_type;

typedef struct
{
    float4 positionRadius;   // position in xyz, inverse-squared radius in w
    float4 normalQuality;    // normal in xyz, quality metric in w
} Splat;

typedef struct
{
    uint hits;
    float sumW;
} Corner;


/**
 * Turn cell coordinates into a cell code.
 *
 * A code consists of the bits of the (shifted) coordinates interleaved (z
 * major).
 *
 * @todo Investigate preloading this (per axis) to shared memory from a table
 * instead.
 */
inline uint makeCode(int3 xyz)
{
    uint ans = 0;
    uint scale = 1;
    xyz.y <<= 1;  // pre-shift these to avoid shifts inside the loop
    xyz.z <<= 2;
    while (any(xyz != 0))
    {
        uint bits = (xyz.x & 1) | (xyz.y & 2) | (xyz.z & 4);
        ans += bits * scale;
        scale <<= 3;
        xyz >>= 1;
    }
    return ans;
}

void processCorner(command_type start, float3 coord, Corner *out,
                   __global const Splat * restrict splats,
                   __global const command_type * restrict commands)
{
    command_type pos = start;
    while (true)
    {
        command_type cmd = commands[pos];
        if (cmd == -1)
            break;
        if (cmd < 0)
        {
            pos = -2 - cmd;
            cmd = commands[pos];
        }

        __global const Splat *splat = &splats[cmd];
        float4 positionRadius = splat->positionRadius;
        float3 offset = positionRadius.xyz - coord;
        float d = dot(offset, offset) * positionRadius.w;
        if (d < 1.0f)
        {
            float w = 1.0f - d;
            w *= w;
            w *= w;
            w *= splat->normalQuality.w;
            out->hits++;
            out->sumW += w;
        }
        pos++;
    }
}


/**
 * Compute isovalues for all grid corners.
 *
 * @todo Investigate making the global ID the linear ID and reversing @ref makeCode.
 */
KERNEL(WGS_X, WGS_Y, WGS_Z)
void processCorners(
    __global Corner * restrict corners,
    __global const Splat * restrict splats,
    __global const command_type * restrict commands,
    __global const command_type * restrict start,
    float3 gridScale,
    float3 gridBias)
{
    int3 gid = (int3) (get_global_id(0), get_global_id(1), get_global_id(2));
    uint linearId = makeCode(gid);
    command_type myStart = start[linearId];

    Corner corner = {0, 0.0f};
    if (myStart >= 0)
    {
        float3 coord = convert_float3(gid.xyz);
        coord = coord * gridScale + gridBias;
        processCorner(myStart, coord, &corner, splats, commands);
    }
    corners[linearId] = corner;
}
