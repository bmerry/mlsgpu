/**
 * @file
 *
 * Required defines:
 * - WGS_X, WGS_Y, WGS_Z
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

__constant sampler_t nearest = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE;

void processCorner(command_type start, float3 coord, __global Corner *out,
                   __global const Splat * restrict splats,
                   __global const command_type * restrict commands)
{
    Corner corner = {0, 0.0f};
    if (start >= 0)
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

            Splat splat = splats[cmd];
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
            pos++;
        }
    }
    *out = corner;
}


KERNEL(WGS_X, WGS_Y, WGS_Z)
void processCorners(
    __global Corner *corners,
    __global const Splat *splats,
    __global const command_type *commands,
    __read_only image3d_t start,
    float3 gridScale,
    float3 gridBias)
{
    int4 gid = (int4) (get_global_id(0), get_global_id(1), get_global_id(2), 0);
    uint linearId = (gid.z * get_global_size(1) + gid.y) * get_global_size(0) + gid.x;
    command_type myStart = read_imagei(start, nearest, gid).x;

    float3 coord = convert_float3(gid.xyz);
    coord = coord * gridScale + gridBias;
    processCorner(myStart, coord, &corners[linearId], splats, commands);
}
