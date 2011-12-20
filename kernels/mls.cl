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
    float iso;
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

// This seems to generate fewer instructions than NVIDIA's implementation
inline float dot3(float3 a, float3 b)
{
    return fma(a.x, b.x, fma(a.y, b.y, a.z * b.z));
}

inline void fitSphere(float sumWpp, float sumWpn, float3 sumWp, float3 sumWn, float sumW, uint hits,
                      float params[5])
{
    float invSumW = 1.0f / sumW;
    float3 m = sumWp * invSumW;
    float qNum = sumWpn - dot3(m, sumWn);
    float qDen = sumWpp - dot3(m, sumWp);
    float q = qNum / qDen;
    if (fabs(qDen) < (4 * FLT_EPSILON) * hits * fabs(sumWpp) || !isfinite(q))
    {
        q = 0.0f; // numeric instability
    }

    params[3] = 0.5f * q;
    float3 u = (sumWn - q * sumWp) * invSumW;
    params[4] = -params[3] * sumWpp - dot3(u, sumWp);
    params[0] = u.s0;
    params[1] = u.s1;
    params[2] = u.s2;
}

/**
 * Returns the root of ax^2 + bx + c which is larger (a > 0) or smaller (a < 0).
 * Returns NaN if there are no roots or infinitely many roots.
 */
inline float solveQuadratic(float a, float b, float c)
{
    float x;
    if (fabs(a) < 1e-20f)
    {
        // Treat as linear to get initial estimate
        x = -c / b;
    }
    else
    {
        // Start with a closed-form but numerically unstable solution
        float det = sqrt(b * b - 4 * a * c);
        x = (-b + det) / (2.0f * a);
    }
    // Refine using Newton iteration
    for (uint pass = 0; pass < 1; pass++)
    {
        float fx = fma(fma(a, x, b), x, c);
        float fpx = fma(2.0f * a, x, b);
        // Prevent divide by zero when at the critical point
        fpx = maxmag(fpx, 1e-20f);
        x -= fx / fpx;
    }
    return isfinite(x) ? x : nan(0U);
}

inline float projectDist(const float params[5], float3 origin, float3 p)
{
    const float3 d = p - origin;
    const float3 u = (float3) (params[0], params[1], params[2]);

    float3 g = u + 2.0f * params[3] * d; // gradient
    float3 dir = normalize(g);
    if (dot3(dir, dir) < 0.5f)
    {
        // g was exactly 0, i.e. the centre of the sphere. Pick any
        // direction.
        dir = (float3) (0.0f, 0.0f, 0.0f);
    }

    float a = params[3];
    float b = dot3(dir, g);
    float c = dot3(d, u) + params[3] * dot3(d, d) + params[4];
    // b will always be positive
    return -solveQuadratic(a, b, c);
}

void processCorner(command_type start, float3 coord, Corner *out,
                   __global const Splat * restrict splats,
                   __global const command_type * restrict commands)
{
    command_type pos = start;

    float3 sumWp = (float3) (0.0f, 0.0f, 0.0f);
    float3 sumWn = (float3) (0.0f, 0.0f, 0.0f);
    float sumWpn = 0.0f, sumWpp = 0.0f, sumW = 0.0f;
    float3 origin;
    uint hits = 0;
    while (true)
    {
        command_type cmd = commands[pos];
        if (cmd < 0)
        {
            if (cmd == -1)
                break;
            pos = -2 - cmd;
            cmd = commands[pos];
        }

        __global const Splat *splat = &splats[cmd];
        float4 positionRadius = splat->positionRadius;
        float3 offset = positionRadius.xyz - coord;
        float d = dot3(offset, offset) * positionRadius.w;
        if (d < 1.0f)
        {
            float w = 1.0f - d;
            w *= w; // raise to the 4th power
            w *= w;
            w *= splat->normalQuality.w;

            if (hits == 0)
                origin = positionRadius.xyz;
            hits++;
            sumW += w;
            float3 p = positionRadius.xyz - origin;
            float3 wp = w * p;
            float3 wn = w * splat->normalQuality.xyz;
            sumWpp += dot3(wp, p);
            sumWpn += dot3(wn, p);
            sumWp += wp;
            sumWn += wn;
        }
        pos++;
    }
    out->hits = hits;
    if (hits >= 4)
    {
        float params[5];
        fitSphere(sumWpp, sumWpn, sumWp, sumWn, sumW, hits, params);
        out->iso = projectDist(params, origin, coord);
    }
    else
    {
        out->iso = nan(0U);
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
    float3 gridBias,
    uint startShift,
    int cornerOffset)
{
    int3 gid = (int3) (get_global_id(0), get_global_id(1), get_global_id(2));
    uint code = makeCode(gid) >> startShift;
    command_type myStart = start[code];
    uint linearId = (gid.z * get_global_size(1) + gid.y) * get_global_size(0) + gid.x + cornerOffset;

    Corner corner = {0, 0.0f};
    if (myStart >= 0)
    {
        float3 coord = convert_float3(gid.xyz);
        coord = coord * gridScale + gridBias;
        processCorner(myStart, coord, &corner, splats, commands);
    }
    corners[linearId] = corner;
}

/*******************************************************************************
 * Test code only below here.
 *******************************************************************************/

#if UNIT_TESTS

__kernel void testSolveQuadratic(__global float *out, float a, float b, float c)
{
    *out = solveQuadratic(a, b, c);
}

#endif /* UNIT_TESTS */
