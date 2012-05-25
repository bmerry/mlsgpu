/**
 * @file
 *
 * Required defines:
 * - WGS_X, WGS_Y
 * - USE_IMAGES
 */

/**
 * Shorthand for defining a kernel with a fixed work group size.
 * This is needed to unconfuse Doxygen's parser.
 */
#define KERNEL(xsize, ysize, zsize) __kernel __attribute__((reqd_work_group_size(xsize, ysize, zsize)))

#define RADIUS_CUTOFF 0.99f
#define HITS_CUTOFF 4

#if !defined(FIT_PLANE) || (FIT_PLANE != 0 && FIT_PLANE != 1)
# error "FIT_PLANE must be defined as 0 or 1"
#endif
#if !defined(FIT_SPHERE) || (FIT_SPHERE != 0 && FIT_SPHERE != 1)
# error "FIT_SPHERE must be defined as 0 or 1"
#endif
#if FIT_PLANE + FIT_SPHERE != 1
# error "Exactly one of FIT_PLANE and FIT_SPHERE must be defined"
#endif

typedef int command_type;

typedef struct
{
    float4 positionRadius;   // position in xyz, inverse-squared radius in w
    float4 normalQuality;    // normal in xyz, quality metric in w
} Splat;

typedef struct
{
    float sumWpp;
    float sumWpn;
    float3 sumWp;
    float3 sumWn;
    float sumW;
    uint hits;
} SphereFit;

typedef struct
{
    float sumW;
    float3 sumWn;
    float3 sumWp;
    uint hits;
} PlaneFit;

// This seems to generate fewer instructions than NVIDIA's implementation
inline float dot3(float3 a, float3 b)
{
    return fma(a.x, b.x, fma(a.y, b.y, a.z * b.z));
}

inline void sphereFitInit(SphereFit *sf)
{
    sf->sumWpp = 0.0f;
    sf->sumWpn = 0.0f;
    sf->sumWp = (float3) (0.0f, 0.0f, 0.0f);
    sf->sumWn = (float3) (0.0f, 0.0f, 0.0f);
    sf->sumW = 0.0f;
    sf->hits = 0;
}

inline void planeFitInit(PlaneFit *pf)
{
    pf->sumW = 0.0f;
    pf->sumWp = (float3) (0.0f, 0.0f, 0.0f);
    pf->sumWn = (float3) (0.0f, 0.0f, 0.0f);
    pf->hits = 0;
}

inline void sphereFitAdd(SphereFit *sf, float w, float3 p, float pp, float3 n)
{
    float3 wp = w * p;
    float3 wn = w * n;
    sf->sumW += w;
    sf->sumWp += wp;
    sf->sumWn += wn;
    sf->sumWpp += w * pp;
    sf->sumWpn += dot3(wn, p);
    sf->hits++;
}

inline void planeFitAdd(PlaneFit *pf, float w, float3 p, float3 n)
{
    pf->sumW += w;
    pf->sumWp += w * p;
    pf->sumWn += w * n;
    pf->hits++;
}

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

/**
 * Turns an index along a 2D space-filling curve into coordinates.
 *
 * The code consists of the bits of the x and y coordinates interleaved
 * (y major). It is thus more-or-less an inverse of @ref makeCode, but in 2D.
 *
 * @todo Investigate computing this from a table instead.
 */
inline int2 decode(uint code)
{
    int2 ans = (int2) (0, 0);
    uint scale = 1;
    while (code >= scale)
    {
        ans.x += code & scale;
        ans.y += (code >> 1) & scale;
        code >>= 1;
        scale <<= 1;
    }
    return ans;
}

inline float4 fitPlane(const PlaneFit * restrict pf)
{
    float4 ans;
    ans.xyz = normalize(pf->sumWn);
    ans.w = -dot3(ans.xyz, pf->sumWp / pf->sumW);
    return ans;
}

/**
 * Fit an algebraic sphere given cumulated sums.
 * @param      sf      The accumulated sums.
 * @param[out] params  Output parameters for the sphere.
 */
inline void fitSphere(const SphereFit * restrict sf, float params[restrict 5])
{
    float invSumW = 1.0f / sf->sumW;
    float3 m = sf->sumWp * invSumW;
    float qNum = sf->sumWpn - dot3(m, sf->sumWn);
    float qDen = sf->sumWpp - dot3(m, sf->sumWp);
    float q = qNum / qDen;
    if (fabs(qDen) < (4 * FLT_EPSILON) * sf->hits * fabs(sf->sumWpp) || !isfinite(q))
    {
        q = 0.0f; // numeric instability
    }

    params[3] = 0.5f * q;
    float3 u = (sf->sumWn - q * sf->sumWp) * invSumW;
    params[4] = (-params[3] * sf->sumWpp - dot3(u, sf->sumWp)) * invSumW;
    params[0] = u.s0;
    params[1] = u.s1;
    params[2] = u.s2;
}

/**
 * Returns the root of ax^2 + bx + c which is larger (a > 0) or smaller (a < 0).
 * Returns NaN if there are no roots or infinitely many roots.
 *
 * @pre b &gt;= 0.
 */
inline float solveQuadratic(float a, float b, float c)
{
    float bdet = b + sqrt(b * b - 4.0f * a * c);
    float x = -2.0f * c / bdet;
    if (!isfinite(x))
    {
        // happens if either b = 0 and ac = 0, or if the quadratic
        // has no real solutions
        x = bdet / (-2.0f * a);
    }
    return isfinite(x) ? x : nan(0U);
}

/**
 * Computes the signed distance of the (local) origin to the sphere.
 * It is positive outside and negative inside, or vice versa for an inside-out sphere.
 */
inline float projectDistOriginSphere(const float params[5])
{
    const float3 g = (float3) (params[0], params[1], params[2]);

    // b will always be positive
    return -solveQuadratic(params[3], length(g), params[4]);
}

inline float projectDistOriginPlane(const float4 params)
{
    return params.w;
}

float processCorner(command_type start, int3 coord,
                    __global const Splat * restrict splats,
                    __global const command_type * restrict commands)
{
    command_type pos = start;

#if FIT_SPHERE
    SphereFit fit;
    sphereFitInit(&fit);
#elif FIT_PLANE
    PlaneFit fit;
    planeFitInit(&fit);
#else
#error "Expected FIT_SPHERE or FIT_PLANE"
#endif
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
        float3 p = positionRadius.xyz - convert_float3(coord);
        float pp = dot3(p, p);
        float d = pp * positionRadius.w; // .w is the inverse squared radius
        if (d < RADIUS_CUTOFF)
        {
            float w = 1.0f - d;
            w *= w; // raise to the 4th power
            w *= w;
            w *= splat->normalQuality.w;

#if FIT_SPHERE
            sphereFitAdd(&fit, w, p, pp, splat->normalQuality.xyz);
#elif FIT_PLANE
            planeFitAdd(&fit, w, p, splat->normalQuality.xyz);
#else
#error "Expected FIT_SPHERE or FIT_PLANE"
#endif
        }
        pos++;
    }

    float ans;
    if (fit.hits >= HITS_CUTOFF)
    {
#if FIT_SPHERE
        float params[5];
        fitSphere(&fit, params);
        ans = projectDistOriginSphere(params);
#elif FIT_PLANE
        ans = projectDistOriginPlane(fitPlane(&fit));
#else
#error "Expected FIT_SPHERE or FIT_PLANE"
#endif
    }
    else
    {
        ans = nan(0U);
    }
    return ans;
}


/**
 * Compute isovalues for all grid corners in a slice. Those with no defined
 * isovalue are assigned a value of NaN.
 *
 * @param[out] corners     The isovalues from a slice.
 * @param      splats      Input splats, in global grid coordinates, and with the inverse squared radius in the w component.
 * @param      commands, start Encoded octree for the region of interest.
 * @param      startShift  Subsampling shift for octree, times 3.
 * @param      offset      Difference between global grid coordinates and the local region of interest.
 * @param      z           Z value of the slice, in local region coordinates.
 *
 * The global ID uses all three dimensions:
 * X: linear ID within the workgroup, turned into coordinates with @ref code
 * Y: X position of the workgroup block
 * Z: Y position of the workgroup block
 */
KERNEL(WGS_X * WGS_Y, 1, 1)
void processCorners(
    __write_only image2d_t corners,
    __global const Splat * restrict splats,
    __global const command_type * restrict commands,
    __global const command_type * restrict start,
    uint startShift,
    int3 offset,
    int z)
{
    int3 gid;
    gid.x = get_global_id(1) * WGS_X;
    gid.y = get_global_id(2) * WGS_Y;
    gid.xy += decode(get_global_id(0));
    gid.z = z;
    uint code = makeCode(gid) >> startShift;
    command_type myStart = start[code];

    float f = nan(0U);
    if (myStart >= 0)
    {
        int3 coord = gid + offset;
        f = processCorner(myStart, coord, splats, commands);
    }
    write_imagef(corners, gid.xy, f);
}

/**
 * Boundary detector based on a half-disc criterion (see Bendels et al, Detecting Holes
 * in Point Set Surfaces). There is one work-item per vertex to classify.
 *
 * @param[out] discriminant       Signed distances to the boundary (negative for inside).
 * @param      vertices           Vertices to process, as tightly-packed xyz tuples.
 * @param      splats,commands,start,startShift,offset See @ref processCorners.
 * @param      boundaryFactor     Inverse of scaled boundary limit.
 */
__kernel
void measureBoundaries(
    __global float * restrict discriminant,
    __global const float * restrict vertices,
    __global const Splat * restrict splats,
    __global const command_type * restrict commands,
    __global const command_type * restrict start,
    uint startShift,
    int3 offset,
    float boundaryFactor)
{
    int gid = get_global_id(0);
    float3 vertex = vload3(gid, vertices);
    int3 coord = convert_int3_rtn(vertex) - offset;
    uint code = makeCode(coord) >> startShift;
    command_type myStart = start[code];

    float f = 1000.0f;
    if (myStart >= 0)
    {
        command_type pos = myStart;
        float3 sumWp = (float3) (0.0f, 0.0f, 0.0f);
        float sumWpp = 0.0f;
        float sumW = 0.0f;
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
            float3 p = positionRadius.xyz - vertex;
            float pp = dot3(p, p);
            float d = pp * positionRadius.w; // .w is the inverse squared radius
            if (d < RADIUS_CUTOFF)
            {
                float w = 1.0f - d;
                w *= w; // raise to the 4th power
                w *= w;
                w *= splat->normalQuality.w;

                sumWp += w * p;
                sumWpp += w * pp;
                sumW += w;
                hits++;
            }
            pos++;
        }
        if (hits >= HITS_CUTOFF)
        {
            // (sqrt(6) * 512) / (693 * pi) (based on weight function)
            const float factor = 0.5760530f;
            const float factor2 = factor * factor;

            float3 meanPos = sumWp;       // mean position scaled by W
            float scale2 = sumWpp * sumW; // mean squared (distance * W)
            f = fma(sqrt(dot3(meanPos, meanPos) / scale2), boundaryFactor, - 1.0f);
        }
    }
    discriminant[gid] = f;
}

/*******************************************************************************
 * Test code only below here.
 *******************************************************************************/

#if UNIT_TESTS

__kernel void testSolveQuadratic(__global float *out, float a, float b, float c)
{
    *out = solveQuadratic(a, b, c);
}

__kernel void testProjectDistOriginSphere(__global float *out, float p0, float p1, float p2, float p3, float p4)
{
    float params[5] = {p0, p1, p2, p3, p4};
    *out = projectDistOriginSphere(params);
}

__kernel void testFitSphere(__global float *out, __global const Splat *in, uint nsplats)
{
    SphereFit sf;
    sphereFitInit(&sf);
    for (uint i = 0; i < nsplats; i++)
    {
        const float3 p = in[i].positionRadius.xyz;
        const float3 n = in[i].normalQuality.xyz;
        sphereFitAdd(&sf, in[i].normalQuality.w, p, dot3(p, p), n);
    }
    float params[5];
    fitSphere(&sf, params);
    for (uint i = 0; i < 5; i++)
        out[i] = params[i];
}

__kernel void testMakeCode(__global uint *out, int3 xyz)
{
    *out = makeCode(xyz);
}

__kernel void testDecode(__global int2 *out, uint code)
{
    *out = decode(code);
}

#endif /* UNIT_TESTS */
