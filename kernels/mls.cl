/**
 * @file
 *
 * Required defines:
 * - WGS_X, WGS_Y, WGS_Z
 * - FIT_SPHERE (0 or 1)
 * - FIT_PLANE (0 or 1)
 */

/**
 * Shorthand for defining a kernel with a fixed work group size.
 * This is needed to unconfuse Doxygen's parser.
 */
#define KERNEL(xsize, ysize, zsize) __kernel __attribute__((reqd_work_group_size(xsize, ysize, zsize)))

#define RADIUS_CUTOFF 0.99f
#define HITS_CUTOFF 4

#if !defined(WGS_X) || !defined(WGS_Y) || !defined(WGS_Z)
# error "WGS_X, WGS_Y and WGS_Z must all be defined"
#endif
#if !defined(FIT_PLANE) || (FIT_PLANE != 0 && FIT_PLANE != 1)
# error "FIT_PLANE must be defined as 0 or 1"
#endif
#if !defined(FIT_SPHERE) || (FIT_SPHERE != 0 && FIT_SPHERE != 1)
# error "FIT_SPHERE must be defined as 0 or 1"
#endif
#if FIT_PLANE + FIT_SPHERE != 1
# error "Exactly one of FIT_PLANE and FIT_SPHERE must be defined"
#endif

/**
 * The number of workitems that cooperate to load splat IDs.
 */
#define MAX_BUCKET 256

#if WGS_X * WGS_Y * WGS_Z < MAX_BUCKET
# error "The workgroup must have at least MAX_BUCKET elements"
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
    float sumWpp;  // needed for boundary testing
    float sumW;
    float3 sumWn;
    float3 sumWp;
    uint hits;
} PlaneFit;

typedef struct
{
    float a;   // quadratic term
    float3 b;  // linear term
    float c;   // constant term
    float qDen;
    float b2;  // squared length of b
} Sphere;

typedef struct
{
    float3 mean;
    float3 normal;
    float dist;
} Plane;

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
    pf->sumWpp = 0.0f;
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

inline void planeFitAdd(PlaneFit *pf, float w, float3 p, float pp, float3 n)
{
    pf->sumWpp += w * pp;
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
 * Turns an index along a 3D space-filling curve into coordinates.
 *
 * The code consists of the bits of the x, y and z coordinates interleaved
 * (z major). It is thus more-or-less an inverse of @ref makeCode.
 *
 * @todo Investigate computing this from a table instead.
 */
inline int3 decode(uint code)
{
    int3 ans = (int3) (0, 0, 0);
    uint scale = 1;
    while (code >= scale)
    {
        ans.x += code & scale;
        ans.y += (code >> 1) & scale;
        ans.z += (code >> 2) & scale;
        code >>= 2;
        scale <<= 1;
    }
    return ans;
}

inline void fitPlane(const PlaneFit * restrict pf, Plane * restrict out)
{
    out->mean = pf->sumWp / pf->sumW;
    out->normal = normalize(pf->sumWn);
    out->dist = -dot3(out->normal, out->mean);
}

/**
 * Fit an algebraic sphere given cumulated sums.
 * @param      sf      The accumulated sums.
 * @param[out] out     Output parameters for the sphere.
 */
inline void fitSphere(const SphereFit * restrict sf, Sphere * restrict out)
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

    float a = 0.5f * q;
    float3 b = (sf->sumWn - q * sf->sumWp) * invSumW;
    out->a = a;
    out->b = b;
    out->c = (-a * sf->sumWpp - dot3(b, sf->sumWp)) * invSumW;
    out->qDen = qDen;
    out->b2 = dot3(b, b);
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
inline float projectDistOriginSphere(const Sphere * restrict sphere)
{
    // b will always be positive
    return -solveQuadratic(sphere->a, length(sphere->b), sphere->c);
}

/**
 * Projects the local origin onto the sphere.
 */
inline float3 projectOriginSphere(const Sphere * restrict sphere)
{
    float l = solveQuadratic(sphere->a * sphere->b2, sphere->b2, sphere->c);
    return l * sphere->b;
}

inline float projectDistOriginPlane(const Plane * restrict plane)
{
    return plane->dist;
}

/**
 * Projects the local origin onto the plane.
 */
inline float3 projectOriginPlane(const Plane * restrict plane)
{
    return plane->normal * -plane->dist;
}

/**
 * Compute isovalues for all grid corners in a slice. Those with no defined
 * isovalue are assigned a value of NaN.
 *
 * @param[out] corners     The isovalues from a slice.
 * @param      splats      Input splats, in global grid coordinates, and with the inverse squared radius in the w component.
 * @param      commands, start Encoded octree for the local bin
 * @param      startShift  Subsampling shift for octree, times 3.
 * @param      offset      Difference between global grid coordinates and the local region of interest.
 * @param      zStride, zBias See @ref Marching::ImageParams
 * @param      boundaryFactor Value of \f$1 - \gamma^2\f$ where \f$\gamma\f$ is the maximum
 *                         normalised distance between the projection point and the weighted
 *                         center of the region.
 *
 * The local ID is a one-dimension encoding of a 3D local ID (see @ref decode).
 * The group ID specifies which of these 3D blocks we are processing.
 */
KERNEL(WGS_X * WGS_Y * WGS_Z, 1, 1)
void processCorners(
    __write_only image2d_t corners,
    __global const Splat * restrict splats,
    __global const command_type * restrict commands,
    __global const command_type * restrict start,
    uint startShift,
    int3 offset,
    uint zStride,
    int zBias,
    float boundaryFactor)
{
    __local command_type lSplatIds[MAX_BUCKET];
    __local float4 lPositionRadius[MAX_BUCKET];

    int3 wid;  // position of one corner of the workgroup in region coordinates
    wid.x = get_group_id(0) * WGS_X;
    wid.y = get_group_id(1) * WGS_Y;
    wid.z = get_group_id(2) * WGS_Z + get_global_offset(2);
    uint code = makeCode(wid) >> startShift;
    command_type pos = start[code];

    uint lid = get_local_id(0);

    float f = nan(0U);

    if (pos >= 0)
    {
        float3 coord = convert_float3(wid + decode(lid) + offset);

#if FIT_SPHERE
        SphereFit fit;
        sphereFitInit(&fit);
#elif FIT_PLANE
        PlaneFit fit;
        planeFitInit(&fit);
#else
#error "Expected FIT_SPHERE or FIT_PLANE"
#endif

        command_type end = commands[pos++];
        while (pos < end)
        {
            if (lid < MAX_BUCKET)
            {
                command_type lpos = pos + lid;
                command_type mine = (lpos < end) ? commands[lpos] : -1;
                lSplatIds[lid] = mine;
                if (mine >= 0)
                {
                    lPositionRadius[lid] = splats[mine].positionRadius;
                }
            }

            pos += MAX_BUCKET;
            if (pos >= end)
            {
                pos = commands[end];
                end = (pos >= 0) ? commands[pos++] : INT_MIN;
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            for (int i = 0; i < MAX_BUCKET; i++)
            {
                command_type splatId = lSplatIds[i];
                if (splatId < 0)
                {
                    break;
                }

                float4 positionRadius = lPositionRadius[i];
                float3 p = positionRadius.xyz - coord;
                float pp = dot3(p, p);
                float d = pp * positionRadius.w; // .w is the inverse squared radius
                if (d < RADIUS_CUTOFF)
                {
                    __global const Splat *splat = &splats[splatId];
                    float w = 1.0f - d;
                    w *= w; // raise to the 4th power
                    w *= w;
                    w *= splat->normalQuality.w;

#if FIT_SPHERE
                    sphereFitAdd(&fit, w, p, pp, splat->normalQuality.xyz);
#elif FIT_PLANE
                    planeFitAdd(&fit, w, p, pp, splat->normalQuality.xyz);
#else
#error "Expected FIT_SPHERE or FIT_PLANE"
#endif
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (fit.hits >= HITS_CUTOFF)
        {
#if FIT_SPHERE
            Sphere sphere;
            fitSphere(&fit, &sphere);
            float3 a = projectOriginSphere(&sphere);
            float aa = dot3(a, a);
            if (aa < 3.0f)
            {
                float rhs = (fit.sumWpp - 2 * dot3(fit.sumWp, a) + fit.sumW * aa);
                if (sphere.qDen > boundaryFactor * rhs)
                {
                    f = -dot3(sphere.b, a) * half_rsqrt(sphere.b2);
                }
            }
#elif FIT_PLANE
            Plane plane;
            fitPlane(&fit, &plane);
            float3 a = projectOriginPlane(&plane);
            float aa = dot3(a, a);
            if (aa < 3.0f)
            {
                float qDen = fit.sumWpp - dot3(plane.mean, fit.sumWp);
                float rhs = (fit.sumWpp - 2 * dot3(fit.sumWp, a) + fit.sumW * aa);
                if (qDen > boundaryFactor * rhs)
                {
                    f = projectDistOriginPlane(&plane);
                }
            }
#else
#error "Expected FIT_SPHERE or FIT_PLANE"
#endif
        }
    }

    int3 lid3 = decode(lid);
    int3 outCoord = wid + lid3;
    outCoord.y += outCoord.z * zStride + zBias;
    write_imagef(corners, outCoord.xy, f);
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
    Sphere sphere = {p3, (float3) (p0, p1, p2), p4};
    *out = projectDistOriginSphere(&sphere);
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
    Sphere sphere;
    fitSphere(&sf, &sphere);
    out[0] = sphere.b.s0;
    out[1] = sphere.b.s1;
    out[2] = sphere.b.s2;
    out[3] = sphere.a;
    out[4] = sphere.c;
}

__kernel void testMakeCode(__global uint *out, int3 xyz)
{
    *out = makeCode(xyz);
}

__kernel void testDecode(__global int3 *out, uint code)
{
    *out = decode(code);
}

#endif /* UNIT_TESTS */
