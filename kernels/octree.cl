/**
 * @file
 *
 * Construction of an octree containing splats.
 */

/**
 * GPU representation of a splat.
 * Only the position and radius are used in this file, but the full set of information
 * is there for compatibility with other files.
 */
typedef struct
{
    float4 positionRadius;   // position in xyz, radius in w
    float4 normalQuality;    // normal in xyz, quality metric in w
} Splat;

/**
 * Determine the number of bits to shift off ilo and ihi so that they
 * differ by at most 1.
 *
 * ihi = aaa1000???? and
 * ilo = aaa0111????.
 * where the number of ?'s is minimum (ihi is 1 or ilo is 0 at the first
 * one). Then shifting off the ?'s is the minimum shift such that the
 * difference becomes 1. The minimum difference is
 *       aaa10001000
 *     - aaa01111111
 *     = 00000001001
 * and the maximum is 11111. Thus, the shift count is either the number of
 * bits in ihi-ilo, or one less.
 */
int levelShift(int3 ilo, int3 ihi)
{
    int3 diff = max(ihi - ilo, 1);
    int3 first = 31 - clz(diff); // one less than the number of bits in diff
    // Vector comparisons return -1 for true, hence -= instead of +=
    first -= ((ihi >> first) - (ilo >> first) > 1);
    return max(first.x, max(first.y, first.z));
}

/**
 * Compute the squared distance of a point from a solid box.
 */
float pointBoxDist2(float3 pos, float3 lo, float3 hi)
{
    // Find nearest point in the box to pos.
    float3 nearest = max(lo, min(hi, pos));
    float3 dist = nearest - pos;
    return dot(dist, dist);
}

/**
 * Transforms a splat to grid coordinates and computes the
 * coordinates for the first cell it is to be placed in.
 *
 * @param[out]  ilo             Coordinates for the first cell.
 * @param[out]  shift           Bits shifted off to produce @a ilo.
 * @param       positionRadius  Position (xyz) and radius (w) of splat.
 * @param       invScale,invBias Transformation from world to grid coordinates
 */
inline void prepare(
    int3 *ilo, int *shift,
    float4 positionRadius, float3 invScale, float3 invBias)
{
    float3 vlo = positionRadius.xyz - positionRadius.w;
    float3 vhi = positionRadius.xyz + positionRadius.w;
    vlo = vlo * invScale + invBias;
    vhi = vhi * invScale + invBias;
    *ilo = convert_int3_rtp(vlo);
    int3 ihi = convert_int3_rtn(vhi);
    *shift = levelShift(*ilo, ihi);
    *ilo >>= *shift;
}

/**
 * Determines whether a splat intersects a given cell.
 *
 * @param cell            Cell coordinates (pre-shifted).
 * @param shift           Bits shifted off to give @a cell.
 * @param position        Splat position in world coordinates.
 * @param radius2         Squared splat radius in world coordinates.
 * @param scale,bias      Transformation from grid to world coordinates.
 */
inline bool goodEntry(
    int3 cell, int shift,
    float3 position, float radius2,
    float3 scale, float3 bias)
{
    int3 blo = cell << shift;
    int3 bhi = ((cell + 1) << shift) - 1;
    float3 vblo = convert_float3(blo) * scale + bias;
    float3 vbhi = convert_float3(bhi) * scale + bias;
    return pointBoxDist2(position, vblo, vbhi) < radius2;
}

/**
 * Turn cell coordinates into a cell code.
 *
 * A code consists of the bits of the (shifted) coordinates interleaved (z
 * major), with an offset of (8^level / 4) to make room for prior levels.
 *
 * @todo Investigate using an image lookup instead, and also compacting
 * the codes better using a table.
 */
inline uint makeCode(int3 xyz, int level)
{
    uint ans = 0;
    uint scale = 1;
    xyz.y <<= 1;  // pre-shift these to avoid shifts inside the loop
    xyz.z <<= 2;
    for (unsigned int i = 0; i < level; i++)
    {
        uint bits = (xyz.x & 1) | (xyz.y & 2) | (xyz.z & 4);
        ans += bits * scale;
        scale <<= 3;
        xyz >>= 1;
    }
    ans += scale / 4;
    return ans;
}

/**
 * Write splat entries for an octree.
 *
 * Each splat produces up to 8 "entries", consisting of a cell code/splat ID
 * pair. In fact, each splat must produce exactly 8 entries, and for unwanted
 * slots, it must write a cell code of UINT_MAX.
 *
 * The output arrays are slot-major i.e. of layout [8][numsplats] and splat i
 * writes to slots [j][i] for j in 0..7. The number of splats is given by
 * <code>get_global_size(0)</code>.
 *
 * @param keys          The cell codes for the entries.
 * @param values        The splat IDs for the entries.
 * @param splats        The original splats.
 * @param scale,bias    The grid-to-world transformation.
 * @param invScale,invBias The world-to-grid transformation.
 * @param maxLevel      One less than the number of levels in the octree.
 */
__kernel void writeEntries(
    __global uint *keys,
    __global uint *values,
    __global const Splat *splats,
    float3 scale,
    float3 bias,
    float3 invScale,
    float3 invBias,
    int maxLevel)
{
    uint gid = get_global_id(0);
    uint stride = get_global_size(0);
    uint pos = gid;

    float4 positionRadius = splats[gid].positionRadius;
    int3 ilo;
    int shift;
    prepare(&ilo, &shift, positionRadius, invScale, invBias);
    int level = maxLevel - shift;

    float radius2 = positionRadius.w * positionRadius.w * 1.00001f;
    int3 ofs;
    for (ofs.z = 0; ofs.z < 2; ofs.z++)
        for (ofs.y = 0; ofs.y < 2; ofs.y++)
            for (ofs.x = 0; ofs.x < 2; ofs.x++)
            {
                int3 addr = ilo + ofs;
                bool isect = goodEntry(addr, shift, positionRadius.xyz, radius2, scale, bias);
                uint key = isect ? makeCode(addr, level) : UINT_MAX;

                values[pos] = gid;
                keys[pos] = key;
                pos += stride;
            }
}

/**
 * Binary search a sorted for the section equal to a value.
 *
 * @param codes         The array to search.
 * @param codesLen      The length of @a codes.
 * @param code          The search key.
 *
 * @pre
 * - @a codes is increasing.
 * - @a codes ends with a value strictly greater than @a code (e.g. @c UINT_MAX).
 *
 * @return index to first and to one past the end within @a codes
 */
inline uint2 findRange(__global const uint *codes, uint codesLen, uint code)
{
    int L = -1;             // index of something strictly less than code
    int R = codesLen - 1;   // index of something greater than or equal to code
    while (R - L > 1)
    {
        uint M = (L + R) / 2;
        if (codes[M] >= code)
            R = M;
        else
            L = M;
    }
    int E = R;
    while (codes[E] == code)
        E++;
    return (uint2) (R, E);
}

/**
 * Count the number of spots in the command array for each cell.
 * See @ref SplatTree for the definition of the command array.
 *
 * This kernel is launched once per level of the octree, in increasing
 * levels. The global ids are offset such that the global id is always
 * the code to consider.
 *
 * @param[out] sizes           Number of entries needed for cell.
 * @param[out] ranges          Range of values to copy in to slot.
 * @param      codes           The cell codes for the input entries.
 * @param      codesLen        The length of the @a codes array.
 * @param      level           The level being processed.
 *
 * @todo Rewrite this to emit just start values (rather than start+end), since
 * the following code's start is our end.
 */
__kernel void countLevel(
    __global int *sizes,
    __global uint2 *ranges,
    __global const uint *codes,
    uint codesLen)
{
    uint code = get_global_id(0);
    uint2 range = findRange(codes, codesLen, code);
    ranges[code] = range;
    sizes[code] = (range.y - range.x) + (range.x != range.y);
}

/**
 * Emit the command buffer for a level of the octree.
 * This kernel is launched in the same way as @ref countLevel.
 *
 * @param[in,out] codeStart    On input, the position in the command array allocated
 *                             for writing (even if there is nothing to write). On output,
 *                             the start array (see @ref SplatTree).
 * @param[out]    commands     The command array (see @ref SplatTree).
 * @param         ranges       The ranges of entries for each code, as written by @ref countLevel.
 * @param         splatIds     The splat IDs written by @ref writeEntries.
 */
__kernel void writeLevel(
    __global int *codeStart,
    __global int *commands,
    __global const uint2 *ranges,
    __global const uint *splatIds)
{
    uint code = get_global_id(0);
    uint2 range = ranges[code];

    int prev = codeStart[code >> 3];
    if (range.x == range.y)
    {
        codeStart[code] = prev;
    }
    else
    {
        int s = codeStart[code];
        commands += s;
        for (uint i = range.x; i < range.y; i++)
        {
            *commands++ = splatIds[i];
        }
        *commands = (prev == -1) ? -1 : -2 - prev;
    }
}

/**
 * Turn the splats from the form they were used in computing the octree to
 * the form they will be used in by @ref mls.cl.
 *
 * The latter stores an inverse-squared radius instead of the raw radius.
 * @param[in,out] splats     The splats to transform (one per launch).
 *
 * @todo NVIDIA is compiling this to load the float4 and write it all back
 * again.
 */
__kernel void transformSplats(
    __global Splat *splats)
{
    uint gid = get_global_id(0);
    __global Splat *splat = splats + gid;
    float radius = splat->positionRadius.w;
    splat->positionRadius.w = 1.0f / (radius * radius);
}


/*******************************************************************************
 * Test code only below here.
 *******************************************************************************/

#if UNIT_TESTS

__kernel void testLevelShift(__global int *out, int3 ilo, int3 ihi)
{
    *out = levelShift(ilo, ihi);
}

__kernel void testPointBoxDist2(__global float *out, float3 pos, float3 lo, float3 hi)
{
    *out = pointBoxDist2(pos, lo, hi);
}

__kernel void testMakeCode(__global uint *out, int3 xyz, int level)
{
    *out = makeCode(xyz, level);
}

__kernel void testFindRange(__global uint2 *out, __global const uint *codes, uint codesLen, uint code)
{
    *out = findRange(codes, codesLen, code);
}

#endif /* UNIT_TESTS */
