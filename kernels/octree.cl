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
 * Write splat entries for an octree.
 *
 * Each splat produces up to 8 "entries", consisting of a cell key/splat ID
 * pair. In fact, each splat must produce exactly 8 entries, and for unwanted
 * slots, it must write a cell code of UINT_MAX.
 *
 * The output arrays are slot-major i.e. of layout [8][numsplats] and splat i
 * writes to slots [j][i] for j in 0..7. The number of splats is given by
 * <code>get_global_size(0)</code>.
 *
 * Each workitem corresponds to a single splat.
 *
 * @param keys          The cell codes for the entries.
 * @param values        The splat IDs for the entries.
 * @param splats        The original splats.
 * @param scale,bias    The grid-to-world transformation.
 * @param invScale,invBias The world-to-grid transformation.
 */
__kernel void writeEntries(
    __global uint *keys,
    __global uint *values,
    __global const Splat *splats,
    float3 scale,
    float3 bias,
    float3 invScale,
    float3 invBias)
{
    uint gid = get_global_id(0);
    uint stride = get_global_size(0);
    uint pos = gid;

    float4 positionRadius = splats[gid].positionRadius;
    int3 ilo;
    int shift;
    prepare(&ilo, &shift, positionRadius, invScale, invBias);

    float radius2 = positionRadius.w * positionRadius.w * 1.00001f;
    int3 ofs;
    for (ofs.z = 0; ofs.z < 2; ofs.z++)
        for (ofs.y = 0; ofs.y < 2; ofs.y++)
            for (ofs.x = 0; ofs.x < 2; ofs.x++)
            {
                int3 addr = ilo + ofs;
                uint key = makeCode(addr) | (0x80000000 >> shift);
                bool isect = goodEntry(addr, shift, positionRadius.xyz, radius2, scale, bias);
                key = isect ? key : UINT_MAX;

                values[pos] = gid;
                keys[pos] = key;
                pos += stride;
            }
}

/**
 * Generate an indicator function over the entries that is 2 for the last
 * entry of each key and 1 elsewhere. This is later scanned to determine the
 * mapping between entries and commands.
 *
 * There is one work-item per entry.
 *
 * @param[out] indicator       Indicator function.
 * @param      keys            The cell keys for the input entries.
 *
 * @todo See if clcpp can be extended to allow this to be welded onto
 * the scan kernel.
 * @todo Test whether loading the data to __local first helps.
 */
__kernel void countCommands(
    __global uint *indicator,
    __global const uint *keys)
{
    uint pos = get_global_id(0);
    indicator[pos] = (keys[pos] != keys[pos + 1]) ? 2 : 1;
}

/**
 * Emit the command array for a level of the octree, excluding jumps.
 *
 * There is one workitem per entry.
 *
 * @param[out]  commands       The command array (see @ref SplatTree).
 * @param       commandMap     Mapping from entry to command position.
 * @param       splatIds       The splat IDs written by @ref writeEntries (and sorted).
 */
__kernel void writeSplatIds(
    __global int *commands,
    __global const uint *commandMap,
    __global const uint *splatIds)
{
    uint pos = get_global_id(0);
    commands[commandMap[pos]] = splatIds[pos];
}

inline uint lowerBound(__global const uint *keys, uint keysLen, uint key)
{
    int L = -1;        // < key
    int R = keysLen;   // >= key
    while (R - L > 1)
    {
        int M = (L + R) / 2;
        if (keys[M] >= key)
            R = M;
        else
            L = M;
    }
    return R;
}

/**
 * Writes the start array for jump commands for one level. The codes are divided
 * into groups of size M, each of which is processed by a workgroup of M + 1
 * workitems. The algorithm is
 * -# Binary search M + 1 codes in the list of keys.
 * -# Use __local memory to share these between workitems.
 * -# If the commands for a code are non-empty, compute the start and write a jump
 *    at the end.
 * -# Write to the start array.
 *
 * @param[in,out]  start           Start array for previous and current level.
 * @param[out]     commands        Command array in which to write jump commands.
 * @param          commandMap      Mapping from entry to command positions.
 * @param          keys            Keys written by @ref writeEntries and sorted.
 * @param          numCodes        One more than the highest code to process.
 * @param          keysLen         Length of the keys array.
 * @param          curOffset       Offset added to code to get position in start array on current level.
 * @param          prevOffset      Offset added to parent code to get position in parent start array.
 * @param          keyOffset       Offset added to code to get corresponding key.
 * @param          search          Local memory of M+1 uints.
 */
__kernel void writeStart(
    __global int *start,
    __global int *commands,
    __global const uint *commandMap,
    __global const uint *keys,
    uint numCodes,
    uint keysLen,
    uint curOffset,
    uint prevOffset,
    uint keyOffset,
    __local uint *search)
{
    uint lid = get_global_id(0);
    uint code = get_group_id(0) * (get_local_size(0) - 1) + get_local_id(0);
    uint key = code + keyOffset;

    uint pos = lowerBound(keys, keysLen, key);
    uint posCmd = commandMap[pos];
    search[lid] = posCmd;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid < get_local_size(0) - 1 && code < numCodes)
    {
        int prev = start[(code >> 3) + prevOffset];
        int cur = prev;
        if (keys[pos] == key)
        {
            cur = posCmd;
            uint jumpPos = search[lid + 1] - 1;
            commands[jumpPos] = (prev == -1) ? -1 : -2 - prev;
        }
        start[code + curOffset] = cur;
    }
}

/**
 * Fill a buffer with a constant value.
 */
__kernel void fill(__global int *out, int value)
{
    out[get_global_id(0)] = value;
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
__kernel void transformSplats(__global Splat *splats)
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

__kernel void testMakeCode(__global uint *out, int3 xyz)
{
    *out = makeCode(xyz);
}

__kernel void testLowerBound(__global uint *out, __global const uint *keys, uint keysLen, uint key)
{
    *out = lowerBound(keys, keysLen, key);
}

#endif /* UNIT_TESTS */
