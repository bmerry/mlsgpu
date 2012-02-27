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
 * differ by at most 1 in any dimension.
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
 * Transforms a splat to cell coordinates and computes the
 * coordinates for the first cell it is to be placed in.
 *
 * @param[out]  ilo             Coordinates for the first cell.
 * @param[out]  shift           Bits shifted off to produce @a ilo.
 * @param       minShift        Minimum allowed shift.
 * @param       maxShift        Maximum allowed shift (one less than number of levels).
 * @param       positionRadius  Position (xyz) and radius (w) of splat.
 * @param       invScale,invBias Transformation from world to grid coordinates.
 */
inline void prepare(
    int3 *ilo, int *shift, int minShift, int maxShift,
    float4 positionRadius, float3 invScale, float3 invBias)
{
    float3 vlo = positionRadius.xyz - positionRadius.w;
    float3 vhi = positionRadius.xyz + positionRadius.w;
    vlo = vlo * invScale + invBias;
    vhi = vhi * invScale + invBias;
    *ilo = max(convert_int3_rtn(vlo), 0);
    int3 ihi = convert_int3_rtn(vhi);
    *shift = clamp(levelShift(*ilo, ihi), minShift, maxShift);
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
    int3 bhi = ((cell + 1) << shift);
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
 * @param keys             The cell codes for the entries.
 * @param values           The splat IDs for the entries.
 * @param[in,out] splats   The original splats. On output, radius replaced by 1/radius^2.
 * @param scale,bias       The grid-to-world transformation.
 * @param invScale,invBias The world-to-grid transformation.
 * @param levelOffsets     Values added to codes to give sort keys (allocated to hold @a maxShift + 1 values).
 * @param minShift         Minimum bit shift (determines subsampling of grid to give finest level).
 * @param maxShift         Maximum bit shift (determines base level).
 */
__kernel void writeEntries(
    __global uint *keys,
    __global uint *values,
    __global Splat *splats,
    float scale,
    float3 bias,
    float invScale,
    float3 invBias,
    __local uint *levelOffsets,
    uint minShift,
    uint maxShift)
{
    if (get_local_id(0) == 0)
    {
        // TODO: compute in parallel, as long as splats array is big enough
        uint pos = 0;
        uint add = 1U << (3 * (maxShift - minShift));
        for (uint i = minShift; i <= maxShift; i++)
        {
            levelOffsets[i] = pos;
            pos += add;
            add >>= 3;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint gid = get_global_id(0);
    uint stride = get_global_size(0);
    uint pos = gid;

    float4 positionRadius = splats[gid].positionRadius;
    int3 ilo;
    int shift;
    prepare(&ilo, &shift, minShift, maxShift, positionRadius, invScale, invBias);

    float radius2 = positionRadius.w * positionRadius.w;
    splats[gid].positionRadius.w = 1.0f / radius2; // replace with form used in mls.cl
    radius2 *= 1.00001f;   // be conservative in deciding intersections
    int3 ofs;
    uint levelOffset = levelOffsets[shift];
    int bound = 1 << (maxShift - shift);
    for (ofs.z = 0; ofs.z < 2; ofs.z++)
        for (ofs.y = 0; ofs.y < 2; ofs.y++)
            for (ofs.x = 0; ofs.x < 2; ofs.x++)
            {
                int3 addr = ilo + ofs;
                uint key = makeCode(addr) + levelOffset;
                bool isect = goodEntry(addr, shift, positionRadius.xyz, radius2, scale, bias);
                // Avoid going outside the octree bounds. ilo was already clamped to >= 0 in
                // prepare so we don't need to worry about the lower bound
                isect &= all(addr < bound);
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
 * There is one work-item per entry, excluding the last one.
 *
 * @param[out] indicator       Indicator function.
 * @param      keys            The cell keys for the input entries.
 *
 * @todo See if clogs can be extended to allow this to be welded onto
 * the scan kernel.
 * @todo Test whether loading the data to __local first helps.
 */
__kernel void countCommands(
    __global uint *indicator,
    __global const uint *keys)
{
    uint pos = get_global_id(0);
    uint curKey = keys[pos];
    uint nextKey = keys[pos + 1];
    bool end = curKey != nextKey;
    indicator[pos] = end ? 2 : 1;
}

/**
 * Emit the command array for a level of the octree, excluding jumps.
 * Also computes the start and jumpPos for each range of commands.
 *
 * There is one workitem per entry.
 *
 * @param[out]  commands       The command array (see @ref SplatTree).
 * @param[out]  start          First command for each code.
 * @param[out]  jumpPos        Position in command array of jump commands.
 * @param       commandMap     Mapping from entry to command position.
 * @param       keys           Sorted keys written by @ref writeEntries.
 * @param       splatIds       The splat IDs written by @ref writeEntries (and sorted).
 *
 * @todo Investigate using local memory to avoid multiple global reads.
 */
__kernel void writeSplatIds(
    __global int *commands,
    __global int *start,
    __global int *jumpPos,
    __global const uint *commandMap,
    __global const uint *keys,
    __global const uint *splatIds)
{
    uint pos = get_global_id(0);
    uint curKey = keys[pos];

    if (curKey != UINT_MAX)
    {
        uint cpos = commandMap[pos];
        commands[cpos] = splatIds[pos];

        uint prevKey = pos > 0 ? keys[pos - 1] : UINT_MAX;
        uint nextKey = (pos < get_global_size(0) - 1) ? keys[pos + 1] : UINT_MAX;
        if (prevKey != curKey)
            start[curKey] = cpos;
        if (curKey != nextKey)
            jumpPos[curKey] = cpos + 1;
    }
}

/**
 * Writes the start array and jump commands for one level.
 *
 * @param[in,out]  start           Start array for previous and current level.
 * @param[out]     commands        Command array in which to write jump commands.
 * @param          jumpPos         Jump positions in command array, as written by @ref writeSplatIds.
 * @param          curOffset       Offset added to code to get position in start array on current level.
 * @param          prevOffset      Offset added to parent code to get position in parent start array.
 *
 * @todo Investigate copying prev to local memory using subset of threads.
 */
__kernel void writeStart(
    __global int *start,
    __global int *commands,
    __global const uint *jumpPos,
    uint curOffset,
    uint prevOffset)
{
    uint code = get_global_id(0);
    uint pos = code + curOffset;
    int jp = jumpPos[pos];
    int prev = start[prevOffset + (code >> 3)];
    if (jp >= 0)
    {
        commands[jp] = (prev == -1) ? -1 : -2 - prev;
    }
    else
    {
        start[pos] = prev;
    }
}

/**
 * Variant of @ref writeStart for the coarsest level. In this level,
 * there is no previous level to chain to.
 *
 * @param[in,out]  start           Start array for previous and current level.
 * @param[out]     commands        Command array in which to write jump commands.
 * @param          jumpPos         Jump positions in command array, as written by @ref writeSplatIds.
 * @param          curOffset       Offset added to code to get position in start array on current level.
 */
__kernel void writeStartTop(
    __global int *start,
    __global int *commands,
    __global const uint *jumpPos,
    uint curOffset)
{
    uint code = get_global_id(0);
    uint pos = code + curOffset;
    int jp = jumpPos[pos];
    if (jp >= 0)
    {
        commands[jp] = -1;
    }
    else
    {
        start[pos] = -1;
    }
}

/**
 * Fill a buffer with a constant value.
 */
__kernel void fill(__global int *out, int value)
{
    out[get_global_id(0)] = value;
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

#endif /* UNIT_TESTS */
