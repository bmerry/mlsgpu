/**
 * @file
 *
 * Implementation of marching tetrahedra.
 */

/// Number of edges in a cell
#define NUM_EDGES 19

/// Number of bits in fixed-point xyz fields in a vertex key (including fractional bits)
#define KEY_AXIS_BITS 21
#define KEY_AXIS_MASK ((1U << KEY_AXIS_BITS) - 1)
#define KEY_EXTERNAL_FLAG (1UL << 63)

__constant sampler_t nearest = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

/**
 * Computes a cell code from 8 isovalues. Non-negative (outside) values are
 * given 1 bits, negative (inside) and NaNs are given 0 bits.
 */
inline uint makeCode(const float iso[8])
{
    return (iso[0] >= 0.0f ? 0x01U : 0U)
        | (iso[1] >= 0.0f ? 0x02U : 0U)
        | (iso[2] >= 0.0f ? 0x04U : 0U)
        | (iso[3] >= 0.0f ? 0x08U : 0U)
        | (iso[4] >= 0.0f ? 0x10U : 0U)
        | (iso[5] >= 0.0f ? 0x20U : 0U)
        | (iso[6] >= 0.0f ? 0x40U : 0U)
        | (iso[7] >= 0.0f ? 0x80U : 0U);
}

inline bool isValid(const float iso[8])
{
    return isfinite(iso[0])
        && isfinite(iso[1])
        && isfinite(iso[2])
        && isfinite(iso[3])
        && isfinite(iso[4])
        && isfinite(iso[5])
        && isfinite(iso[6])
        && isfinite(iso[7]);
}

/**
 * For each cell which might produce triangles, appends the coordinates of the
 * cell to a buffer and the count of vertices and triangles to another.
 *
 * There is one work-item per cell in a slice, arranged in a 2D NDRange.
 *
 * @param[out] occupied      List of cell coordinates for occupied cells
 * @param[out] viCount       Number of triangles+indices per cell.
 * @param[in,out] N          Number of occupied cells, incremented atomically
 * @param      isoImage      Image holding samples.
 * @param      yOffsetA      Initial Y offset in iso for lower z.
 * @param      yOffsetB      Initial Y offset in iso for higher z.
 * @param      countTable    Lookup table of counts per cube code.
 *
 * @todo
 * - Explore Morton order, which will have better texture cache hits.
 * - Consider storing the count table in an image
 */
__kernel void genOccupied(
    __global uint2 * restrict occupied,
    __global uint2 * restrict viCount,
    volatile __global uint * restrict N,
    __read_only image2d_t isoImage,
    uint yOffsetA,
    uint yOffsetB,
    __constant uchar2 * restrict countTable)
{
    uint2 gid = (uint2) (get_global_id(0), get_global_id(1));

    float iso[8];
    iso[0] = read_imagef(isoImage, nearest, convert_int2(gid + (uint2) (0, yOffsetA))).x;
    iso[1] = read_imagef(isoImage, nearest, convert_int2(gid + (uint2) (1, yOffsetA))).x;
    iso[2] = read_imagef(isoImage, nearest, convert_int2(gid + (uint2) (0, yOffsetA + 1))).x;
    iso[3] = read_imagef(isoImage, nearest, convert_int2(gid + (uint2) (1, yOffsetA + 1))).x;
    iso[4] = read_imagef(isoImage, nearest, convert_int2(gid + (uint2) (0, yOffsetB))).x;
    iso[5] = read_imagef(isoImage, nearest, convert_int2(gid + (uint2) (1, yOffsetB))).x;
    iso[6] = read_imagef(isoImage, nearest, convert_int2(gid + (uint2) (0, yOffsetB + 1))).x;
    iso[7] = read_imagef(isoImage, nearest, convert_int2(gid + (uint2) (1, yOffsetB + 1))).x;

    uint code = makeCode(iso);
    bool valid = isValid(iso);

    if (valid && code != 0 && code != 255)
    {
        uint pos = atomic_inc(N);
        occupied[pos] = gid;
        viCount[pos] = convert_uint2(countTable[code]);
    }
}

/**
 * Generate coordinates of a new vertex by interpolation along an edge.
 * @param iso0       Function sample at one corner.
 * @param iso1       Function sample at a second corner.
 * @param cell       Local coordinates of the lowest corner of the cell.
 * @param offset0    Local coordinate offset from @a cell to corner @a iso0.
 * @param offset1    Local coordinate offset from @a cell to corner @a iso1.
 */
inline float3 interp(float iso0, float iso1, uint3 cell, uint3 offset0, uint3 offset1)
{
    // This needs to operate in an invariant manner, so take manual control over FMAs
#pragma OPENCL FP_CONTRACT OFF
    float inv = 1.0f / (iso0 - iso1);
    uint3 delta = offset1 - offset0;
    float3 lcoord = fma(iso0 * inv, convert_float3(delta), convert_float3(cell + offset0));
    return lcoord;
}

#define INTERP(a, b) \
    interp(iso[a], iso[b], globalCell, (uint3) (a & 1, (a >> 1) & 1, (a >> 2) & 1), (uint3) (b & 1, (b >> 1) & 1, (b >> 2) & 1))

/**
 * Computes a key for coordinates. See @ref generateElements for the definition.
 * @param coords           Coordinates in .1 fixed-point format.
 * @param top              Coordinates that indicate an external vertices in .1 fixed-point format.
 */
ulong computeKey(uint3 coords, uint3 top)
{
    ulong key = ((ulong) coords.z << (2 * KEY_AXIS_BITS)) | ((ulong) coords.y << (KEY_AXIS_BITS)) | ((ulong) coords.x);
    if (any(coords.xy == 0U) || any(coords == top))
        key |= KEY_EXTERNAL_FLAG;
    return key;
}

/**
 * Generate vertices and indices for a slice.
 * There is one work-item per compacted cell.
 *
 * Vertices are considered to be external if they lie on the surface of the box
 * bounded by (0, 0, top.z)/2, (top.x, top.y, inf)/2. Note that vertices with
 * maximum z will naturally sort to the end, so they do not get explicitly
 * marked as external (which cannot necessarily be done when there is a split).
 *
 * The @a gridOffset parameter controls how cell coordinates are mapped to
 * global grid space to produce output vertices. For a cell with local
 * coordinates xyz, the corresponding position in world space is
 * xyz + gridOffset.
 *
 * @param[out] vertices        Vertices in world coordinates (unwelded).
 * @param[out] vertexKeys      Vertex keys corresponding to @a vertices.
 * @param[out] indices         Indices into @a vertices.
 * @param      viStart         Position to start writing vertices/indices for each cell.
 * @param      cells           List of compacted cells written by @ref genOccupied.
 * @param      isoImage        Image holding samples.
 * @param      yOffsetA        Initial Y offset in iso for lower z.
 * @param      yOffsetB        Initial Y offset in iso for higher z.
 * @param      startTable      Lookup table indicating where to find vertices/indices in @a dataTable.
 * @param      dataTable       Lookup table of vertex and index indices.
 * @param      keyTable        Lookup table for cell-relative vertex keys.
 * @param      z               Z coordinate of the current slice.
 * @param      gridOffset      Transformation from grid-local to grid-global coordinates.
 * @param      offsets         Offset to add to all elements of @a viStart.
 * @param      top             See above.
 * @param      lvertices       Scratch space of @ref NUM_EDGES elements per work item.
 */
__kernel void generateElements(
    __global float4 *vertices,
    __global ulong *vertexKeys,
    __global uint *indices,
    __global const uint2 * restrict viStart,
    __global const uint2 * restrict cells,
    __read_only image2d_t isoImage,
    uint yOffsetA,
    uint yOffsetB,
    __global const ushort2 * restrict startTable,
    __global const uchar * restrict dataTable,
    __global const uint3 * restrict keyTable,
    uint z,
    uint3 gridOffset,
    uint2 offsets,
    uint3 top,
    __local float3 *lvertices)
{
    const uint gid = get_global_id(0);
    const uint lid = get_local_id(0);
    uint3 cell;
    cell.xy = cells[gid];
    cell.z = z;
    const uint3 globalCell = cell + gridOffset;
    __local float3 *lverts = lvertices + NUM_EDGES * lid;

    float iso[8];
    iso[0] = read_imagef(isoImage, nearest, convert_int2(cell.xy + (uint2) (0, yOffsetA))).x;
    iso[1] = read_imagef(isoImage, nearest, convert_int2(cell.xy + (uint2) (1, yOffsetA))).x;
    iso[2] = read_imagef(isoImage, nearest, convert_int2(cell.xy + (uint2) (0, yOffsetA + 1))).x;
    iso[3] = read_imagef(isoImage, nearest, convert_int2(cell.xy + (uint2) (1, yOffsetA + 1))).x;
    iso[4] = read_imagef(isoImage, nearest, convert_int2(cell.xy + (uint2) (0, yOffsetB))).x;
    iso[5] = read_imagef(isoImage, nearest, convert_int2(cell.xy + (uint2) (1, yOffsetB))).x;
    iso[6] = read_imagef(isoImage, nearest, convert_int2(cell.xy + (uint2) (0, yOffsetB + 1))).x;
    iso[7] = read_imagef(isoImage, nearest, convert_int2(cell.xy + (uint2) (1, yOffsetB + 1))).x;

    lverts[0] = INTERP(0, 1);
    lverts[1] = INTERP(0, 2);
    lverts[2] = INTERP(0, 3);
    lverts[3] = INTERP(1, 3);
    lverts[4] = INTERP(2, 3);
    lverts[5] = INTERP(0, 4);
    lverts[6] = INTERP(0, 5);
    lverts[7] = INTERP(1, 5);
    lverts[8] = INTERP(4, 5);
    lverts[9] = INTERP(0, 6);
    lverts[10] = INTERP(2, 6);
    lverts[11] = INTERP(4, 6);
    lverts[12] = INTERP(0, 7);
    lverts[13] = INTERP(1, 7);
    lverts[14] = INTERP(2, 7);
    lverts[15] = INTERP(3, 7);
    lverts[16] = INTERP(4, 7);
    lverts[17] = INTERP(5, 7);
    lverts[18] = INTERP(6, 7);

    uint code = makeCode(iso);
    uint2 viNext = viStart[gid] + offsets;
    uint vNext = viNext.s0;
    uint iNext = viNext.s1;

    ushort2 start = startTable[code];
    ushort2 end = startTable[code + 1];

    for (uint i = 0; i < end.x - start.x; i++)
    {
        float4 vertex;
        vertex.xyz = lverts[dataTable[start.x + i]];
        vertex.w = as_float(vNext + i);
        vertices[vNext + i] = vertex;
        vertexKeys[vNext + i] = computeKey(2 * cell + keyTable[start.x + i], top);
    }
    for (uint i = 0; i < end.y - start.y; i++)
    {
        indices[iNext + i] = vNext + dataTable[start.y + i];
    }
}

/**
 * Determines which vertex keys are unique. For a range of equal keys, the @em last
 * one is given an indicator of 1, while the others get an indicator of 0.
 *
 * @param[out] vertexUnique         1 for exactly one instance of each key, 0 elsewhere.
 * @param      vertexKeys           Vertex keys.
 *
 * @pre @a vertexKeys must be sorted such that equal keys are adjacent.
 *
 * @todo Investigate using @c __local to avoid two key reads (might not matter with a cache).
 */
__kernel void countUniqueVertices(__global uint * restrict vertexUnique,
                                  __global const ulong * restrict vertexKeys)
{
    const uint gid = get_global_id(0);
    const ulong key = vertexKeys[gid];
    const ulong nextKey = vertexKeys[gid + 1];
    bool last = key != nextKey;
    vertexUnique[gid] = last ? 1 : 0;
}

/**
 * Copy the unique vertices to a new array and generate a remapping table.
 * There is one work-item per input vertex.
 *
 * @param[out] outVertices     Output vertices, written as packed x,y,z triplets.
 * @param[out] outKeys         Vertex keys corresponding to @a outVertices, only written for external vertices, and with the high bit stripped off.
 * @param[out] indexRemap      Table mapping original (pre-sorting) indices to output indices.
 * @param[out] firstExternal   The first output position that contains an external vertex.
 * @param      vertexUnique    Scan of the table emitted by @ref countUniqueVertices.
 * @param      inVertices      Sorted vertices, with original ID stored in @c w.
 * @param      inKeys          Vertex keys corresponding to @a inVertices (plus a sentinel @c ULONG_MAX).
 * @param      minExternalKey  Vertex keys >= @a minExternalKey are considered to be external vertices.
 * @param      keyOffset       Value added to keys on output (after comparison with @a minExternalKey).
 */
__kernel void compactVertices(
    __global float * restrict outVertices,
    __global ulong * restrict outKeys,
    __global uint * restrict indexRemap,
    __global uint * firstExternal,
    __global const uint * restrict vertexUnique,
    __global const float4 * restrict inVertices,
    __global const ulong * restrict inKeys,
    ulong minExternalKey,
    ulong keyOffset)
{
    const uint gid = get_global_id(0);
    const uint u = vertexUnique[gid];
    const float4 v = inVertices[gid];
    const ulong key = inKeys[gid];
    const ulong nextKey = inKeys[gid + 1];
    bool ext = key >= minExternalKey;
    if (key != nextKey)
    {
        vstore3(v.xyz, u, outVertices);
        if (ext)
        {
            outKeys[u] = (key & (KEY_EXTERNAL_FLAG - 1)) + keyOffset;
            if (u == 0)
                *firstExternal = 0;
        }
        else if (nextKey >= minExternalKey)
            *firstExternal = u + 1;
    }
    uint originalIndex = as_uint(v.w);
    indexRemap[originalIndex] = u;
}

/**
 * Apply an index remapping table to the indices. There is one work-item
 * per index.
 * @param[in,out]    indices      Indices to rewrite.
 * @param            indexRemap   Remapping table to apply.
 */
__kernel void reindex(
    __global uint *indices,
    __global const uint * restrict indexRemap)
{
    const uint gid = get_global_id(0);
    indices[gid] = indexRemap[indices[gid]];
}

/*******************************************************************************
 * Test code only below here.
 *******************************************************************************/

#if UNIT_TESTS

__kernel void testComputeKey(__global ulong *out, uint3 coords, uint3 top)
{
    *out = computeKey(coords, top);
}

#endif /* UNIT_TESTS */
