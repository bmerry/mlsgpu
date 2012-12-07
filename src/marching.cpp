/**
 * @file
 *
 * Marching tetrahedra algorithms.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif

#ifndef __CL_ENABLE_EXCEPTIONS
# define __CL_ENABLE_EXCEPTIONS
#endif

#include <vector>
#include <utility>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include "tr1_cstdint.h"
#include "clh.h"
#include "marching.h"
#include "grid.h"
#include "errors.h"
#include "statistics.h"
#include "statistics_cl.h"
#include "misc.h"

const unsigned char Marching::edgeIndices[NUM_EDGES][2] =
{
    {0, 1},
    {0, 2},
    {0, 3},
    {1, 3},
    {2, 3},
    {0, 4},
    {0, 5},
    {1, 5},
    {4, 5},
    {0, 6},
    {2, 6},
    {4, 6},
    {0, 7},
    {1, 7},
    {2, 7},
    {3, 7},
    {4, 7},
    {5, 7},
    {6, 7}
};

const unsigned char Marching::tetrahedronIndices[NUM_TETRAHEDRA][4] =
{
    { 0, 7, 1, 3 },
    { 0, 7, 3, 2 },
    { 0, 7, 2, 6 },
    { 0, 7, 6, 4 },
    { 0, 7, 4, 5 },
    { 0, 7, 5, 1 }
};

unsigned int Marching::findEdgeByVertexIds(unsigned int v0, unsigned int v1)
{
    if (v0 > v1) std::swap(v0, v1);
    for (unsigned int i = 0; i < NUM_EDGES; i++)
        if (edgeIndices[i][0] == v0 && edgeIndices[i][1] == v1)
            return i;
    assert(false);
    return -1;
}

template<typename Iterator>
unsigned int Marching::permutationParity(Iterator first, Iterator last)
{
    unsigned int parity = 0;
    // Simple implementation, since we will only ever call it with 4
    // items.
    for (Iterator i = first; i != last; ++i)
    {
        Iterator j = i;
        for (++j; j != last; ++j)
            if (*i > *j)
                parity ^= 1;
    }
    return parity;
}

void Marching::makeTables()
{
    std::vector<cl_uchar> hVertexTable, hIndexTable;
    std::vector<cl_uint3> hKeyTable;
    std::vector<cl_uchar2> hCountTable(NUM_CUBES);
    std::vector<cl_ushort2> hStartTable(NUM_CUBES + 1);
    for (unsigned int i = 0; i < NUM_CUBES; i++)
    {
        hStartTable[i].s[0] = hVertexTable.size();
        hStartTable[i].s[1] = hIndexTable.size();

        /* Find triangles. For now we record triangle indices
         * as edge numbers, which we will compact later.
         */
        std::vector<cl_uchar> triangles;
        for (unsigned int j = 0; j < NUM_TETRAHEDRA; j++)
        {
            // Holds a vertex index together with the inside/outside flag
            typedef std::pair<unsigned char, bool> tvtx;
            tvtx tvtxs[4];
            unsigned int outside = 0;
            // Copy the vertices to tvtxs, and count vertices that are external
            for (unsigned int k = 0; k < 4; k++)
            {
                unsigned int v = tetrahedronIndices[j][k];
                bool o = (i & (1 << v));
                outside += o;
                tvtxs[k] = tvtx(v, o);
            }
            unsigned int baseParity = permutationParity(tvtxs, tvtxs + 4);

            // Reduce number of cases to handle by flipping inside/outside to
            // ensure that outside <= 2.
            if (outside > 2)
            {
                // Causes triangle winding to flip as well - otherwise
                // the triangle will be inside out.
                baseParity ^= 1;
                for (unsigned int k = 0; k < 4; k++)
                    tvtxs[k].second = !tvtxs[k].second;
            }

            /* To reduce the number of cases to handle, the tetrahedron is
             * rotated to match one of the canonical configurations (all
             * inside, v0 outside, (v0, v1) outside). There are 24 permutations
             * of the vertices, half of which are rotations and half of which are
             * reflections. Not all of them need to be tried, but this code isn't
             * performance critical.
             */
            sort(tvtxs, tvtxs + 4);
            do
            {
                // Check that it is a rotation rather than a reflection
                if (permutationParity(tvtxs, tvtxs + 4) == baseParity)
                {
                    const unsigned int t0 = tvtxs[0].first;
                    const unsigned int t1 = tvtxs[1].first;
                    const unsigned int t2 = tvtxs[2].first;
                    const unsigned int t3 = tvtxs[3].first;
                    unsigned int mask = 0;
                    for (unsigned int k = 0; k < 4; k++)
                        mask |= tvtxs[k].second << k;
                    if (mask == 0)
                    {
                        break; // no outside vertices, so no triangles needed
                    }
                    else if (mask == 1)
                    {
                        // One outside vertex, one triangle needed
                        triangles.push_back(findEdgeByVertexIds(t0, t1));
                        triangles.push_back(findEdgeByVertexIds(t0, t3));
                        triangles.push_back(findEdgeByVertexIds(t0, t2));
                        break;
                    }
                    else if (mask == 3)
                    {
                        // Two outside vertices, two triangles needed to tile a quad
                        triangles.push_back(findEdgeByVertexIds(t0, t2));
                        triangles.push_back(findEdgeByVertexIds(t1, t2));
                        triangles.push_back(findEdgeByVertexIds(t1, t3));

                        triangles.push_back(findEdgeByVertexIds(t1, t3));
                        triangles.push_back(findEdgeByVertexIds(t0, t3));
                        triangles.push_back(findEdgeByVertexIds(t0, t2));
                        break;
                    }
                }
            } while (next_permutation(tvtxs, tvtxs + 4));
        }

        // Determine which edges are in use, and assign indices
        int edgeCompact[NUM_EDGES];
        int pool = 0;
        for (unsigned int j = 0; j < NUM_EDGES; j++)
        {
            if (std::count(triangles.begin(), triangles.end(), j))
            {
                edgeCompact[j] = pool++;
                hVertexTable.push_back(j);
                cl_uint3 key = { {0, 0, 0} };
                for (unsigned int axis = 0; axis < 3; axis++)
                {
                    key.s[axis] =
                        ((edgeIndices[j][0] >> axis) & 1)
                        + ((edgeIndices[j][1] >> axis) & 1);
                }
                hKeyTable.push_back(key);
            }
        }
        for (unsigned int j = 0; j < triangles.size(); j++)
        {
            hIndexTable.push_back(edgeCompact[triangles[j]]);
        }

        hCountTable[i].s[0] = hVertexTable.size() - hStartTable[i].s[0];
        hCountTable[i].s[1] = hIndexTable.size() - hStartTable[i].s[1];
    }

    hStartTable[NUM_CUBES].s[0] = hVertexTable.size();
    hStartTable[NUM_CUBES].s[1] = hIndexTable.size();

    /* We're going to concatenate hVertexTable and hIndexTable, so the start values
     * need to be offset to point to where hIndexTable sits afterwards.
     */
    for (unsigned int i = 0; i <= NUM_CUBES; i++)
    {
        hStartTable[i].s[1] += hVertexTable.size();
    }
    // Concatenate the two tables into one
    hVertexTable.insert(hVertexTable.end(), hIndexTable.begin(), hIndexTable.end());

    countTable = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            hCountTable.size() * sizeof(hCountTable[0]), &hCountTable[0]);
    startTable = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            hStartTable.size() * sizeof(hStartTable[0]), &hStartTable[0]);
    dataTable =  cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            hVertexTable.size() * sizeof(hVertexTable[0]), &hVertexTable[0]);
    keyTable =   cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            hKeyTable.size() * sizeof(hKeyTable[0]), &hKeyTable[0]);
    assert(countTable.getInfo<CL_MEM_SIZE>() == COUNT_TABLE_BYTES);
    assert(startTable.getInfo<CL_MEM_SIZE>() == START_TABLE_BYTES);
    assert(dataTable.getInfo<CL_MEM_SIZE>() == DATA_TABLE_BYTES);
    assert(keyTable.getInfo<CL_MEM_SIZE>() == KEY_TABLE_BYTES);
}

void Marching::validateDevice(const cl::Device &device)
{
    if (!device.getInfo<CL_DEVICE_IMAGE_SUPPORT>())
        throw CLH::invalid_device(device, "images are not supported");
}

CLH::ResourceUsage Marching::resourceUsage(
    const cl::Device &device,
    Grid::size_type maxWidth, Grid::size_type maxHeight, Grid::size_type maxDepth,
    Grid::size_type maxSwathe,
    std::size_t meshMemory,
    const Grid::size_type alignment[3])
{
    MLSGPU_ASSERT(2 <= maxWidth && maxWidth <= MAX_DIMENSION, std::invalid_argument);
    MLSGPU_ASSERT(2 <= maxHeight && maxHeight <= MAX_DIMENSION, std::invalid_argument);
    MLSGPU_ASSERT(2 <= maxDepth && maxDepth <= MAX_DIMENSION, std::invalid_argument);
    MLSGPU_ASSERT(alignment[2] <= maxSwathe, std::invalid_argument);
    MLSGPU_ASSERT(meshMemory >= (maxWidth - 1) * (maxHeight - 1) * MAX_CELL_BYTES, std::invalid_argument);
    (void) device; // not currently used, but should be used to determine usage of clogs

    Grid::size_type imageWidth = roundUp(maxWidth, alignment[0]);
    Grid::size_type imageHeight = roundUp(maxHeight, alignment[1]);
    maxSwathe = std::min(maxSwathe, maxDepth) / alignment[2] * alignment[2];

    // The asserts above guarantee that these will not overflow
    const std::tr1::uint64_t sliceCells = (maxWidth - 1) * (maxHeight - 1);
    const std::tr1::uint64_t swatheCells = sliceCells * maxSwathe;
    const std::tr1::uint64_t meshCells = meshMemory / MAX_CELL_BYTES;
    const std::tr1::uint64_t vertexSpace = meshCells * MAX_CELL_VERTICES;
    const std::tr1::uint64_t indexSpace = meshCells * MAX_CELL_INDICES;

    CLH::ResourceUsage ans;
    // Keep this in sync with the actual allocations below

    // image = cl::Image2D(context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), imageWidth, imageHeight * (maxSwathe + 1));
    ans.addImage(imageWidth, imageHeight * (maxSwathe + 1), sizeof(cl_float));

    // cells = cl::Buffer(context, CL_MEM_READ_WRITE, swatheCells * sizeof(cl_uint3));
    ans.addBuffer(swatheCells * sizeof(cl_uint3));

    // numOccupied = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(cl_uint));
    ans.addBuffer(sizeof(cl_uint));

    // viHistogram = cl::Buffer(context, CL_MEM_READ_WRITE, maxDepth * sizeof(cl_uint2));
    ans.addBuffer(maxDepth * sizeof(cl_uint2));

    // viCount = cl::Buffer(context, CL_MEM_READ_WRITE, swatheCells * sizeof(cl_uint2));
    ans.addBuffer(swatheCells * sizeof(cl_uint2));

    // vertexUnique = cl::Buffer(context, CL_MEM_READ_WRITE, (vertexSpace + 1) * sizeof(cl_uint));
    ans.addBuffer((vertexSpace + 1) * sizeof(cl_uint));

    // indexRemap = cl::Buffer(context, CL_MEM_READ_WRITE, vertexSpace * sizeof(cl_uint));
    ans.addBuffer(vertexSpace * sizeof(cl_uint));

    // unweldedVertices = cl::Buffer(context, CL_MEM_READ_WRITE, vertexSpace * sizeof(cl_float4));
    // unweldedVertexKeys = cl::Buffer(context, CL_MEM_READ_WRITE, (vertexSpace + 1) * sizeof(cl_ulong));
    ans.addBuffer(vertexSpace * sizeof(cl_float4));
    ans.addBuffer((vertexSpace + 1) * sizeof(cl_ulong));

    // weldedVertices = cl::Buffer(context, CL_MEM_WRITE_ONLY, vertexSpace * sizeof(cl_float4));
    // weldedVertexKeys = cl::Buffer(context, CL_MEM_WRITE_ONLY, vertexSpace * sizeof(cl_ulong));
    ans.addBuffer(vertexSpace * sizeof(cl_float4));
    ans.addBuffer(vertexSpace * sizeof(cl_ulong));

    // indices = cl::Buffer(context, CL_MEM_READ_WRITE, indexSpace * sizeof(cl_uint));
    ans.addBuffer(indexSpace * sizeof(cl_uint));

    // firstExternal = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(cl_uint));
    ans.addBuffer(sizeof(cl_uint));

    // Lookup tables
    ans.addBuffer(COUNT_TABLE_BYTES);
    ans.addBuffer(START_TABLE_BYTES);
    ans.addBuffer(DATA_TABLE_BYTES);
    ans.addBuffer(KEY_TABLE_BYTES);
    // TODO: temporaries for the sorter and scanners

    return ans;
}

Marching::Marching(const cl::Context &context, const cl::Device &device,
                   Grid::size_type maxWidth, Grid::size_type maxHeight, Grid::size_type maxDepth,
                   Grid::size_type maxSwathe,
                   std::size_t meshMemory,
                   const Grid::size_type alignment[3])
:
    maxWidth(maxWidth), maxHeight(maxHeight), maxDepth(maxDepth),
    context(context),
    genOccupiedKernelTime(Statistics::getStatistic<Statistics::Variable>("kernel.marching.genOccupied.time")),
    generateElementsKernelTime(Statistics::getStatistic<Statistics::Variable>("kernel.marching.generateElements.time")),
    countUniqueVerticesKernelTime(Statistics::getStatistic<Statistics::Variable>("kernel.marching.countUniqueVertices.time")),
    compactVerticesKernelTime(Statistics::getStatistic<Statistics::Variable>("kernel.marching.compactVertices.time")),
    reindexKernelTime(Statistics::getStatistic<Statistics::Variable>("kernel.marching.reindex.time")),
    copySliceTime(Statistics::getStatistic<Statistics::Variable>("kernel.marching.copySlice.time")),
    scanUint(context, device, clogs::TYPE_UINT),
    scanElements(context, device, clogs::Type(clogs::TYPE_UINT, 2)),
    sortVertices(context, device, clogs::TYPE_ULONG, clogs::Type(clogs::TYPE_FLOAT, 4)),
    readback(context, device),
    viReadback(context, device, maxSwathe)
{
    MLSGPU_ASSERT(2 <= maxWidth && maxWidth <= MAX_DIMENSION, std::invalid_argument);
    MLSGPU_ASSERT(2 <= maxHeight && maxHeight <= MAX_DIMENSION, std::invalid_argument);
    MLSGPU_ASSERT(2 <= maxDepth && maxDepth <= MAX_DIMENSION, std::invalid_argument);
    MLSGPU_ASSERT(alignment[2] <= maxSwathe, std::invalid_argument);
    MLSGPU_ASSERT(meshMemory >= (maxWidth - 1) * (maxHeight - 1) * MAX_CELL_BYTES, std::invalid_argument);

    Grid::size_type imageWidth = roundUp(maxWidth, alignment[0]);
    Grid::size_type imageHeight = roundUp(maxHeight, alignment[1]);
    this->maxSwathe = std::min(maxSwathe, maxDepth) / alignment[2] * alignment[2];

    scanUint.setEventCallback(
        &Statistics::timeEventCallback,
        &Statistics::getStatistic<Statistics::Variable>("kernel.marching.scanUint.time"));
    scanElements.setEventCallback(
        &Statistics::timeEventCallback,
        &Statistics::getStatistic<Statistics::Variable>("kernel.marching.scanElements.time"));
    sortVertices.setEventCallback(
        &Statistics::timeEventCallback,
        &Statistics::getStatistic<Statistics::Variable>("kernel.marching.sortVertices.time"));

    makeTables();
    image = cl::Image2D(context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT),
                        imageWidth, imageHeight * (maxSwathe + 1));
    zStride = imageHeight;

    const std::size_t sliceCells = (maxWidth - 1) * (maxHeight - 1);
    const std::size_t swatheCells = sliceCells * maxSwathe;
    const std::size_t meshCells = meshMemory / MAX_CELL_BYTES;
    vertexSpace = meshCells * MAX_CELL_VERTICES;
    indexSpace = meshCells * MAX_CELL_INDICES;

    // If these are updated, also update deviceMemory
    cells = cl::Buffer(context, CL_MEM_READ_WRITE, swatheCells * sizeof(cl_uint3));
    numOccupied = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(cl_uint));
    viHistogram = cl::Buffer(context, CL_MEM_READ_WRITE, maxDepth * sizeof(cl_uint2));
    viCount = cl::Buffer(context, CL_MEM_READ_WRITE, swatheCells * sizeof(cl_uint2));
    vertexUnique = cl::Buffer(context, CL_MEM_READ_WRITE, (vertexSpace + 1) * sizeof(cl_uint));
    indexRemap = cl::Buffer(context, CL_MEM_READ_WRITE, vertexSpace * sizeof(cl_uint));
    unweldedVertices = cl::Buffer(context, CL_MEM_READ_WRITE, vertexSpace * sizeof(cl_float4));
    unweldedVertexKeys = cl::Buffer(context, CL_MEM_READ_WRITE, (vertexSpace + 1) * sizeof(cl_ulong));
    // weldedVertices holds packed float3s, but because it's also used as the
    // temporary buffer for sorting it needs to be able to hold float4s.
    weldedVertices = cl::Buffer(context, CL_MEM_WRITE_ONLY, vertexSpace * sizeof(cl_float4));
    weldedVertexKeys = cl::Buffer(context, CL_MEM_WRITE_ONLY, vertexSpace * sizeof(cl_ulong));
    indices = cl::Buffer(context, CL_MEM_READ_WRITE, indexSpace * sizeof(cl_uint));
    firstExternal = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(cl_uint));
    sortVertices.setTemporaryBuffers(weldedVertices, weldedVertexKeys);

    cl::Program program = CLH::build(context, std::vector<cl::Device>(1, device), "kernels/marching.cl");
    genOccupiedKernel = cl::Kernel(program, "genOccupied");
    generateElementsKernel = cl::Kernel(program, "generateElements");
    countUniqueVerticesKernel = cl::Kernel(program, "countUniqueVertices");
    compactVerticesKernel = cl::Kernel(program, "compactVertices");
    reindexKernel = cl::Kernel(program, "reindex");
    copySliceKernel = cl::Kernel(program, "copySlice");

    // Set up kernel arguments that never change.
    genOccupiedKernel.setArg(0, cells);
    genOccupiedKernel.setArg(1, viCount);
    genOccupiedKernel.setArg(2, numOccupied);
    genOccupiedKernel.setArg(3, viHistogram);
    genOccupiedKernel.setArg(7, countTable);

    generateElementsKernel.setArg(0, unweldedVertices);
    generateElementsKernel.setArg(1, unweldedVertexKeys);
    generateElementsKernel.setArg(2, indices);
    generateElementsKernel.setArg(3, viCount);
    generateElementsKernel.setArg(4, cells);
    generateElementsKernel.setArg(5, image);
    generateElementsKernel.setArg(6, startTable);
    generateElementsKernel.setArg(7, dataTable);
    generateElementsKernel.setArg(8, keyTable);

    countUniqueVerticesKernel.setArg(0, vertexUnique);
    countUniqueVerticesKernel.setArg(1, unweldedVertexKeys);

    compactVerticesKernel.setArg(0, weldedVertices);
    compactVerticesKernel.setArg(1, weldedVertexKeys);
    compactVerticesKernel.setArg(2, indexRemap);
    compactVerticesKernel.setArg(3, firstExternal);
    compactVerticesKernel.setArg(4, vertexUnique);
    compactVerticesKernel.setArg(5, unweldedVertices);
    compactVerticesKernel.setArg(6, unweldedVertexKeys);

    reindexKernel.setArg(0, indices);
    reindexKernel.setArg(1, indexRemap);
}

void Marching::copySlice(
    const cl::CommandQueue &queue,
    const cl::Image2D &image,
    Grid::size_type src,
    Grid::size_type trg,
    const ImageParams &params,
    const std::vector<cl::Event> *events,
    cl::Event *event)
{
    cl::size_t<3> srcOrigin, trgOrigin, region;
    srcOrigin[0] = 0;
    srcOrigin[1] = src * params.zStride;
    srcOrigin[2] = 0;
    trgOrigin[0] = 0;
    trgOrigin[1] = trg * params.zStride;
    trgOrigin[2] = 0;
    region[0] = params.width;
    region[1] = params.height;
    region[2] = 1;

    try
    {
        cl::Event last;
        queue.enqueueCopyImage(image, image, srcOrigin, trgOrigin, region, events, &last);
        Statistics::timeEvent(last, copySliceTime);
        if (event != NULL)
            *event = last;
    }
    catch (cl::Error &e)
    {
        if (e.err() != CL_MEM_COPY_OVERLAP)
            throw;
        /* Workaround for AMD APP SDK v2.8 (and earlier), which does not
         * allow copies from an image to the same image.
         */
        cl_int2 offset;
        offset.s[0] = (cl_int) trgOrigin[0] - (cl_int) srcOrigin[0];
        offset.s[1] = (cl_int) trgOrigin[1] - (cl_int) srcOrigin[1];

        copySliceKernel.setArg(0, image);
        copySliceKernel.setArg(1, image);
        copySliceKernel.setArg(2, offset);
        CLH::enqueueNDRangeKernelSplit(
            queue,
            copySliceKernel,
            cl::NDRange(srcOrigin[0], srcOrigin[1]),
            cl::NDRange(region[0], region[1]),
            cl::NullRange,
            events, event,
            &copySliceTime);
    }
}

std::size_t Marching::generateCells(
    const cl::CommandQueue &queue,
    const Swathe &swathe,
    const std::vector<cl::Event> *events)
{
    const std::size_t viOffset = swathe.zFirst * sizeof(cl_uint2);
    const std::size_t viSize = (swathe.zLast - swathe.zFirst) * sizeof(cl_uint2);
    cl::Event last, last2;

    readback->compacted = 0;
    queue.enqueueWriteBuffer(numOccupied, CL_FALSE, 0, sizeof(cl_uint),
                             &readback->compacted, events, &last);
    std::memset(viReadback.get() + swathe.zFirst, 0, viSize);
    queue.enqueueWriteBuffer(viHistogram, CL_FALSE,
                             viOffset, viSize,
                             viReadback.get() + swathe.zFirst, events, &last2);
    // TODO: account for time

    std::vector<cl::Event> wait(2);
    wait[0] = last;
    wait[1] = last2;

    genOccupiedKernel.setArg(4, image);
    genOccupiedKernel.setArg(5, swathe.zStride);
    genOccupiedKernel.setArg(6, swathe.zBias);
    // TODO: round image size up to multiple of local work group size,
    // to avoid extra splits; will only work if combined with NaN padding
    // though, and also requires the generator to respect the padding.
    CLH::enqueueNDRangeKernelSplit(
        queue,
        genOccupiedKernel,
        cl::NDRange(0, 0, swathe.zFirst),
        cl::NDRange(swathe.width - 1, swathe.height - 1, swathe.zLast - swathe.zFirst),
        cl::NDRange(16, 16, 1),
        &wait, &last, &genOccupiedKernelTime);

    wait.resize(1);
    wait[0] = last;

    queue.enqueueReadBuffer(viHistogram, CL_FALSE, viOffset, viSize,
                            viReadback.get() + swathe.zFirst, &wait, NULL);
    queue.enqueueReadBuffer(numOccupied, CL_FALSE, 0, sizeof(cl_uint),
                            &readback->compacted,
                            &wait, NULL);
    // TODO: account for time for the above
    queue.finish();

    return readback->compacted;
}

void Marching::shipOut(const cl::CommandQueue &queue,
                       const cl_uint3 &keyOffset,
                       const cl_uint2 &sizes,
                       cl_uint zMax,
                       const OutputFunctor &output,
                       const std::vector<cl::Event> *events,
                       cl::Event *event)
{
    std::vector<cl::Event> wait(1);
    cl::Event last;

    // Write a sentinel key after the real vertex keys
    cl_ulong key = CL_ULONG_MAX;
    queue.enqueueWriteBuffer(unweldedVertexKeys, CL_FALSE, sizes.s[0] * sizeof(cl_ulong), sizeof(cl_ulong), &key,
                             events, &last);
    wait[0] = last;

    // TODO: figure out how many actual bits there are
    // TODO: revisit the dependency tracking
    sortVertices.enqueue(queue, unweldedVertexKeys, unweldedVertices, sizes.s[0], 0, &wait, &last);
    wait[0] = last;

    CLH::enqueueNDRangeKernel(queue,
                              countUniqueVerticesKernel,
                              cl::NullRange,
                              cl::NDRange(sizes.s[0]),
                              cl::NullRange,
                              &wait, &last, &countUniqueVerticesKernelTime);
    wait[0] = last;

    scanUint.enqueue(queue, vertexUnique, sizes.s[0] + 1, NULL, &wait, &last);
    wait[0] = last;

    // Start this readback - but we don't immediately need the result.
    queue.enqueueReadBuffer(vertexUnique, CL_FALSE, sizes.s[0] * sizeof(cl_uint), sizeof(cl_uint),
                            &readback->numWelded, &wait, NULL);

    // TODO: should we be sorting key/value pairs? The values are going to end up moving
    // twice, and most of them will be eliminated entirely! However, sorting them does
    // give later passes better spatial locality and fewer indirections.
    cl_ulong minExternalKey = cl_ulong(zMax) << (2 * KEY_AXIS_BITS + 1);
    cl_ulong keyOffsetL =
        (cl_ulong(keyOffset.s[2]) << (2 * KEY_AXIS_BITS + 1))
        | (cl_ulong(keyOffset.s[1]) << (KEY_AXIS_BITS + 1))
        | (cl_ulong(keyOffset.s[0]) << 1);
    compactVerticesKernel.setArg(7, minExternalKey);
    compactVerticesKernel.setArg(8, keyOffsetL);
    CLH::enqueueNDRangeKernel(queue,
                              compactVerticesKernel,
                              cl::NullRange,
                              cl::NDRange(sizes.s[0]),
                              cl::NullRange,
                              &wait, &last, &compactVerticesKernelTime);
    wait[0] = last;

    queue.enqueueReadBuffer(firstExternal, CL_FALSE, 0, sizeof(cl_uint),
                            &readback->firstExternal, &wait, NULL);

    CLH::enqueueNDRangeKernel(queue,
                              reindexKernel,
                              cl::NullRange,
                              cl::NDRange(sizes.s[1]),
                              cl::NullRange,
                              &wait, &last, &reindexKernelTime);
    queue.finish(); // wait for readback of numWelded and firstExternal (TODO: overkill)

    DeviceKeyMesh outputMesh; // TODO: store buffers in this instead of copying references
    outputMesh.vertices = weldedVertices;
    outputMesh.vertexKeys = weldedVertexKeys;
    outputMesh.triangles = indices;
    outputMesh.numVertices = readback->numWelded;
    outputMesh.numInternalVertices = readback->firstExternal;
    outputMesh.numTriangles = sizes.s[1] / 3;
    output(queue, outputMesh, NULL, event);
}

Grid::size_type Marching::addSlices(
    const cl::CommandQueue &queue,
    const OutputFunctor &output,
    const Swathe &swathe,
    const cl_uint3 &keyOffset,
    const cl::NDRange &localSize,
    cl_uint2 &offsets, cl_uint3 &top,
    const std::vector<cl::Event> *events,
    cl::Event *event)
{
    Statistics::Variable &nonempty = Statistics::getStatistic<Statistics::Variable>("marching.slice.nonempty");

    std::vector<cl::Event> wait;
    cl::Event last;
    Grid::size_type shipOuts = 0;

    std::size_t compacted = generateCells(queue, swathe, events);

    if (compacted > 0)
    {
        // Reduce the per-slice histogam
        cl_uint2 counts = {{ 0, 0 }};
        for (Grid::size_type i = swathe.zFirst; i < swathe.zLast; i++)
            for (int j = 0; j < 2; j++)
                counts.s[j] += viReadback[i].s[j];

        if (counts.s[0] > vertexSpace
            || counts.s[1] > indexSpace)
        {
            // Swathe is too big on its own. Subdivide it and start
            // again
            Grid::size_type subFirst = swathe.zFirst;
            while (subFirst < swathe.zLast)
            {
                Grid::size_type subLast = swathe.zLast;
                counts.s[0] = 0;
                counts.s[1] = 0;
                while (subLast < swathe.zLast
                       && offsets.s[0] + counts.s[0] + viReadback[subLast].s[0] <= vertexSpace
                       && offsets.s[1] + counts.s[1] + viReadback[subLast].s[1] <= indexSpace)
                {
                    counts.s[0] += viReadback[subLast].s[0];
                    counts.s[1] += viReadback[subLast].s[1];
                    subLast++;
                }
                if (subFirst == subLast)
                {
                    // Can't combine with previous swathe in a shipOut, so just ignore
                    // previous data when picking a size. The recursive addSlice call
                    // will call shipOut.
                    while (subLast < swathe.zLast
                           && counts.s[0] + viReadback[subLast].s[0] <= vertexSpace
                           && counts.s[1] + viReadback[subLast].s[1] <= indexSpace)
                    {
                        counts.s[0] += viReadback[subLast].s[0];
                        counts.s[1] += viReadback[subLast].s[1];
                        subLast++;
                    }
                }
                // This should pass because we impose a lower bound of one slice on
                // meshMemory.
                assert(subLast > subFirst);
                Swathe subSwathe = swathe;
                subSwathe.zFirst = subFirst;
                subSwathe.zLast = subLast;
                shipOuts += addSlices(
                    queue, output,
                    subSwathe, keyOffset, localSize,
                    offsets, top,
                    &wait, &last);
                wait.resize(1);
                wait[0] = last;

                subFirst = subLast;
            }
        }
        else
        {
            if (offsets.s[0] + counts.s[0] > vertexSpace
                || offsets.s[1] + counts.s[1] > indexSpace)
            {
                /* The swathe fits, but only after we flush previous data.
                 */
                shipOut(queue, keyOffset, offsets, swathe.zFirst, output, &wait, &last);
                shipOuts++;
                wait.resize(1);
                wait[0] = last;

                offsets.s[0] = 0;
                offsets.s[1] = 0;
                top.s[2] = 2 * swathe.zFirst;
            }

            scanElements.enqueue(queue, viCount, compacted, &offsets, &wait, &last);
            wait.resize(1);
            wait[0] = last;

            generateElementsKernel.setArg(9, swathe.zStride);
            generateElementsKernel.setArg(10, swathe.zBias);
            generateElementsKernel.setArg(12, top);
            CLH::enqueueNDRangeKernelSplit(queue,
                                           generateElementsKernel,
                                           cl::NullRange,
                                           cl::NDRange(compacted),
                                           localSize,
                                           &wait, &last, &generateElementsKernelTime);
            wait.resize(1);
            wait[0] = last;

            offsets.s[0] += counts.s[0];
            offsets.s[1] += counts.s[1];
        }
    }
    nonempty.add(compacted > 0);
    if (event != NULL)
        CLH::enqueueMarkerWithWaitList(queue, &wait, event);
    return shipOuts;
}

void Marching::generate(
    const cl::CommandQueue &queue,
    Generator &generator,
    const OutputFunctor &output,
    const Grid::size_type size[3],
    const cl_uint3 &keyOffset,
    const std::vector<cl::Event> *events)
{
    std::size_t localSize = queue.getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
    // Work group size for kernels that operate on compacted cells.
    // We make it the largest sane size that will fit into local mem
    std::size_t wgsCompacted = 512;
    while (wgsCompacted > 1 && wgsCompacted * NUM_EDGES * sizeof(cl_float3) >= localSize)
        wgsCompacted /= 2;
    generateElementsKernel.setArg(11, keyOffset);
    generateElementsKernel.setArg(13, cl::__local(NUM_EDGES * wgsCompacted * sizeof(cl_float3)));

    Swathe swathe;
    swathe.width = size[0];
    swathe.height = size[1];
    swathe.zStride = zStride;

    const Grid::size_type depth = size[2];
    MLSGPU_ASSERT(1U <= swathe.width && swathe.width <= maxWidth, std::length_error);
    MLSGPU_ASSERT(1U <= swathe.height && swathe.height <= maxHeight, std::length_error);
    MLSGPU_ASSERT(1U <= depth && depth <= maxDepth, std::length_error);

    std::vector<cl::Event> wait;
    cl::Event last, readEvent;
    cl_uint2 offsets = { {0, 0} };
    cl_uint3 top = { {2 * (swathe.width - 1), 2 * (swathe.height - 1), 0} };

    if (events != NULL)
        wait = *events;

    Grid::size_type shipOuts = 0;
    for (Grid::size_type z = 0; z < depth; z += maxSwathe)
    {
        swathe.zFirst = z;
        swathe.zLast = std::min(depth, z + maxSwathe) - 1;
        swathe.zBias = (1 - cl_int(z)) * cl_int(swathe.zStride);

        if (z != 0)
        {
            // Copy end of previous range to start of current one
            copySlice(queue, image, maxSwathe, 0, swathe, &wait, &last);
            wait.resize(1);
            wait[0] = last;
        }
        generator.enqueue(queue, image, swathe, &wait, &last);
        wait.resize(1);
        wait[0] = last;

        if (z > 0)
            swathe.zFirst--; // Use the copied previous slice as well

        shipOuts += addSlices(
            queue, output,
            swathe, keyOffset,
            cl::NDRange(wgsCompacted),
            offsets, top,
            &wait, &last);
        wait.resize(1);
        wait[0] = last;
    }

    if (offsets.s[0] > 0)
    {
        shipOut(queue, keyOffset, offsets, depth - 1, output, &wait, &last);
        shipOuts++;
        wait.resize(1);
        wait[0] = last;
    }
    if (shipOuts > 0)
        Statistics::getStatistic<Statistics::Variable>("marching.shipouts").add(shipOuts);
    queue.finish(); // will normally be finished already, but there may be corner cases
}
