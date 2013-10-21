/*
 * mlsgpu: surface reconstruction from point clouds
 * Copyright (C) 2013  University of Cape Town
 *
 * This file is part of mlsgpu.
 *
 * mlsgpu is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/**
 * @file
 *
 * Tests for @ref Marching.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif

#ifndef __CL_ENABLE_EXCEPTIONS
# define __CL_ENABLE_EXCEPTIONS
#endif

#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/extensions/HelperMacros.h>
#include <cstddef>
#include <vector>
#include <map>
#include <string>
#include <cmath>
#include <fstream>
#include <boost/array.hpp>
#include <boost/ref.hpp>
#include <CL/cl.hpp>
#include "testutil.h"
#include "test_clh.h"
#include "memory_writer.h"
#include "manifold.h"
#include "../src/clh.h"
#include "../src/marching.h"
#include "../src/mesher.h"
#include "../src/fast_ply.h"
#include "../src/misc.h"

using namespace std;

/**
 * Helper class to simplify writing generators that just generate
 * data on the host.
 */
class HostGenerator : public Marching::Generator, public boost::noncopyable
{
private:
    cl::Context context;
    std::size_t maxWidth, maxHeight, maxDepth;
    vector<float> sliceData;

protected:
    virtual cl_float generate(cl_uint x, cl_uint y, cl_uint z) const = 0;

    HostGenerator(
        const cl::Context &context,
        std::size_t maxWidth, std::size_t maxHeight, std::size_t maxDepth)
        : context(context),
        maxWidth(maxWidth), maxHeight(maxHeight), maxDepth(maxDepth),
        sliceData(maxWidth * maxHeight)
    {
    }

public:
    virtual const Grid::size_type *alignment() const
    {
        static const Grid::size_type ans[3] = { 7, 5, 11 }; // non power-of-two to make sure that works
        return ans;
    }

    virtual void enqueue(
        const cl::CommandQueue &queue,
        const cl::Image2D &distance,
        const Marching::Swathe &swathe,
        const std::vector<cl::Event> *events,
        cl::Event *event)
    {
        CPPUNIT_ASSERT(swathe.zStride >= roundUp(swathe.height + 1, alignment()[1]));
        CPPUNIT_ASSERT(0 < swathe.width);
        CPPUNIT_ASSERT(0 < swathe.height);
        CPPUNIT_ASSERT(swathe.width <= maxWidth);
        CPPUNIT_ASSERT(swathe.height <= maxHeight);
        CPPUNIT_ASSERT(swathe.zFirst <= swathe.zLast);
        CPPUNIT_ASSERT(distance.getImageInfo<CL_IMAGE_WIDTH>() >= roundUp(swathe.width, alignment()[0]));
        CPPUNIT_ASSERT(distance.getImageInfo<CL_IMAGE_HEIGHT>() >= swathe.zStride * roundUp(swathe.zLast + 1, alignment()[2]) + swathe.zBias);

        std::vector<cl::Event> wait;
        cl::Event last;
        if (events != NULL)
            wait = *events;

        for (cl_uint z = swathe.zFirst; z <= swathe.zLast; z++)
        {
            for (cl_uint y = 0; y < swathe.height; y++)
                for (cl_uint x = 0; x < swathe.width; x++)
                {
                    sliceData[y * swathe.width + x] = generate(x, y, z);
                }

            cl::size_t<3> origin, region;
            origin[0] = 0; origin[1] = z * swathe.zStride + swathe.zBias; origin[2] = 0;
            region[0] = swathe.width; region[1] = swathe.height; region[2] = 1;
            queue.enqueueWriteImage(distance, CL_TRUE, origin, region,
                                    swathe.width * sizeof(float), 0, &sliceData[0],
                                    &wait, &last);
            wait.resize(1);
            wait[0] = last;
        }
        if (event != NULL)
            *event = last;
    }
};


/**
 * Function object to model a signed distance from a sphere.
 */
class SphereGenerator : public HostGenerator
{
private:
    float cx, cy, cz;
    float radius;

protected:
    virtual cl_float generate(cl_uint x, cl_uint y, cl_uint z) const
    {
        cl_float d = std::sqrt((x - cx) * (x - cx) + (y - cx) * (y - cy) + (z - cz) * (z - cz));
        return d - radius;
    }

public:
    SphereGenerator(
        const cl::Context &context,
        Grid::size_type maxWidth, Grid::size_type maxHeight, Grid::size_type maxDepth,
        float cx, float cy, float cz, float radius)
        : HostGenerator(context, maxWidth, maxHeight, maxDepth),
        cx(cx), cy(cy), cz(cz), radius(radius)
    {
    }
};

/**
 * Signed distance generator that alternates positive and negative on every cell.
 */
class AlternatingGenerator : public HostGenerator
{
protected:
    virtual cl_float generate(cl_uint x, cl_uint y, cl_uint z) const
    {
        return ((x ^ y ^ z) & 1) ? 1.0f : -1.0f;
    }

public:
    AlternatingGenerator(
        const cl::Context &context,
        Grid::size_type maxWidth, Grid::size_type maxHeight, Grid::size_type maxDepth)
        : HostGenerator(context, maxWidth, maxHeight, maxDepth)
    {
    }
};

/**
 * Tests for @ref Marching.
 */
class TestMarching : public CLH::Test::TestFixture
{
    CPPUNIT_TEST_SUITE(TestMarching);
    CPPUNIT_TEST(testConstructor);
    CPPUNIT_TEST(testComputeKey);
    CPPUNIT_TEST(testCompactVertices);
    CPPUNIT_TEST(testCopySlice);
    CPPUNIT_TEST(testSphere);
    CPPUNIT_TEST(testTruncatedSphere);
    CPPUNIT_TEST(testAlternating);
    CPPUNIT_TEST_SUITE_END();

private:
    /// Read the contents of a buffer and return it (synchronously) as a vector.
    template<typename T>
    vector<T> bufferToVector(const cl::Buffer &buffer);

    /// Read the contents of a vector and return it (synchronously) as a buffer.
    template<typename T>
    cl::Buffer vectorToBuffer(cl_mem_flags flags, const vector<T> &v);

    /// Build a vertex key
    static cl_ulong makeKey(cl_uint x, cl_uint y, cl_uint z, bool external);
    /// Wrapper that calls @ref computeKey and returns result
    cl_ulong callComputeKey(cl::Kernel &kernel,
                            cl_uint cx, cl_uint cy, cl_uint cz,
                            cl_uint tx, cl_uint ty, cl_uint tz);

    /**
     * Wrapper that calls @ref compactVertices and returns the results in host memory.
     * The output vectors are completely overwritten, so the incoming contents have
     * no effect.
     *
     * @param kernel            The @ref compactVertices kernel.
     * @param outSize           Entries to allocate for output vertices.
     * @param remapSize         Entries to allocate in the index remap table.
     * @param outVertices, outKeys, indexRemap, firstExternal, vertexUnique, inVertices, inKeys, minExternalKey See @ref compactVertices.
     */
    void callCompactVertices(
        cl::Kernel &kernel,
        size_t outSize, size_t remapSize,
        vector<cl_float> &outVertices,
        vector<cl_ulong> &outKeys,
        vector<cl_uint> &indexRemap,
        cl_uint &firstExternal,
        const vector<cl_uint> &vertexUnique,
        const vector<cl_float4> &inVertices,
        const vector<cl_ulong> &inKeys,
        cl_ulong minExternalKey);

    /**
     * Generate a mesh using an input functor, validate that it is manifold,
     * and write it to file.
     */
    void testGenerate(
        Grid::size_type maxWidth, Grid::size_type maxHeight, Grid::size_type maxDepth,
        Grid::size_type width, Grid::size_type height, Grid::size_type depth,
        Marching::Generator &generator, const std::string &filename);

    void testConstructor();     ///< Basic sanity tests on the tables
    void testComputeKey();      ///< Test @ref computeKey helper function
    void testCompactVertices(); ///< Test @ref compactVertices kernel
    void testCopySlice();       ///< Test @ref copySlice, both kernel and wrapper function
    void testSphere();          ///< Builds a sphere
    void testTruncatedSphere(); ///< Builds a sphere that is truncated by the bounding box
    void testAlternating();     ///< Build a structure with lots of geometry
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestMarching, TestSet::perCommit());

template<typename T>
vector<T> TestMarching::bufferToVector(const cl::Buffer &buffer)
{
    size_t size = buffer.getInfo<CL_MEM_SIZE>();
    // We allow size == 1 so that empty buffers can be rounded up to a legal size.
    CPPUNIT_ASSERT(size % sizeof(T) == 0 || size == 1);
    vector<T> ans(size / sizeof(T));
    CLH::enqueueReadBuffer(queue, buffer, CL_TRUE, 0, size, &ans[0]);
    return ans;
}

template<typename T>
cl::Buffer TestMarching::vectorToBuffer(cl_mem_flags flags, const vector<T> &v)
{
    CPPUNIT_ASSERT(!v.empty());
    size_t size = v.size() * sizeof(T);
    return cl::Buffer(context, flags | CL_MEM_COPY_HOST_PTR, size, (void *) (&v[0]));
}

void TestMarching::testConstructor()
{
    AlternatingGenerator generator(context, 2, 2, 2);
    Marching marching(context, device, 2, 2, 2, generator.alignment()[2],
                      4096,
                      generator.alignment());

    vector<cl_uchar2> countTable = bufferToVector<cl_uchar2>(marching.countTable);
    vector<cl_ushort2> startTable = bufferToVector<cl_ushort2>(marching.startTable);
    vector<cl_uchar> dataTable = bufferToVector<cl_uchar>(marching.dataTable);

    CPPUNIT_ASSERT_EQUAL(256, int(countTable.size()));
    CPPUNIT_ASSERT_EQUAL(257, int(startTable.size()));
    CPPUNIT_ASSERT_EQUAL(int(startTable.back().s[1]), int(dataTable.size()));
    CPPUNIT_ASSERT_EQUAL(0, int(startTable.front().s[0]));
    for (unsigned int i = 0; i < 256; i++)
    {
        int sv = startTable[i].s[0];
        int si = startTable[i].s[1];
        int ev = startTable[i + 1].s[0];
        int ei = startTable[i + 1].s[1];
        CPPUNIT_ASSERT_EQUAL(int(countTable[i].s[0]), ev - sv);
        CPPUNIT_ASSERT_EQUAL(int(countTable[i].s[1]), ei - si);
        CPPUNIT_ASSERT(countTable[i].s[1] % 3 == 0);
        for (int j = sv; j < ev; j++)
        {
            if (j > sv)
                CPPUNIT_ASSERT(dataTable[j - 1] < dataTable[j]);
            CPPUNIT_ASSERT(dataTable[j] < 19);
        }
        for (int j = si; j < ei; j++)
        {
            CPPUNIT_ASSERT(dataTable[j] < ev - sv);
        }
    }
}

cl_ulong TestMarching::makeKey(cl_uint x, cl_uint y, cl_uint z, bool external)
{
    cl_ulong ans = (cl_ulong(z) << 42) | (cl_ulong(y) << 21) | (cl_ulong(x));
    if (external)
        ans |= cl_ulong(1) << 63;
    return ans;
}

cl_ulong TestMarching::callComputeKey(
    cl::Kernel &kernel,
    cl_uint cx, cl_uint cy, cl_uint cz,
    cl_uint tx, cl_uint ty, cl_uint tz)
{
    cl_ulong ans = 0;
    cl_uint3 coords = {{ cx, cy, cz }};
    cl_uint3 top = {{ tx, ty, tz }};
    cl::Buffer out(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_ulong), &ans);

    kernel.setArg(0, out);
    kernel.setArg(1, coords);
    kernel.setArg(2, top);
    queue.enqueueTask(kernel);
    queue.enqueueReadBuffer(out, CL_TRUE, 0, sizeof(cl_ulong), &ans);
    return ans;
}

void TestMarching::testComputeKey()
{
    map<string, string> defines;
    defines["UNIT_TESTS"] = "1";
    cl::Program program = CLH::build(context, "kernels/marching.cl", defines);
    cl::Kernel kernel(program, "testComputeKey");

    CPPUNIT_ASSERT_EQUAL(makeKey(0, 0, 0, true),    callComputeKey(kernel, 0, 0, 0, 32, 32, 32));
    CPPUNIT_ASSERT_EQUAL(makeKey(1, 2, 3, false),   callComputeKey(kernel, 1, 2, 3, 32, 32, 32));
    CPPUNIT_ASSERT_EQUAL(makeKey(0, 4, 5, true),    callComputeKey(kernel, 0, 4, 5, 32, 32, 32));
    CPPUNIT_ASSERT_EQUAL(makeKey(6, 0, 7, true),    callComputeKey(kernel, 6, 0, 7, 32, 32, 32));
    CPPUNIT_ASSERT_EQUAL(makeKey(9, 5, 0, false),   callComputeKey(kernel, 9, 5, 0, 32, 32, 32));
    CPPUNIT_ASSERT_EQUAL(makeKey(30, 1, 2, true),   callComputeKey(kernel, 30, 1, 2, 30, 40, 50));
    CPPUNIT_ASSERT_EQUAL(makeKey(5, 40, 3, true),   callComputeKey(kernel, 5, 40, 3, 30, 40, 50));
    CPPUNIT_ASSERT_EQUAL(makeKey(1, 2, 50, true),   callComputeKey(kernel, 1, 2, 50, 30, 40, 50));
    CPPUNIT_ASSERT_EQUAL(makeKey(1, 2, 40, false),  callComputeKey(kernel, 1, 2, 40, 30, 40, 50));
    CPPUNIT_ASSERT_EQUAL(makeKey(1, 2, 30, false),  callComputeKey(kernel, 1, 2, 30, 30, 40, 50));
    CPPUNIT_ASSERT_EQUAL(makeKey(30, 40, 50, true), callComputeKey(kernel, 30, 40, 50, 30, 40, 50));
}

void TestMarching::callCompactVertices(
    cl::Kernel &kernel,
    size_t outSize, size_t remapSize,
    vector<cl_float> &outVertices,
    vector<cl_ulong> &outKeys,
    vector<cl_uint> &indexRemap,
    cl_uint &firstExternal,
    const vector<cl_uint> &vertexUnique,
    const vector<cl_float4> &inVertices,
    const vector<cl_ulong> &inKeys,
    cl_ulong minExternalKey)
{
    const size_t inSize = inVertices.size();
    cl::Buffer dOutVertices   = createBuffer(CL_MEM_WRITE_ONLY, outSize * (3 * sizeof(cl_float)));
    cl::Buffer dOutKeys       = createBuffer(CL_MEM_WRITE_ONLY, outSize * sizeof(cl_ulong));
    cl::Buffer dIndexRemap    = createBuffer(CL_MEM_WRITE_ONLY, remapSize * sizeof(cl_uint));
    cl::Buffer dFirstExternal = createBuffer(CL_MEM_WRITE_ONLY, sizeof(cl_uint));
    cl::Buffer dVertexUnique  = vectorToBuffer(CL_MEM_READ_ONLY, vertexUnique);
    cl::Buffer dInVertices    = vectorToBuffer(CL_MEM_READ_ONLY, inVertices);
    cl::Buffer dInKeys        = vectorToBuffer(CL_MEM_READ_ONLY, inKeys);

    kernel.setArg(0, dOutVertices);
    kernel.setArg(1, dOutKeys);
    kernel.setArg(2, dIndexRemap);
    kernel.setArg(3, dFirstExternal);
    kernel.setArg(4, dVertexUnique);
    kernel.setArg(5, dInVertices);
    kernel.setArg(6, dInKeys);
    kernel.setArg(7, minExternalKey);
    kernel.setArg(8, cl_ulong(0));
    CLH::enqueueNDRangeKernel(queue,
                              kernel,
                              cl::NullRange,
                              cl::NDRange(inSize),
                              cl::NullRange);
    outVertices = bufferToVector<cl_float>(dOutVertices);
    outKeys = bufferToVector<cl_ulong>(dOutKeys);
    indexRemap = bufferToVector<cl_uint>(dIndexRemap);
    queue.enqueueReadBuffer(dFirstExternal, CL_TRUE, 0, sizeof(cl_uint), &firstExternal);
}

static inline cl_float uintAsFloat(cl_uint x)
{
    cl_float y;
    memcpy(&y, &x, sizeof(y));
    return y;
}

void TestMarching::testCompactVertices()
{
    const cl_ulong externalBit = cl_ulong(1) << 63;
    const cl_ulong hInKeys[6] = { 100, 100, 200, externalBit | 50, externalBit | 50, CL_ULONG_MAX };
    const cl_uint hVertexUnique[6] = { 0, 0, 1, 2, 2, 3 };
    const cl_uint ids[5] = { 4, 1, 2, 3, 0 };

    vector<cl_float> outVertices;
    vector<cl_ulong> outKeys;
    vector<cl_uint> indexRemap;
    cl_uint firstExternal;
    vector<cl_uint> vertexUnique(hVertexUnique, hVertexUnique + 6);
    vector<cl_float4> inVertices(5);
    vector<cl_ulong> inKeys(hInKeys, hInKeys + 6);

    for (int i = 0; i < 5; i++)
    {
        inVertices[i].s[0] = i;
        inVertices[i].s[1] = i + 1;
        inVertices[i].s[2] = i + 2;
        inVertices[i].s[3] = uintAsFloat(ids[i]);
    }

    AlternatingGenerator generator(context, 2, 2, 2);
    Marching marching(context, device, 2, 2, 2,
                      generator.alignment()[2], 4096, generator.alignment());
    callCompactVertices(marching.compactVerticesKernel, 3, 5,
                        outVertices, outKeys, indexRemap, firstExternal,
                        vertexUnique, inVertices, inKeys, 200);

    CPPUNIT_ASSERT_EQUAL(1.0f, outVertices[0]);
    CPPUNIT_ASSERT_EQUAL(2.0f, outVertices[3]);
    CPPUNIT_ASSERT_EQUAL(4.0f, outVertices[6]);
    CPPUNIT_ASSERT_EQUAL(cl_ulong(0xDEADBEEFDEADBEEFull), outKeys[0]); // should not be overwritten
    CPPUNIT_ASSERT_EQUAL(cl_ulong(200), outKeys[1]);
    CPPUNIT_ASSERT_EQUAL(cl_ulong(50), outKeys[2]);
    CPPUNIT_ASSERT_EQUAL(cl_uint(2), indexRemap[0]);
    CPPUNIT_ASSERT_EQUAL(cl_uint(0), indexRemap[1]);
    CPPUNIT_ASSERT_EQUAL(cl_uint(1), indexRemap[2]);
    CPPUNIT_ASSERT_EQUAL(cl_uint(2), indexRemap[3]);
    CPPUNIT_ASSERT_EQUAL(cl_uint(0), indexRemap[4]);
    CPPUNIT_ASSERT_EQUAL(cl_uint(1), firstExternal);

    // Same thing, but with all vertices external
    callCompactVertices(marching.compactVerticesKernel, 3, 5,
                        outVertices, outKeys, indexRemap, firstExternal,
                        vertexUnique, inVertices, inKeys, 100);

    CPPUNIT_ASSERT_EQUAL(1.0f, outVertices[0]);
    CPPUNIT_ASSERT_EQUAL(2.0f, outVertices[3]);
    CPPUNIT_ASSERT_EQUAL(4.0f, outVertices[6]);
    CPPUNIT_ASSERT_EQUAL(cl_ulong(100), outKeys[0]);
    CPPUNIT_ASSERT_EQUAL(cl_ulong(200), outKeys[1]);
    CPPUNIT_ASSERT_EQUAL(cl_ulong(50), outKeys[2]);
    CPPUNIT_ASSERT_EQUAL(cl_uint(2), indexRemap[0]);
    CPPUNIT_ASSERT_EQUAL(cl_uint(0), indexRemap[1]);
    CPPUNIT_ASSERT_EQUAL(cl_uint(1), indexRemap[2]);
    CPPUNIT_ASSERT_EQUAL(cl_uint(2), indexRemap[3]);
    CPPUNIT_ASSERT_EQUAL(cl_uint(0), indexRemap[4]);
    CPPUNIT_ASSERT_EQUAL(cl_uint(0), firstExternal);

    // Same again, but with all vertices internal
    callCompactVertices(marching.compactVerticesKernel, 3, 5,
                        outVertices, outKeys, indexRemap, firstExternal,
                        vertexUnique, inVertices, inKeys, externalBit | 60);

    CPPUNIT_ASSERT_EQUAL(1.0f, outVertices[0]);
    CPPUNIT_ASSERT_EQUAL(2.0f, outVertices[3]);
    CPPUNIT_ASSERT_EQUAL(4.0f, outVertices[6]);
    CPPUNIT_ASSERT_EQUAL(cl_ulong(0xDEADBEEFDEADBEEFull), outKeys[0]); // should not be overwritten
    CPPUNIT_ASSERT_EQUAL(cl_ulong(0xDEADBEEFDEADBEEFull), outKeys[1]); // should not be overwritten
    CPPUNIT_ASSERT_EQUAL(cl_ulong(0xDEADBEEFDEADBEEFull), outKeys[2]); // should not be overwritten
    CPPUNIT_ASSERT_EQUAL(cl_uint(2), indexRemap[0]);
    CPPUNIT_ASSERT_EQUAL(cl_uint(0), indexRemap[1]);
    CPPUNIT_ASSERT_EQUAL(cl_uint(1), indexRemap[2]);
    CPPUNIT_ASSERT_EQUAL(cl_uint(2), indexRemap[3]);
    CPPUNIT_ASSERT_EQUAL(cl_uint(0), indexRemap[4]);
    CPPUNIT_ASSERT_EQUAL(cl_uint(3), firstExternal);
}

void TestMarching::testCopySlice()
{
    cl_float values[8][2] =
    {
        { 0.1f, 0.1f },
        { 0.1f, 0.1f },
        { 0.1f, 0.1f },
        { 0.1f, 0.1f },
        { 1.5f, -0.5f },
        { 2.0f, -3.0f },
        { 0.1f, 0.1f },
        { 0.1f, 0.1f }
    };

    const cl_float expected[8][2] =
    {
        { 1.5f, -0.5f },
        { 2.0f, -3.0f },
        { 1.5f, -0.5f },
        { 2.0f, -3.0f },
        { 1.5f, -0.5f },
        { 2.0f, -3.0f },
        { 0.1f, 0.1f },
        { 0.1f, 0.1f }
    };

    cl::Image2D image(
        context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, cl::ImageFormat(CL_R, CL_FLOAT),
        2, 8, 0, values);

    Marching::ImageParams params;
    params.width = 2;
    params.height = 2;
    params.zStride = 2;
    params.zBias = 1; // should not be used

    AlternatingGenerator generator(context, 2, 2, 4);
    Marching marching(context, device, 2, 2, 4,
                      generator.alignment()[2], 4096, generator.alignment());

    cl::size_t<3> srcOrigin, trgOrigin, region;
    srcOrigin[0] = 0; srcOrigin[1] = 4; srcOrigin[2] = 0;
    trgOrigin[0] = 0; trgOrigin[1] = 2; trgOrigin[2] = 0;
    region[0] = 2; region[1] = 2; region[2] = 1;

    cl_int2 trgOffset = {{ 0, -2 }};
    marching.copySliceKernel.setArg(0, image);
    marching.copySliceKernel.setArg(1, image);
    marching.copySliceKernel.setArg(2, trgOffset);

    queue.enqueueNDRangeKernel(
        marching.copySliceKernel,
        cl::NDRange(0, 4),
        cl::NDRange(2, 2),
        cl::NDRange(2, 1));
    queue.enqueueBarrier();
    marching.copySlice(queue, image, 2, 0, params, NULL, NULL);
    queue.finish();

    memset(values, 0, sizeof(values));
    srcOrigin[1] = 0;
    region[1] = 8;
    queue.enqueueReadImage(image, CL_TRUE, srcOrigin, region, 0, 0, &values[0][0]);

    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 2; j++)
            CPPUNIT_ASSERT_EQUAL(expected[i][j], values[i][j]);
}

void TestMarching::testGenerate(
    Grid::size_type maxWidth, Grid::size_type maxHeight, Grid::size_type maxDepth,
    Grid::size_type width, Grid::size_type height, Grid::size_type depth,
    Marching::Generator &generator,
    const std::string &filename)
{
    Timeplot::Worker tworker("test");

    Grid::size_type size[3] = { width, height, depth };

    cl_uint3 keyOffset = {{ 0, 0, 0 }};
    Grid::size_type swathe = generator.alignment()[2];
    Marching marching(context, device, maxWidth, maxHeight, maxDepth,
                      swathe,
                      (maxWidth - 1) * (maxHeight - 1) * Marching::MAX_CELL_BYTES,
                      generator.alignment());

    /*** Pass 1: write to file ***/

    {
        FastPly::Writer writer(STREAM_WRITER);
        OOCMesher mesher(writer, TrivialNamer(filename));
        marching.generate(queue, generator, deviceMesher(mesher.functor(0), ChunkId(), tworker), size, keyOffset, NULL);
        mesher.write(tworker);
    }

    /*** Pass 2: write to memory and validate ***/

    {
        MemoryWriterPly writer;
        OOCMesher mesher(writer, TrivialNamer(filename));
        marching.generate(queue, generator, deviceMesher(mesher.functor(0), ChunkId(), tworker), size, keyOffset, NULL);
        mesher.write(tworker);

        const std::string &output = writer.getOutput(filename);
        std::vector<boost::array<float, 3> > vertices;
        std::vector<boost::array<std::tr1::uint32_t, 3> > triangles;
        writer.parse(output, vertices, triangles);

        std::string reason = Manifold::isManifold(vertices.size(), triangles.begin(), triangles.end());
        CPPUNIT_ASSERT_EQUAL(string(""), reason);
    }
}

void TestMarching::testSphere()
{
    const Grid::size_type maxWidth = 83;
    const Grid::size_type maxHeight = 78;
    const Grid::size_type maxDepth = 66;
    const Grid::size_type width = 71;
    const Grid::size_type height = 75;
    const Grid::size_type depth = 60;

    SphereGenerator generator(context, maxWidth, maxHeight, maxDepth, 30.0, 41.5, 27.75, 25.3);
    testGenerate(maxWidth, maxHeight, maxDepth, width, height, depth,
                 generator, "sphere.ply");
}

void TestMarching::testTruncatedSphere()
{
    const Grid::size_type maxWidth = 83;
    const Grid::size_type maxHeight = 78;
    const Grid::size_type maxDepth = 66;
    const Grid::size_type width = 71;
    const Grid::size_type height = 75;
    const Grid::size_type depth = 60;

    SphereGenerator generator(context, maxWidth, maxHeight, maxDepth,
                              0.5f * width, 0.5f * height, 0.5f * depth, 42.0f);
    testGenerate(maxWidth, maxHeight, maxDepth, width, height, depth,
                 generator, "tsphere.ply");
}

void TestMarching::testAlternating()
{
    const Grid::size_type width = 32;
    const Grid::size_type height = 32;
    const Grid::size_type depth = 32;

    AlternatingGenerator generator(context, width, height, depth);
    testGenerate(width, height, depth, width, height, depth,
                 generator, "alternating.ply");
}
