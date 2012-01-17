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
#include <CL/cl.hpp>
#include "testmain.h"
#include "test_clh.h"
#include "../src/clh.h"
#include "../src/marching.h"
#include "../src/mesh.h"

using namespace std;

/**
 * Function object to model a signed distance from a sphere.
 */
class SphereFunc
{
private:
    std::size_t width, height, depth;
    float cx, cy, cz;
    float radius;
    vector<float> sliceData;

public:
    void operator()(const cl::CommandQueue &queue, const cl::Image2D &slice,
                    cl_uint z,
                    const std::vector<cl::Event> *events,
                    cl::Event *event)
    {
        for (cl_uint y = 0; y < height; y++)
            for (cl_uint x = 0; x < width; x++)
            {
                float d = std::sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy) + (z - cz) * (z - cz));
                sliceData[y * width + x] = d - radius;
            }

        cl::size_t<3> origin, region;
        origin[0] = 0; origin[1] = 0; origin[2] = 0;
        region[0] = width; region[1] = height; region[2] = 1;
        queue.enqueueWriteImage(slice, CL_FALSE, origin, region, width * sizeof(float), 0, &sliceData[0],
                                events, event);
    }

    SphereFunc(std::size_t width, std::size_t height, std::size_t depth,
               float cx, float cy, float cz, float radius)
        : width(width), height(height), depth(depth),
        cx(cx), cy(cy), cz(cz), radius(radius),
        sliceData(width * height)
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
    CPPUNIT_TEST(testSphere);
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

    void testConstructor();     ///< Basic sanity tests on the tables
    void testComputeKey();      ///< Test @ref computeKey helper function
    void testCompactVertices(); ///< Test @ref compactVertices kernel
    void testSphere();          ///< Builds a sphere
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestMarching, TestSet::perCommit());

template<typename T>
vector<T> TestMarching::bufferToVector(const cl::Buffer &buffer)
{
    size_t size = buffer.getInfo<CL_MEM_SIZE>();
    CPPUNIT_ASSERT(size % sizeof(T) == 0);
    vector<T> ans(size / sizeof(T));
    queue.enqueueReadBuffer(buffer, CL_TRUE, 0, size, &ans[0]);
    return ans;
}

template<typename T>
cl::Buffer TestMarching::vectorToBuffer(cl_mem_flags flags, const vector<T> &v)
{
    size_t size = v.size() * sizeof(T);
    return cl::Buffer(context, flags | CL_MEM_COPY_HOST_PTR, size, (void *) (&v[0]));
}

void TestMarching::testConstructor()
{
    Marching marching(context, device, 2, 2);

    vector<cl_uchar2> countTable = bufferToVector<cl_uchar2>(marching.countTable);
    vector<cl_ushort2> startTable = bufferToVector<cl_ushort2>(marching.startTable);
    vector<cl_uchar> dataTable = bufferToVector<cl_uchar>(marching.dataTable);

    CPPUNIT_ASSERT_EQUAL(256, int(countTable.size()));
    CPPUNIT_ASSERT_EQUAL(257, int(startTable.size()));
    CPPUNIT_ASSERT_EQUAL(int(startTable.back().s1), int(dataTable.size()));
    CPPUNIT_ASSERT_EQUAL(0, int(startTable.front().s0));
    for (unsigned int i = 0; i < 256; i++)
    {
        int sv = startTable[i].s0;
        int si = startTable[i].s1;
        int ev = startTable[i + 1].s0;
        int ei = startTable[i + 1].s1;
        CPPUNIT_ASSERT_EQUAL(int(countTable[i].s0), ev - sv);
        CPPUNIT_ASSERT_EQUAL(int(countTable[i].s1), ei - si);
        CPPUNIT_ASSERT(countTable[i].s1 % 3 == 0);
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
    queue.enqueueNDRangeKernel(kernel,
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
        inVertices[i].x = i;
        inVertices[i].y = i + 1;
        inVertices[i].z = i + 2;
        inVertices[i].w = uintAsFloat(ids[i]);
    }

    Marching marching(context, device, 2, 2);
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

void TestMarching::testSphere()
{
    const std::size_t maxWidth = 83;
    const std::size_t maxHeight = 78;
    const std::size_t width = 71;
    const std::size_t height = 75;
    const std::size_t depth = 60;
    cl::Event done;

    const float ref[3] = {0.0f, 0.0f, 0.0f};
    Grid grid(ref, 1.0f, 0, width - 1, 0, height - 1, 0, depth - 1);

    // Replace the command queue with an out-of-order one, to ensure that the
    // events are being handled correctly.
    queue = cl::CommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);

    Marching marching(context, device, maxWidth, maxHeight);
    SphereFunc input(width, height, depth, 30.0, 41.5, 27.75, 25.3);

    SimpleMesh mesh;
    cl_uint3 keyOffset = {{ 0, 0, 0 }};
    marching.generate(queue, input, mesh.outputFunctor(0), grid, keyOffset, 0, NULL);

    mesh.finalize();
    mesh.write("sphere.ply");
}
