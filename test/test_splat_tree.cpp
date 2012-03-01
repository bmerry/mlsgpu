/**
 * @file
 *
 * Test code for @ref SplatTree.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif

#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/extensions/HelperMacros.h>
#include <vector>
#include <set>
#include <numeric>
#include <algorithm>
#include <tr1/random>
#include "testmain.h"
#include "../src/splat.h"
#include "../src/grid.h"
#include "../src/splat_tree.h"
#include "test_splat_tree.h"

using namespace std;

void TestSplatTree::testMakeCode()
{
    /* 123 = 0001111011b
     * 456 = 0111001000b
     * 789 = 1100010101b
     * Morton code: 100110010011001101011100001101b = 642569997
     */
    CPPUNIT_ASSERT_EQUAL(SplatTree::code_type(642569997U), SplatTree::makeCode(123, 456, 789));
}

static void addSplat(vector<Splat> &splats, float x, float y, float z, float r)
{
    Splat s;
    s.position[0] = x;
    s.position[1] = y;
    s.position[2] = z;
    s.radius = r;
    // Normal and quality are irrelevant - just init to avoid undefined data
    s.normal[0] = 1.0f;
    s.normal[1] = 0.0f;
    s.normal[2] = 0.0f;
    s.quality = 1.0f;
    splats.push_back(s);
}

void TestSplatTree::testBuild()
{
    typedef SplatTree::command_type command_type;
    const float ref[3] = {30.0f, 0.0f, 10.0f};
    Grid grid(ref, 10.0f, 0, 15, 0, 15, 0, 11);
    vector<Splat> splats;

    // bbox: 7,7,7 - 8,8,8. level 4, 8 nodes [7,7,7-9,9,9)
    addSplat(splats, 105.0f, 75.0f, 85.0f, 10.0f);
    // bbox: 7,8,3 - 10,11,6. level 2, 4 nodes [1,2,0-3,3,2) -> [4,8,0-12,12,8)
    addSplat(splats, 115.0f, 95.0f, 55.0f, 20.0f);
    // bbox: 3,6,1 - 11,14,9. level 1, 8 nodes [0,0,0-2,2,2) -> [0,0,0-16,16,16)
    addSplat(splats, 100.0f, 100.0f, 60.0f, 45.0f);
    // bbox: 0,0,0 - 0,1,0. level 4, 2 nodes [0,0,0-1,2,1).
    addSplat(splats, 30.0f, 5.0f, 10.0f, 7.5f);
    // bbox: 0,0,0 - 0,0,0. level 4, 1 node [0,0,0-1,1,1).
    addSplat(splats, 30.0f, 0.0f, 10.0f, 5.0f);

    // Various spheres lying entirely outside the octree
    addSplat(splats, 190.0f, 80.0f, 50.0f, 6.0f);
    addSplat(splats, 0.0f, 10.0f, 10.0f, 25.0f);
    addSplat(splats, 50.0f, 10000.0f, 50.0f, 9000.0f);

    std::size_t numLevels;
    std::vector<command_type> commands;
    std::vector<command_type> start;
    build(numLevels, commands, start, splats, 9, 0, 1001, grid);

    CPPUNIT_ASSERT_EQUAL(std::size_t(5), numLevels);

    /* Do a walk from each octree cell to check that the right elements are visited */
    CPPUNIT_ASSERT(size_t(16 * 16 * 16) <= start.size());
    command_type maxPos = -1; // highest-numbered command used
    for (unsigned int z = 0; z < 12; z++)
        for (unsigned int y = 0; y < 16; y++)
            for (unsigned int x = 0; x < 16; x++)
            {
                unsigned int idx = SplatTree::makeCode(x, y, z);
                command_type pos = start[idx];
                maxPos = max(maxPos, pos);
                set<unsigned int> foundSplats;
                if (pos != -1)
                {
                    int ttl = 1000;
                    while (ttl > 0)  // prevents a loop from making the test run forever
                    {
                        maxPos = max(maxPos, pos);
                        ttl--;
                        CPPUNIT_ASSERT(pos >= 0 && pos < (command_type) commands.size());
                        command_type cmd = commands[pos];
                        if (cmd == -1)
                            break;
                        else if (cmd < 0)
                        {
                            pos = -2 - cmd;
                            maxPos = max(maxPos, pos);
                            CPPUNIT_ASSERT(pos >= 0 && pos < (command_type) commands.size());
                            cmd = commands[pos];
                        }
                        CPPUNIT_ASSERT(cmd >= 0 && cmd < (command_type) splats.size());
                        CPPUNIT_ASSERT(!foundSplats.count(cmd));
                        foundSplats.insert(cmd);
                        pos++;
                    }
                    CPPUNIT_ASSERT(ttl > 0);
                }

                // Check against splats we must see
                float lo[3], hi[3]; // corners of cell in world space
                grid.getVertex(x, y, z, lo);
                grid.getVertex(x + 1, y + 1, z + 1, hi);
                for (unsigned int i = 0; i < splats.size(); i++)
                {
                    float dist2 = 0.0f; // squared distance from splat center to nearest point in cell
                    for (unsigned int j = 0; j < 3; j++)
                    {
                        float n = max(min(splats[i].position[j], hi[j]), lo[j]);
                        n -= splats[i].position[j];
                        dist2 += n * n;
                    }
                    if (dist2 <= splats[i].radius * splats[i].radius)
                    {
                        CPPUNIT_ASSERT(foundSplats.count(i));
                    }
                }
            }

    /* Check that no splat appears more than 8 times in the command list */
    map<unsigned int, int> repeats;
    for (command_type i = 0; i <= maxPos; i++)
    {
        command_type cmd = commands[i];
        if (cmd >= 0)
        {
            repeats[cmd]++;
            CPPUNIT_ASSERT(repeats[cmd] <= 8);
        }
    }
}

void TestSplatTree::testRandom()
{
    typedef SplatTree::command_type command_type;
    typedef tr1::mt19937 engine_type;
    typedef tr1::uniform_real<float> dist_type;
    typedef tr1::variate_generator<engine_type &, dist_type> gen_type;
    engine_type engine;

    // These values are chosen from a real-world failure case
    const int numSplats = 207;
    const int maxSplats = 1000;
    const int subsamplingShift = 2;
    const int maxLevels = 8;
    const int cells[3] = {31, 31, 16};
    gen_type xyzGen[3] =
    {
        gen_type(engine, dist_type(-1.0f, cells[0] + 1.0f)),
        gen_type(engine, dist_type(-1.0f, cells[1] + 1.0f)),
        gen_type(engine, dist_type(-1.0f, cells[2] + 1.0f))
    };
    gen_type rGen(engine, dist_type(0.25f, 8.0f));

    const float ref[3] = {0.0f, 0.0f, 0.0f};
    Grid grid(ref, 1.0f, 0, cells[0], 0, cells[1], 0, cells[2]);

    vector<Splat> splats;
    for (int i = 0; i < numSplats; i++)
    {
        addSplat(splats, xyzGen[0](), xyzGen[1](), xyzGen[2](), rGen());
    }

    std::size_t numLevels;
    std::vector<command_type> commands;
    std::vector<command_type> start;
    build(numLevels, commands, start, splats, maxLevels, subsamplingShift, maxSplats, grid);

    // Try each start value and check that it gives a terminating sequence of valid splat IDs
    for (int z = 0; z <= cells[2]; z += 1 << subsamplingShift)
        for (int y = 0; y <= cells[1]; y += 1 << subsamplingShift)
            for (int x = 0; x <= cells[0]; x += 1 << subsamplingShift)
            {
                unsigned int idx = SplatTree::makeCode(x >> subsamplingShift,
                                                       y >> subsamplingShift,
                                                       z >> subsamplingShift);
                CPPUNIT_ASSERT(idx < start.size());
                if (start[idx] != -1)
                {
                    CPPUNIT_ASSERT(start[idx] >= 0 && size_t(start[idx]) < commands.size());
                    command_type pos = start[idx];
                    size_t steps = 0; // for detecting loops
                    while (size_t(steps) <= commands.size())
                    {
                        command_type cmd = commands[pos];
                        if (cmd == -1)
                            break;
                        else if (cmd < 0)
                        {
                            pos = -2 - cmd;
                            CPPUNIT_ASSERT(size_t(pos) < commands.size());
                            cmd = commands[pos];
                        }
                        CPPUNIT_ASSERT(cmd >= 0 && size_t(cmd) < splats.size());
                        steps++;
                        pos++;
                    }
                    CPPUNIT_ASSERT_MESSAGE("Infinite loop in command list", steps <= commands.size());
                }
            }
}
