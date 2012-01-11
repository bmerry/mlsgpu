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
    const float dir[3][3] = { {10.f, 0.0f, 0.0f}, {0.0f, 10.f, 0.0f}, {0.0f, 0.0f, 10.f} };
    Grid grid(ref, dir[0], dir[1], dir[2], 0, 15, 0, 15, 0, 11);
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
    build(numLevels, commands, start, splats, grid);

    const command_type expectedCommands[] =
    {
        3, 4, -29, // code 000 000 000 000
        3, -29,    // code 000 000 000 010
        0, -29,    // code 000 111 111 111
        0, -31,    // code 001 110 110 110
        0, -23,    // code 010 101 101 101
        0, -27,    // code 011 100 100 100
        0, -37,    // code 100 001 001 001
        0, -39,    // code 101 010 010 010
        0, -41,    // code 110 001 001 001
        0, -43,    // code 111 000 000 000
        /* Eliminated because the sphere doesn't cut the cell:
         * 1, ?,   // code 010 001
         */
        1, -33,    // code 010 101
        1, -35,    // code 011 000
        1, -35,    // code 011 100
        2, -1,     // code 000
        2, -1,     // code 001
        2, -1,     // code 010
        2, -1,     // code 011
        2, -1,     // code 100
        2, -1,     // code 101
        2, -1,     // code 110
        2, -1,     // code 111
    };
    const struct
    {
        unsigned int x0, y0, z0;
        unsigned int x1, y1, z1;
        unsigned int start;
    } regions[] =
    {
        { 0, 0, 0,   8,  8,  8,  27 },
        { 8, 0, 0,  16,  8,  8,  29 },
        { 0, 8, 0,   8, 16,  8,  31 },
        { 8, 8, 0,  16, 16,  8,  33 },
        { 0, 0, 8,   8,  8, 16,  35 },
        { 8, 0, 8,  16,  8, 16,  37 },
        { 0, 8, 8,   8, 16, 16,  39 },
        { 8, 8, 8,  16, 16, 16,  41 },
        { 4, 8, 4,   8, 12,  8,  21 },
        { 8, 8, 0,  12, 12,  4,  23 },
        { 8, 8, 4,  12, 12,  8,  25 },
        /* Eliminated because the sphere does not cut the cell.
         * { 4, 8, 0,   8, 12,  4,  ? },
         */
        { 0, 0, 0,   1,  1,  1,   0 },
        { 0, 1, 0,   1,  2,  1,   3 },
        { 7, 7, 7,   8,  8,  8,   5 },
        { 8, 7, 7,   9,  8,  8,   7 },
        { 7, 8, 7,   8,  9,  8,   9 },
        { 8, 8, 7,   9,  9,  8,  11 },
        { 7, 7, 8,   8,  8,  9,  13 },
        { 8, 7, 8,   9,  8,  9,  15 },
        { 7, 8, 8,   8,  9,  9,  17 },
        { 8, 8, 8,   9,  9,  9,  19 }
    };

    CPPUNIT_ASSERT_EQUAL(std::size_t(5), numLevels);

    // Validate commands
    std::size_t nCommands = sizeof(expectedCommands) / sizeof(expectedCommands[0]);
    CPPUNIT_ASSERT(nCommands <= commands.size());
    for (size_t i = 0; i < nCommands; i++)
        CPPUNIT_ASSERT_EQUAL(expectedCommands[i], commands[i]);

    // Validate start
    CPPUNIT_ASSERT(size_t(16 * 16 * 16) <= start.size());
    for (unsigned int z = 0; z < 12; z++)
        for (unsigned int y = 0; y < 16; y++)
            for (unsigned int x = 0; x < 16; x++)
            {
                unsigned int idx = SplatTree::makeCode(x, y, z);
                command_type expected = -1;
                for (size_t i = 0; i < sizeof(regions) / sizeof(regions[0]); i++)
                {
                    if (regions[i].x0 <= x && x < regions[i].x1
                        && regions[i].y0 <= y && y < regions[i].y1
                        && regions[i].z0 <= z && z < regions[i].z1)
                        expected = regions[i].start;
                }
                CPPUNIT_ASSERT_EQUAL(expected, start[idx]);
            }
}
