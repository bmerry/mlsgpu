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
#include "../src/splat_tree_host.h"

using namespace std;

/**
 * Tests for @ref SplatTree.
 */
class TestSplatTree : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestSplatTree);
    CPPUNIT_TEST(testMakeCode);
    CPPUNIT_TEST(testConstructor);
    CPPUNIT_TEST_SUITE_END();
public:
    void testMakeCode();         ///< Test @ref SplatTree::makeCode
    void testConstructor();      ///< Test construction of internal state
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestSplatTree, TestSet::perBuild());

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

void TestSplatTree::testConstructor()
{
    typedef SplatTree::command_type command_type;
    const float ref[3] = {0.0f, 0.0f, 0.0f};
    const float dir[3][3] = { {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f} };
    Grid grid(ref, dir[0], dir[1], dir[2], 0, 15, 0, 15, 0, 11);
    vector<Splat> splats;

    // bbox: 7,7,7 - 8,8,8. level 4, 8 nodes [7,7,7-9,9,9)
    addSplat(splats, 7.5f, 7.5f, 7.5f, 1.0f);
    // bbox: 7,8,3 - 10,11,6. level 2, 4 nodes [1,2,0-3,3,2) -> [4,8,0-12,12,8)
    addSplat(splats, 8.5f, 9.5f, 4.5f, 2.0f);
    // bbox: 3,6,1 - 11,14,9. level 1, 8 nodes [0,0,0-2,2,2) -> [0,0,0-16,16,16)
    addSplat(splats, 7.0f, 10.0f, 5.0f, 4.5f);
    // bbox: 0,0,0 - 0,1,0. level 4, 2 nodes [0,0,0-1,2,1).
    addSplat(splats, 0.0f, 0.5f, 0.0f, 0.75f);
    // bbox: 0,0,0 - 0,0,0. level 4, 1 node [0,0,0-1,1,1).
    addSplat(splats, 0.0f, 0.0f, 0.0f, 0.5f);
    SplatTreeHost tree(splats, grid);

    const command_type expectedCommands[] =
    {
        2, -1,     // code 111
        2, -1,     // code 110
        2, -1,     // code 101
        2, -1,     // code 100
        2, -1,     // code 011
        2, -1,     // code 010
        2, -1,     // code 001
        2, -1,     // code 000
        1, -10,    // code 011 100
        1, -10,    // code 011 000
        1, -12,    // code 010 101
        /* Eliminated because the sphere doesn't cut the cell:
         * 1, -12,    // code 010 001
         */
        0, -2,     // code 111 000 000 000
        0, -4,     // code 110 001 001 001
        0, -6,     // code 101 010 010 010
        0, -8,     // code 100 001 001 001
        0, -18,    // code 011 100 100 100
        0, -22,    // code 010 101 101 101
        0, -14,    // code 001 110 110 110
        0, -16,    // code 000 111 111 111
        3, -16,    // code 000 000 000 010
        3, 4, -16  // code 000 000 000 000
    };
    const struct
    {
        unsigned int x0, y0, z0;
        unsigned int x1, y1, z1;
        unsigned int start;
    } regions[] =
    {
        { 8, 8, 8,  16, 16, 16,   0 },
        { 0, 8, 8,   8, 16, 16,   2 },
        { 8, 0, 8,  16,  8, 16,   4 },
        { 0, 0, 8,   8,  8, 16,   6 },
        { 8, 8, 0,  16, 16,  8,   8 },
        { 0, 8, 0,   8, 16,  8,  10 },
        { 8, 0, 0,  16,  8,  8,  12 },
        { 0, 0, 0,   8,  8,  8,  14 },
        { 8, 8, 4,  12, 12,  8,  16 },
        { 8, 8, 0,  12, 12,  4,  18 },
        { 4, 8, 4,   8, 12,  8,  20 },
        /* Eliminated because the sphere does not cut the cell.
         * { 4, 8, 0,   8, 12,  4,  22 },
         */
        { 8, 8, 8,   9,  9,  9,  22 },
        { 7, 8, 8,   8,  9,  9,  24 },
        { 8, 7, 8,   9,  8,  9,  26 },
        { 7, 7, 8,   8,  8,  9,  28 },
        { 8, 8, 7,   9,  9,  8,  30 },
        { 7, 8, 7,   8,  9,  8,  32 },
        { 8, 7, 7,   9,  8,  8,  34 },
        { 7, 7, 7,   8,  8,  8,  36 },
        { 0, 1, 0,   1,  2,  1,  38 },
        { 0, 0, 0,   1,  1,  1,  40 }
    };

    CPPUNIT_ASSERT_EQUAL(5U, tree.getNumLevels());

    // Validate commands
    CPPUNIT_ASSERT_EQUAL(sizeof(expectedCommands) / sizeof(expectedCommands[0]), tree.commands.size());
    for (size_t i = 0; i < tree.commands.size(); i++)
        CPPUNIT_ASSERT_EQUAL(expectedCommands[i], tree.commands[i]);

    // Validate start
    CPPUNIT_ASSERT_EQUAL(size_t(16 * 16 * 16), tree.start.size());
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
                CPPUNIT_ASSERT_EQUAL(expected, tree.start[idx]);
            }
}
