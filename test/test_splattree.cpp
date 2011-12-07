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
#include "../src/splattree.h"

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
    CPPUNIT_ASSERT_EQUAL(SplatTree::size_type(642569997U), SplatTree::makeCode(123, 456, 789));
}

static void addSplat(vector<Splat> &splats, float x, float y, float z, float r)
{
    Splat s;
    s.position[0] = x;
    s.position[1] = y;
    s.position[2] = z;
    s.radiusSquared = r * r;
    // Normal and quality are irrelevant - just init to avoid undefined data
    s.normal[0] = 1.0f;
    s.normal[1] = 0.0f;
    s.normal[2] = 0.0f;
    s.quality = 1.0f;
    splats.push_back(s);
}

void TestSplatTree::testConstructor()
{
    typedef SplatTree::size_type size_type;
    const float ref[3] = {0.0f, 0.0f, 0.0f};
    const float dir[3][3] = { {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f} };
    Grid grid(ref, dir[0], dir[1], dir[2], 0, 15, 0, 15, 0, 11);
    vector<Splat> splats;

    // bbox: 7,7,7 - 8,8,8. level 4, 8 nodes (7,7,7-8,8,8)
    addSplat(splats, 7.5f, 7.5f, 7.5f, 1.0f);
    // bbox: 7,8,3 - 10,11,6. level 2, 4 nodes (1,2,0-2,2,1)
    addSplat(splats, 8.5f, 9.5f, 4.5f, 2.0f);
    // bbox: 3,6,1 - 11,14,9. level 1, 8 nodes (0,0,0-1,1,1)
    addSplat(splats, 7.0f, 10.0f, 5.0f, 4.5f);
    // bbox: 0,0,0 - 0,1,0. level 4, 2 nodes.
    addSplat(splats, 0.0f, 0.5f, 0.0f, 0.75f);
    // bbox: 0,0,0 - 0,0,0. level 4, 1 node.
    addSplat(splats, 0.0f, 0.0f, 0.0f, 0.5f);
    SplatTree tree(splats, grid);

    const size_type expectedIds[] =
    {
        2, 2, 2, 2, 2, 2, 2, 2,
        1, 1, 1, 1,
        3, 4, 3, 0, 0, 0, 0, 0, 0, 0, 0
    };
    const struct
    {
        unsigned int level;
        unsigned int x, y, z;
        unsigned int count;
    } expectedCounts[] =
    {
        { 1,  0, 0, 0,  1 },
        { 1,  1, 0, 0,  1 },
        { 1,  0, 1, 0,  1 },
        { 1,  1, 1, 0,  1 },
        { 1,  0, 0, 1,  1 },
        { 1,  1, 0, 1,  1 },
        { 1,  0, 1, 1,  1 },
        { 1,  1, 1, 1,  1 },

        { 2,  1, 2, 0,  1 },
        { 2,  1, 2, 1,  1 },
        { 2,  2, 2, 0,  1 },
        { 2,  2, 2, 1,  1 },

        { 4,  0, 0, 0,  2 },
        { 4,  0, 1, 0,  1 },

        { 4,  7, 7, 7,  1 },
        { 4,  8, 7, 7,  1 },
        { 4,  7, 8, 7,  1 },
        { 4,  8, 8, 7,  1 },
        { 4,  7, 7, 8,  1 },
        { 4,  8, 7, 8,  1 },
        { 4,  7, 8, 8,  1 },
        { 4,  8, 8, 8,  1 }
    };

    // Validate levelStart
    CPPUNIT_ASSERT_EQUAL(size_t(6), tree.levelStart.size());
    CPPUNIT_ASSERT_EQUAL(size_type(0), tree.levelStart[0]);
    CPPUNIT_ASSERT_EQUAL(size_type(1), tree.levelStart[1]);
    CPPUNIT_ASSERT_EQUAL(size_type(9), tree.levelStart[2]);
    CPPUNIT_ASSERT_EQUAL(size_type(73), tree.levelStart[3]);
    CPPUNIT_ASSERT_EQUAL(size_type(585), tree.levelStart[4]);
    CPPUNIT_ASSERT_EQUAL(size_type(4681), tree.levelStart[5]);

    // Validate ids
    CPPUNIT_ASSERT_EQUAL(sizeof(expectedIds) / sizeof(expectedIds[0]), tree.ids.size());
    for (size_t i = 0; i < tree.ids.size(); i++)
        CPPUNIT_ASSERT_EQUAL(expectedIds[i], tree.ids[i]);

    // Validate start
    CPPUNIT_ASSERT_EQUAL(size_t(tree.levelStart.back()) + 1, tree.start.size());
    CPPUNIT_ASSERT_EQUAL(size_type(0), tree.start[0]);
    CPPUNIT_ASSERT_EQUAL(size_type(tree.ids.size()), tree.start.back());

    // Make sure it is non-decreasing
    for (size_t i = 0; i + 1 < tree.start.size(); i++)
        CPPUNIT_ASSERT(tree.start[i] <= tree.start[i + 1]);
    // Check the specific counts we are interested in. This forces the
    // others to be zero because we checked the grand total.

    for (size_t i = 0; i < sizeof(expectedCounts) / sizeof(expectedCounts[0]); i++)
    {
        size_type pos = tree.levelStart[expectedCounts[i].level]
            + SplatTree::makeCode(expectedCounts[i].x, expectedCounts[i].y, expectedCounts[i].z);
        CPPUNIT_ASSERT_EQUAL(expectedCounts[i].count, tree.start[pos + 1] - tree.start[pos]);
    }
}
