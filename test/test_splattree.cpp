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
    // bbox: 0,0,0 - 0,0,0. level 4, 1 node.
    addSplat(splats, 0.0f, 0.0f, 0.0f, 0.5f);
    SplatTree tree(splats, grid);

    const struct
    {
        unsigned int level;
        unsigned int x, y, z;
        SplatTree::size_type id;
    } expectedTable[] =
    {
        { 1,  0, 0, 0,  2 },
        { 1,  1, 0, 0,  2 },
        { 1,  0, 1, 0,  2 },
        { 1,  1, 1, 0,  2 },
        { 1,  0, 0, 1,  2 },
        { 1,  1, 0, 1,  2 },
        { 1,  0, 1, 1,  2 },
        { 1,  1, 1, 1,  2 },

        { 2,  1, 2, 0,  1 },
        { 2,  1, 2, 1,  1 },
        { 2,  2, 2, 0,  1 },
        { 2,  2, 2, 1,  1 },

        { 4,  0, 0, 0,  3 },

        { 4,  7, 7, 7,  0 },
        { 4,  8, 7, 7,  0 },
        { 4,  7, 8, 7,  0 },
        { 4,  8, 8, 7,  0 },
        { 4,  7, 7, 8,  0 },
        { 4,  8, 7, 8,  0 },
        { 4,  7, 8, 8,  0 },
        { 4,  8, 8, 8,  0 }
    };

    CPPUNIT_ASSERT_EQUAL(size_t(5), tree.levels.size());
    CPPUNIT_ASSERT_EQUAL(sizeof(expectedTable) / sizeof(expectedTable[0]), tree.ids.size());
    for (size_t i = 0; i < tree.ids.size(); i++)
        CPPUNIT_ASSERT_EQUAL(expectedTable[i].id, tree.ids[i]);

    // Check that the start values are monotonically non-decreasing
    CPPUNIT_ASSERT_EQUAL(SplatTree::size_type(0), tree.levels[0].start[0]);
    for (unsigned int i = 0; i < tree.levels.size(); i++)
    {
        CPPUNIT_ASSERT_EQUAL(size_t(1 << (3 * i)), tree.levels[i].start.size());
        for (unsigned int j = 0; j + 1 < tree.levels[i].start.size(); j++)
            CPPUNIT_ASSERT(tree.levels[i].start[j] <= tree.levels[i].start[j + 1]);
        if (i + 1 < tree.levels.size())
            CPPUNIT_ASSERT(tree.levels[i].start.back() <= tree.levels[i + 1].start[0]);
    }

    // Check the start values where there is data
    for (SplatTree::size_type i = 0; i < tree.ids.size(); i++)
    {
        SplatTree::size_type code = SplatTree::makeCode(expectedTable[i].x, expectedTable[i].y, expectedTable[i].z);
        CPPUNIT_ASSERT_EQUAL(i, tree.levels[expectedTable[i].level].start[code]);
    }

    // TODO: check the start values for those entries which have no nodes
}
