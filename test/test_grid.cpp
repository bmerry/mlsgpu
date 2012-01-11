/**
 * @file
 *
 * Test code for @ref Grid.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif

#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/extensions/HelperMacros.h>
#include <utility>
#include "testmain.h"
#include "../src/grid.h"

using namespace std;

class TestGrid : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestGrid);
    CPPUNIT_TEST(testGetReference);
    CPPUNIT_TEST(testGetSpacing);
    CPPUNIT_TEST(testGetExtent);
    CPPUNIT_TEST(testNumCells);
    CPPUNIT_TEST(testNumVertices);
    CPPUNIT_TEST(testGetVertex);
    CPPUNIT_TEST(testWorldToVertex);
    CPPUNIT_TEST(testSubGrid);
    CPPUNIT_TEST_SUITE_END();
private:
    float ref[3];
    float spacing;
    std::pair<int, int> extents[3];
    Grid grid;

public:
    void setUp();

    void testGetReference();
    void testGetSpacing();
    void testGetExtent();
    void testNumCells();
    void testNumVertices();
    void testGetVertex();
    void testWorldToVertex();
    void testSubGrid();
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestGrid, TestSet::perBuild());

void TestGrid::setUp()
{
    ref[0] = 1.5f;
    ref[1] = -3.0f;
    ref[2] = 2.25f;
    spacing = 3.0f;
    extents[0] = std::make_pair(-5, 30);
    extents[1] = std::make_pair(7, 25);
    extents[2] = std::make_pair(-1000, -2);
    grid = Grid(ref, spacing,
                extents[0].first, extents[0].second,
                extents[1].first, extents[1].second,
                extents[2].first, extents[2].second);
}

void TestGrid::testGetReference()
{
    const float *test = grid.getReference();
    CPPUNIT_ASSERT_EQUAL(ref[0], test[0]);
    CPPUNIT_ASSERT_EQUAL(ref[1], test[1]);
    CPPUNIT_ASSERT_EQUAL(ref[2], test[2]);
}

void TestGrid::testGetSpacing()
{
    const float s = grid.getSpacing();
    CPPUNIT_ASSERT_EQUAL(spacing, s);
}

void TestGrid::testGetExtent()
{
    for (unsigned int i = 0; i < 3; i++)
    {
        const std::pair<int, int> test = grid.getExtent(i);
        CPPUNIT_ASSERT_EQUAL(extents[i].first, test.first);
        CPPUNIT_ASSERT_EQUAL(extents[i].second, test.second);
    }
}

void TestGrid::testNumCells()
{
    CPPUNIT_ASSERT_EQUAL(35, grid.numCells(0));
    CPPUNIT_ASSERT_EQUAL(18, grid.numCells(1));
    CPPUNIT_ASSERT_EQUAL(998, grid.numCells(2));
}

void TestGrid::testNumVertices()
{
    CPPUNIT_ASSERT_EQUAL(36, grid.numVertices(0));
    CPPUNIT_ASSERT_EQUAL(19, grid.numVertices(1));
    CPPUNIT_ASSERT_EQUAL(999, grid.numVertices(2));
}

void TestGrid::testGetVertex()
{
    float test[3];
    grid.getVertex(0, 0, 0, test);
    CPPUNIT_ASSERT_EQUAL(-13.5f, test[0]);
    CPPUNIT_ASSERT_EQUAL(18.0f, test[1]);
    CPPUNIT_ASSERT_EQUAL(-2997.75f, test[2]);

    grid.getVertex(5, 7, 500, test);
    CPPUNIT_ASSERT_EQUAL(1.5f, test[0]);
    CPPUNIT_ASSERT_EQUAL(39.0f, test[1]);
    CPPUNIT_ASSERT_EQUAL(-1497.75f, test[2]);
}

void TestGrid::testWorldToVertex()
{
    float world[3];
    float test[3];

    world[0] = -13.5f; world[1] = 18.0f; world[2] = -2997.75f;
    grid.worldToVertex(world, test);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0, test[0], 1e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0, test[1], 1e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0, test[2], 1e-6);

    world[0] = 1.5f; world[1] = 39.0f; world[2] = -1497.75f;
    grid.worldToVertex(world, test);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(5, test[0], 1e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(7, test[1], 1e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(500, test[2], 1e-3);
}

void TestGrid::testSubGrid()
{
    Grid g = grid.subGrid(3, 7, 10, 15, -5, -5);

    float test[3];
    g.getVertex(0, 0, 0, test);
    CPPUNIT_ASSERT_EQUAL(-13.5f + 3 * 3.0f, test[0]);
    CPPUNIT_ASSERT_EQUAL(18.0f + 10 * 3.0f, test[1]);
    CPPUNIT_ASSERT_EQUAL(-2997.75f + -5 * 3.0f, test[2]);

    CPPUNIT_ASSERT_EQUAL(4, g.numCells(0));
    CPPUNIT_ASSERT_EQUAL(5, g.numCells(1));
    CPPUNIT_ASSERT_EQUAL(0, g.numCells(2));
}
