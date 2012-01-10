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
    CPPUNIT_TEST(testGetDirection);
    CPPUNIT_TEST(testGetExtent);
    CPPUNIT_TEST(testNumCells);
    CPPUNIT_TEST(testNumVertices);
    CPPUNIT_TEST(testGetVertex);
    CPPUNIT_TEST(testWorldToVertex);
    CPPUNIT_TEST(testSubGrid);
    CPPUNIT_TEST_SUITE_END();
private:
    float ref[3];
    float dirs[3][3];
    std::pair<int, int> extents[3];
    Grid grid;

public:
    void setUp();

    void testGetReference();
    void testGetDirection();
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
    for (unsigned int i = 0; i < 3; i++)
        for (unsigned int j = 0; j < 3; j++)
            dirs[i][j] = 0.0f;
    dirs[0][0] = 2.75f;
    dirs[1][1] = 3.25f;
    dirs[2][2] = -1.5f;
    extents[0] = std::make_pair(-5, 30);
    extents[1] = std::make_pair(7, 25);
    extents[2] = std::make_pair(-1000, -2);
    grid = Grid(ref, dirs[0], dirs[1], dirs[2],
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

void TestGrid::testGetDirection()
{
    for (unsigned int i = 0; i < 3; i++)
    {
        const float *test = grid.getDirection(i);
        CPPUNIT_ASSERT_EQUAL(dirs[i][0], test[0]);
        CPPUNIT_ASSERT_EQUAL(dirs[i][1], test[1]);
        CPPUNIT_ASSERT_EQUAL(dirs[i][2], test[2]);
    }
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
    CPPUNIT_ASSERT_EQUAL(-12.25f, test[0]);
    CPPUNIT_ASSERT_EQUAL(19.75f, test[1]);
    CPPUNIT_ASSERT_EQUAL(1502.25f, test[2]);

    grid.getVertex(5, 7, 500, test);
    CPPUNIT_ASSERT_EQUAL(1.5f, test[0]);
    CPPUNIT_ASSERT_EQUAL(42.5f, test[1]);
    CPPUNIT_ASSERT_EQUAL(752.25f, test[2]);
}

void TestGrid::testWorldToVertex()
{
    float world[3];
    float test[3];

    world[0] = -12.25f; world[1] = 19.75f; world[2] = 1502.25f;
    grid.worldToVertex(world, test);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0, test[0], 1e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0, test[1], 1e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0, test[2], 1e-6);

    world[0] = 1.5f; world[1] = -3.0f; world[2] = 2.25f;
    grid.worldToVertex(world, test);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(5, test[0], 1e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-7, test[1], 1e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1000, test[2], 1e-3);

    world[0] = 1.5f; world[1] = 42.5f; world[2] = 752.25f;
    grid.worldToVertex(world, test);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(5, test[0], 1e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(7, test[1], 1e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(500, test[2], 1e-3);

    world[0] = 0.0f; world[1] = 0.0f; world[2] = 0.0f;
    grid.worldToVertex(world, test);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(49.0 / 11.0, test[0], 1e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-79.0 / 13.0, test[1], 1e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1001.5, test[2], 1e-6);
}

void TestGrid::testSubGrid()
{
    Grid g = grid.subGrid(3, 7, 10, 15, -5, -5);

    float test[3];
    g.getVertex(0, 0, 0, test);
    CPPUNIT_ASSERT_EQUAL(-12.25f + 3 * 2.75f, test[0]);
    CPPUNIT_ASSERT_EQUAL(19.75f + 10 * 3.25f, test[1]);
    CPPUNIT_ASSERT_EQUAL(1502.25f + -5 * -1.5f, test[2]);

    CPPUNIT_ASSERT_EQUAL(4, g.numCells(0));
    CPPUNIT_ASSERT_EQUAL(5, g.numCells(1));
    CPPUNIT_ASSERT_EQUAL(0, g.numCells(2));
}
