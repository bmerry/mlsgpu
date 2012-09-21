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
#include <boost/numeric/conversion/converter.hpp>
#include "testutil.h"
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
    CPPUNIT_TEST(testWorldToCell);
    CPPUNIT_TEST(testWorldToCellOverflow);
    CPPUNIT_TEST(testSubGrid);
    CPPUNIT_TEST_SUITE_END();
private:
    float ref[3];
    float spacing;
    Grid::extent_type extents[3];
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
    void testWorldToCell();
    void testWorldToCellOverflow();
    void testSubGrid();
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestGrid, TestSet::perBuild());

void TestGrid::setUp()
{
    ref[0] = 1.5f;
    ref[1] = -3.0f;
    ref[2] = 2.25f;
    spacing = 3.0f;
    extents[0] = Grid::extent_type(-5, 30);
    extents[1] = Grid::extent_type(7, 25);
    extents[2] = Grid::extent_type(-1000, -2);
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
        const Grid::extent_type test = grid.getExtent(i);
        CPPUNIT_ASSERT_EQUAL(extents[i].first, test.first);
        CPPUNIT_ASSERT_EQUAL(extents[i].second, test.second);
    }
}

void TestGrid::testNumCells()
{
    CPPUNIT_ASSERT_EQUAL(Grid::size_type(35), grid.numCells(0));
    CPPUNIT_ASSERT_EQUAL(Grid::size_type(18), grid.numCells(1));
    CPPUNIT_ASSERT_EQUAL(Grid::size_type(998), grid.numCells(2));
    CPPUNIT_ASSERT_EQUAL(std::tr1::uint64_t(35 * 18 * 998), grid.numCells());

    // Test overflow handling
    Grid bigGrid = grid.subGrid(0, 1000000, 0, 1000000, 0, 10000000);
    CPPUNIT_ASSERT_EQUAL(UINT64_C(10000000000000000000), bigGrid.numCells());
}

void TestGrid::testNumVertices()
{
    CPPUNIT_ASSERT_EQUAL(Grid::size_type(36), grid.numVertices(0));
    CPPUNIT_ASSERT_EQUAL(Grid::size_type(19), grid.numVertices(1));
    CPPUNIT_ASSERT_EQUAL(Grid::size_type(999), grid.numVertices(2));
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

void TestGrid::testWorldToCell()
{
    float world[3];
    Grid::difference_type test[3];

    world[0] = -13.4f; world[1] = 17.9f; world[2] = -2998.0f;
    grid.worldToCell(world, test);
    CPPUNIT_ASSERT_EQUAL(Grid::difference_type(0), test[0]);
    CPPUNIT_ASSERT_EQUAL(Grid::difference_type(-1), test[1]);
    CPPUNIT_ASSERT_EQUAL(Grid::difference_type(-1), test[2]);

    world[0] = 0.0f; world[1] = 0.0f; world[2] = 0.0f;
    grid.worldToCell(world, test);
    CPPUNIT_ASSERT_EQUAL(Grid::difference_type(4), test[0]);
    CPPUNIT_ASSERT_EQUAL(Grid::difference_type(-6), test[1]); // corner case, may need to be changed
    CPPUNIT_ASSERT_EQUAL(Grid::difference_type(999), test[2]);
}

void TestGrid::testWorldToCellOverflow()
{
    float world[3];
    Grid::difference_type test[3];

    // NaN
    world[0] = 0.0f;
    world[1] = 0.0f;
    world[2] = std::numeric_limits<float>::quiet_NaN();
    CPPUNIT_ASSERT_THROW(grid.worldToCell(world, test), boost::numeric::bad_numeric_cast);

    // Infinity
    world[0] = 0.0f;
    world[1] = std::numeric_limits<float>::infinity();
    world[2] = 0.0f;
    CPPUNIT_ASSERT_THROW(grid.worldToCell(world, test), boost::numeric::bad_numeric_cast);

    // Overflow before biasing
    world[0] = grid.getSpacing() * 3e9f;
    CPPUNIT_ASSERT_THROW(grid.worldToCell(world, test), boost::numeric::bad_numeric_cast);

    // Positive overflow after biasing
    grid.setExtent(0, -2000000000, 0);
    world[0] = grid.getSpacing() * 1.5e9f;
    world[1] = 0.0f;
    world[2] = 0.0f;
    CPPUNIT_ASSERT_THROW(grid.worldToCell(world, test), boost::numeric::bad_numeric_cast);

    // Negative overflow after biasing
    grid.setExtent(0, 2000000000, 2000000002);
    world[0] = grid.getSpacing() * -1.5e9f;
    world[1] = 0.0f;
    world[2] = 0.0f;
    CPPUNIT_ASSERT_THROW(grid.worldToCell(world, test), boost::numeric::bad_numeric_cast);
}

void TestGrid::testSubGrid()
{
    Grid g = grid.subGrid(3, 7, 10, 15, -5, -5);

    float test[3];
    g.getVertex(0, 0, 0, test);
    CPPUNIT_ASSERT_EQUAL(-13.5f + 3 * 3.0f, test[0]);
    CPPUNIT_ASSERT_EQUAL(18.0f + 10 * 3.0f, test[1]);
    CPPUNIT_ASSERT_EQUAL(-2997.75f + -5 * 3.0f, test[2]);

    CPPUNIT_ASSERT_EQUAL(Grid::size_type(4), g.numCells(0));
    CPPUNIT_ASSERT_EQUAL(Grid::size_type(5), g.numCells(1));
    CPPUNIT_ASSERT_EQUAL(Grid::size_type(0), g.numCells(2));
}
