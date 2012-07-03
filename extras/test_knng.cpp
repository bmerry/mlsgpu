/**
 * @file
 *
 * Tests for @ref extras/knng.h.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif

#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/extensions/HelperMacros.h>
#include <vector>
#include <limits>
#include <algorithm>
#include <Eigen/Core>
#include <boost/tr1/random.hpp>
#include <boost/bind.hpp>
#include <boost/smart_ptr/scoped_ptr.hpp>
#include "../test/testmain.h"
#include "knng.h"

typedef Eigen::Vector3f Point;

/**
 * Create an empty point list.
 */
static std::vector<Point> makePointsEmpty()
{
    return std::vector<Point>();
}

/**
 * Create a point list with one point.
 */
static std::vector<Point> makePointsSingle()
{
    std::vector<Point> points;
    points.push_back(Point(1.0f, -1.5f, 100.0f));
    return points;
}

/**
 * Create a point list with duplicate points.
 */
static std::vector<Point> makePointsDuplicates()
{
    std::vector<Point> points;
    for (int i = 0; i < 10; i++)
        points.push_back(Point(1.0f, -1.5f, 100.0f));
    for (int i = 0; i < 15; i++)
        points.push_back(Point(-1.5f, -2.5f, 50.0f));
    return points;
}

/**
 * Create a point list with lots of points along one axis.
 */
static std::vector<Point> makePointsAxis(int N)
{
    std::vector<Point> points;
    for (int i = 0; i < N; i++)
        points.push_back(Point(1.0f, -3.0f, i * 0.2f - 11.0f));
    return points;
}

/**
 * Create a point list with random points
 */
static std::vector<Point> makePointsRandom(int N)
{
    using namespace std::tr1;

    mt19937 engine;
    uniform_real<float> dist(-3.0f, 3.0f);
    variate_generator<mt19937 &, uniform_real<float> > gen(engine, dist);

    std::vector<Point> points;
    for (int i = 0; i < N; i++)
    {
        points.push_back(Point(gen(), gen(), gen()));
    }
    return points;
}

#define TEST_WITH_POINTS(testMethod, pointGen) \
    CPPUNIT_TEST_SUITE_ADD_TEST( (new GenericTestCaller<TestFixtureType>( \
        context.getTestNameFor(#testMethod "(" #pointGen ")"), \
        boost::bind(&TestFixtureType::testMethod, _1, boost::bind(pointGen)), \
        context.makeFixture() ) ) )

#define TEST_WITH_POINTS_N(testMethod, pointGen, N) \
    CPPUNIT_TEST_SUITE_ADD_TEST( (new GenericTestCaller<TestFixtureType>( \
        context.getTestNameFor(#testMethod "(" #pointGen "(" #N "))"), \
        boost::bind(&TestFixtureType::testMethod, _1, boost::bind(pointGen, N)), \
        context.makeFixture() ) ) )


/**
 * Tests the internal structure of a @ref KDTree.
 */
class TestKDTree : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestKDTree);
    TEST_WITH_POINTS(testConstruct, makePointsEmpty);
    TEST_WITH_POINTS(testConstruct, makePointsSingle);
    TEST_WITH_POINTS(testConstruct, makePointsDuplicates);
    TEST_WITH_POINTS_N(testConstruct, makePointsAxis, 100);
    TEST_WITH_POINTS_N(testConstruct, makePointsRandom, 5000);
    CPPUNIT_TEST_SUITE_END();

private:
    /// Does the actual test of the construction
    void testConstruct(const std::vector<Point> &points);
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestKDTree, TestSet::perBuild());

void TestKDTree::testConstruct(const std::vector<Point> &points)
{
    typedef unsigned int size_type;
    KDTree<float, 3, size_type> tree(points.begin(), points.end());

    size_type N = points.size();
    CPPUNIT_ASSERT_EQUAL(points.size(), tree.points.size());
    // Check that every point occurs exactly once
    std::vector<bool> seen(N);
    for (size_type i = 0; i < N; i++)
    {
        size_type id = tree.points[i].id;
        CPPUNIT_ASSERT(id < N);
        CPPUNIT_ASSERT(!seen[id]);
        CPPUNIT_ASSERT_EQUAL(points[id], tree.points[i].pos);
        seen[id] = true;
    }

    // Validate the leaves.
    size_type last = 0;
    for (size_type i = 0; i < tree.nodes.size(); i++)
    {
        const KDTree<float, 3, size_type>::KDNode &node = tree.nodes[i];
        if (node.isLeaf())
        {
            CPPUNIT_ASSERT_EQUAL(last, node.first());
            CPPUNIT_ASSERT(points.size() < tree.MIN_LEAF || node.last() >= last + tree.MIN_LEAF);
            CPPUNIT_ASSERT(node.last() < last + 2 * tree.MIN_LEAF);
            last = node.last();
            CPPUNIT_ASSERT(last <= N);

            for (int axis = 0; axis < 3; axis++)
            {
                float lo = std::numeric_limits<float>::infinity();
                float hi = -lo;
                for (size_type j = node.first(); j < node.last(); j++)
                {
                    lo = std::min(lo, tree.points[j].pos[axis]);
                    hi = std::max(hi, tree.points[j].pos[axis]);
                }
                CPPUNIT_ASSERT_EQUAL(lo, node.bbox[axis][0]);
                CPPUNIT_ASSERT_EQUAL(hi, node.bbox[axis][1]);
            }
        }
    }
    CPPUNIT_ASSERT(last == N);

    // Validate all the internal nodes
    std::vector<int> parent(tree.nodes.size(), -1);
    for (size_type i = 0; i < tree.nodes.size(); i++)
    {
        const KDTree<float, 3, size_type>::KDNode &node = tree.nodes[i];
        if (node.axis != -1)
        {
            CPPUNIT_ASSERT(node.axis >= 0 && node.axis < 3);
            CPPUNIT_ASSERT_EQUAL(i + 1, node.left());
            CPPUNIT_ASSERT(node.left() < tree.nodes.size());
            CPPUNIT_ASSERT(node.right() > node.left());
            CPPUNIT_ASSERT(node.right() < tree.nodes.size());
            for (int j = 0; j < 3; j++)
            {
                float lo = std::min(tree.nodes[node.left()].bbox[j][0],
                                    tree.nodes[node.right()].bbox[j][0]);
                float hi = std::max(tree.nodes[node.left()].bbox[j][1],
                                    tree.nodes[node.right()].bbox[j][1]);
                CPPUNIT_ASSERT_EQUAL(lo, node.bbox[j][0]);
                CPPUNIT_ASSERT_EQUAL(hi, node.bbox[j][1]);
            }
            CPPUNIT_ASSERT(tree.nodes[node.left()].bbox[node.axis][1] <= node.split());
            CPPUNIT_ASSERT(tree.nodes[node.right()].bbox[node.axis][0] >= node.split());

            CPPUNIT_ASSERT_EQUAL(-1, int(parent[node.left()]));
            CPPUNIT_ASSERT_EQUAL(-1, int(parent[node.right()]));
            parent[node.left()] = i;
            parent[node.right()] = i;
        }
    }

    if (!tree.nodes.empty())
    {
        CPPUNIT_ASSERT(std::count(parent.begin() + 1, parent.end(), -1) == 0);
        CPPUNIT_ASSERT_EQUAL(-1, parent[0]);
    }
}

/**
 * Test the nearest neighbour search
 */
class TestKNN : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestKNN);
    TEST_WITH_POINTS(testKNN, makePointsEmpty);
    TEST_WITH_POINTS(testKNN, makePointsSingle);
    TEST_WITH_POINTS(testKNN, makePointsDuplicates);
    TEST_WITH_POINTS_N(testKNN, makePointsAxis, 100);
    TEST_WITH_POINTS_N(testKNN, makePointsRandom, 10);
    TEST_WITH_POINTS_N(testKNN, makePointsRandom, 400);
    TEST_WITH_POINTS_N(testKNNRadius, makePointsRandom, 400);
    CPPUNIT_TEST_SUITE_END();

private:
    /**
     * Common function for @ref testKNN and @ref testKNNRadius.
     *
     * @return The number of points whose neighbourhood sets had K elements.
     */
    int testHelper(const std::vector<Point> &points, int K, float radius);

    /// Test the k-nearest neighbor search against brute-force
    void testKNN(const std::vector<Point> &points);

    /// Test the k-nearest neighbor search including a cutoff radius
    void testKNNRadius(const std::vector<Point> &points);
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestKNN, TestSet::perBuild());

int TestKNN::testHelper(const std::vector<Point> &points, int K, float radius)
{
    typedef unsigned int size_type;
    const float radiusSquared = radius * radius;
    KDTree<float, 3, size_type> kdtree(points.begin(), points.end());
    boost::scoped_ptr<KNNG<float, size_type> > knng(kdtree.knn(K, radiusSquared));

    size_type N = points.size();
    size_type full = 0;
    for (size_type i = 0; i < N; i++)
    {
        // Brute force search
        std::vector<std::pair<float, size_type> > expected;
        for (size_type j = 0; j < N; j++)
            if (i != j)
            {
                float ds = (points[i] - points[j]).squaredNorm();
                if (ds <= radiusSquared)
                {
                    std::pair<float, size_type> cur(ds, j);
                    std::vector<std::pair<float, size_type> >::iterator pos
                        = lower_bound(expected.begin(), expected.end(), cur);
                    expected.insert(pos, cur);
                    if (expected.size() > size_type(K))
                        expected.pop_back();
                }
            }

        std::vector<std::pair<float, size_type> > actual = (*knng)[i];
        std::sort(actual.begin(), actual.end());

        // Indices might not match due to ties, but distances must match
        CPPUNIT_ASSERT_EQUAL(expected.size(), actual.size());
        for (size_type j = 0; j < expected.size(); j++)
        {
            CPPUNIT_ASSERT_DOUBLES_EQUAL(expected[j].first, actual[j].first, 1e-9);
            float ds = (points[i] - points[actual[j].second]).squaredNorm();
            CPPUNIT_ASSERT_DOUBLES_EQUAL(actual[j].first, ds, 1e-9);
            CPPUNIT_ASSERT(actual[j].second != i);
        }

        if (expected.size() == size_type(K))
            full++;
    }
    return full;
}

void TestKNN::testKNN(const std::vector<Point> &points)
{
    testHelper(points, 8, std::numeric_limits<float>::infinity());
}

void TestKNN::testKNNRadius(const std::vector<Point> &points)
{
    int full = testHelper(points, 6, 1.5f);
    // Just to check that the test has proper coverage
    CPPUNIT_ASSERT(full != 0);
    CPPUNIT_ASSERT(full != int(points.size()));
}
