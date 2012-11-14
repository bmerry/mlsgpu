/**
 * @file
 *
 * Test code for @ref UnionFind.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif

#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/extensions/HelperMacros.h>
#include <vector>
#include "../src/union_find.h"
#include "testutil.h"

/// Tests for @ref UnionFind
class TestUnionFind : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestUnionFind);
    CPPUNIT_TEST(testIsRoot);
    CPPUNIT_TEST(testMerge);
    CPPUNIT_TEST(testFind);
    CPPUNIT_TEST(testSize);
    CPPUNIT_TEST_SUITE_END();

private:
    class Node : public UnionFind::Node<int>
    {
    public:
        int x;     ///< Generic value that is summed over a component

        void merge(const Node &b)
        {
            UnionFind::Node<int>::merge(b);
            x += b.x;
        }

        Node() : x(0) {}
    };

    std::vector<Node> nodes;

public:
    virtual void setUp(); ///< Populates a union-find structure

    void testIsRoot();
    void testMerge();
    void testFind();
    void testSize();
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestUnionFind, TestSet::perBuild());

void TestUnionFind::setUp()
{
    nodes.clear();
    nodes.resize(9);
    for (int i = 0; i < 9; i++)
        nodes[i].x = i;
    UnionFind::merge(nodes, 0, 1);
    UnionFind::merge(nodes, 2, 3);
    UnionFind::merge(nodes, 4, 6);
    UnionFind::merge(nodes, 1, 3);
    UnionFind::merge(nodes, 6, 7);
    // Components are (0, 1, 2, 3), (4, 6, 7), (5), (8)
}

void TestUnionFind::testIsRoot()
{
    {
        int roots = nodes[0].isRoot() + nodes[1].isRoot() + nodes[2].isRoot() + nodes[3].isRoot();
        CPPUNIT_ASSERT_EQUAL(1, roots);
    }
    {
        int roots = nodes[4].isRoot() + nodes[6].isRoot() + nodes[7].isRoot();
        CPPUNIT_ASSERT_EQUAL(1, roots);
    }
    CPPUNIT_ASSERT(nodes[5].isRoot());
    CPPUNIT_ASSERT(nodes[8].isRoot());
}

void TestUnionFind::testMerge()
{
    // Merge two that are already unified
    bool merged = UnionFind::merge(nodes, 0, 2);
    CPPUNIT_ASSERT(!merged);

    // Merge two new ones
    merged = UnionFind::merge(nodes, 1, 7);
    CPPUNIT_ASSERT(merged);
    int root = UnionFind::findRoot(nodes, 1);
    CPPUNIT_ASSERT_EQUAL(root, UnionFind::findRoot(nodes, 7));
    CPPUNIT_ASSERT_EQUAL(7, nodes[root].size());
    CPPUNIT_ASSERT_EQUAL(23, nodes[root].x);
}

void TestUnionFind::testFind()
{
    int roots[4];
    roots[0] = UnionFind::findRoot(nodes, 0);
    roots[1] = UnionFind::findRoot(nodes, 4);
    roots[2] = UnionFind::findRoot(nodes, 5);
    roots[3] = UnionFind::findRoot(nodes, 8);
    for (int i = 0; i < 4; i++)
        for (int j = i + 1; j < 4; j++)
            CPPUNIT_ASSERT(roots[i] != roots[j]);
    CPPUNIT_ASSERT_EQUAL(roots[0], UnionFind::findRoot(nodes, 1));
    CPPUNIT_ASSERT_EQUAL(roots[0], UnionFind::findRoot(nodes, 2));
    CPPUNIT_ASSERT_EQUAL(roots[0], UnionFind::findRoot(nodes, 3));
    CPPUNIT_ASSERT_EQUAL(roots[1], UnionFind::findRoot(nodes, 6));
    CPPUNIT_ASSERT_EQUAL(roots[1], UnionFind::findRoot(nodes, 7));

    // Check that path compression works
    for (int i = 0; i < 9; i++)
    {
        CPPUNIT_ASSERT(nodes[i].isRoot() || nodes[nodes[i].parent()].isRoot());
    }
}

void TestUnionFind::testSize()
{
    int roots[4];
    roots[0] = UnionFind::findRoot(nodes, 0);
    roots[1] = UnionFind::findRoot(nodes, 4);
    roots[2] = UnionFind::findRoot(nodes, 5);
    roots[3] = UnionFind::findRoot(nodes, 8);
    CPPUNIT_ASSERT_EQUAL(4, nodes[roots[0]].size());
    CPPUNIT_ASSERT_EQUAL(3, nodes[roots[1]].size());
    CPPUNIT_ASSERT_EQUAL(1, nodes[roots[2]].size());
    CPPUNIT_ASSERT_EQUAL(1, nodes[roots[3]].size());
}
