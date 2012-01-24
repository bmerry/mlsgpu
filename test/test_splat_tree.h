/**
 * @file
 *
 * Declaration of @ref TestSplatTree.
 */

#ifndef TEST_SPLAT_TREE_H
#define TEST_SPLAT_TREE_H

#if HAVE_CONFIG_H
# include <config.h>
#endif

#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/extensions/HelperMacros.h>
#include "testmain.h"
#include "../src/splat.h"
#include "../src/grid.h"
#include "../src/splat_tree.h"

/**
 * Tests for @ref SplatTree.
 */
class TestSplatTree : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestSplatTree);
    CPPUNIT_TEST(testMakeCode);
    CPPUNIT_TEST(testBuild);
    CPPUNIT_TEST(testRandom);
    CPPUNIT_TEST_SUITE_END_ABSTRACT();
protected:
    virtual void build(
        std::size_t &numLevels,
        std::vector<SplatTree::command_type> &commands,
        std::vector<SplatTree::command_type> &start,
        const std::vector<Splat> &splats,
        int maxLevels, int subsampling, std::size_t maxSplats,
        const Grid &grid) = 0;

public:
    void testMakeCode();         ///< Test @ref SplatTree::makeCode
    void testBuild();            ///< Test construction of internal state
    void testRandom();           ///< Build a larger tree and do sanity checks on output
};

#endif /* !TEST_SPLAT_TREE_H */
