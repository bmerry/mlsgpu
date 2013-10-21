/*
 * mlsgpu: surface reconstruction from point clouds
 * Copyright (C) 2013  University of Cape Town
 *
 * This file is part of mlsgpu.
 *
 * mlsgpu is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

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
#include "testutil.h"
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
        const Grid::size_type size[3], const Grid::difference_type offset[3]) = 0;

public:
    void testMakeCode();         ///< Test @ref SplatTree::makeCode
    void testBuild();            ///< Test construction of internal state
    void testRandom();           ///< Build a larger tree and do sanity checks on output
};

#endif /* !TEST_SPLAT_TREE_H */
