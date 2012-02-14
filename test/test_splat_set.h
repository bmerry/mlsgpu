/**
 * @file
 *
 * Utility functions used by @ref TestSplatSet and other classes.
 */

#ifndef TEST_SPLAT_SET_H
#define TEST_SPLAT_SET_H

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <vector>
#include <boost/ptr_container/ptr_vector.hpp>
#include "../src/splat.h"
#include "../src/collection.h"

/**
 * Creates a sample set of splats for use in a test case. The
 * resulting set of splats is intended to be interesting when
 * used with a grid spacing of 2.5 and an origin at the origin.
 *
 * To make this easy to visualise, all splats are placed on a single Z plane.
 * This plane is along a major boundary, so when bucketing, each block can be
 * expected to appear twice (once on each side of the boundary).
 *
 * To see the splats graphically, save the following to a file
 * and run gnuplot over it. The coordinates are in grid space
 * rather than world space:
 * <pre>
 * set xrange [0:16]
 * set yrange [0:20]
 * set size square
 * set xtics 4
 * set ytics 4
 * set grid
 * plot '-' with points
 * 4 8
 * 12 6.8
 * 12.8 4.8
 * 12.8 7.2
 * 14.8 7.2
 * 14 6.4
 * 4.8 14.8
 * 5.2 14.8
 * 4.8 15.2
 * 5.2 15.2
 * 6.8 12.8
 * 7.2 13.2
 * 10 18
 * e
 * pause -1
 * </pre>
 */
void createSplats(std::vector<std::vector<Splat> > &splats);

/**
 * Creates the same splats as @ref createSplats, and also
 * sets up collections to reference them.
 */
void createSplats(std::vector<std::vector<Splat> > &splats,
                  boost::ptr_vector<StdVectorCollection<Splat> > &collections);

#endif /* !TEST_SPLAT_SET_H */
