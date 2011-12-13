/**
 * @file
 *
 * Common definitions for tests.
 */

#ifndef TESTMAIN_H
#define TESTMAIN_H

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <string>
#include <boost/program_options.hpp>

const boost::program_options::variables_map &testGetOptions();

namespace TestSet
{

std::string perBuild();
std::string perCommit();
std::string perNightly();

};

#endif /* !TESTMAIN_H */
