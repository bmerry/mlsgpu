#ifndef EXTRAS_NORMALS_SWEEP_H
#define EXTRAS_NORMALS_SWEEP_H

#include <boost/program_options.hpp>

void addSweepOptions(boost::program_options::options_description &opts);

void runSweep(const boost::program_options::variables_map &vm, bool continuous);

#endif /* !EXTRAS_NORMALS_SWEEP_H */
