/**
 * @file
 *
 * Utility functions only used in the main program.
 */

#ifndef MLSGPU_CORE_H
#define MLSGPU_CORE_H

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <boost/program_options.hpp>
#include <ostream>
#include <exception>
#include "splat_set.h"
#include <CL/cl.hpp>

namespace CLH
{
    class ResourceUsage;
}

namespace Option
{
    const char * const help = "help";
    const char * const quiet = "quiet";
    const char * const debug = "debug";
    const char * const responseFile = "response-file";

    const char * const fitSmooth = "fit-smooth";
    const char * const maxRadius = "max-radius";
    const char * const fitGrid = "fit-grid";
    const char * const fitPrune = "fit-prune";
    const char * const fitKeepBoundary = "fit-keep-boundary";
    const char * const fitBoundaryLimit = "fit-boundary-limit";
    const char * const fitShape = "fit-shape";

    const char * const inputFile = "input-file";
    const char * const outputFile = "output-file";
    const char * const split = "split";
    const char * const splitSize = "split-size";

    const char * const statistics = "statistics";
    const char * const statisticsFile = "statistics-file";
    const char * const timeplot = "timeplot";

    const char * const maxHostSplats = "max-host-splats";
    const char * const maxDeviceSplats = "max-device-splats";
    const char * const maxSplit = "max-split";
    const char * const levels = "levels";
    const char * const subsampling = "subsampling";
    const char * const bucketThreads = "bucket-threads";
    const char * const deviceThreads = "device-threads";
    const char * const reader = "reader";
    const char * const writer = "writer";
    const char * const decache = "decache";
};

/**
 * Write usage information to an output stream.
 */
void usage(std::ostream &o, const boost::program_options::options_description desc);

/**
 * Process the argv array to produce command-line options.
 */
boost::program_options::variables_map processOptions(int argc, char **argv);

/**
 * Translate the command-line options back into the form they would be given
 * on the command line.
 */
std::string makeOptions(const boost::program_options::variables_map &vm);

/**
 * Write the statistics to the statistics output.
 *
 * @param vm    Indicates where the output should be sent.
 * @param force If true, write statistics even if --statistics was not given
 */
void writeStatistics(const boost::program_options::variables_map &vm, bool force = false);

/**
 * Check that command-line option values are valid and in range.
 * @param vm    Command-line options.
 *
 * @throw invalid_option if any of the options were invalid.
 */
void validateOptions(const boost::program_options::variables_map &vm);

/**
 * Set the logging level based on the command-line options.
 */
void setLogLevel(const boost::program_options::variables_map &vm);

/**
 * Estimate the per-device resource usage based on command-line options.
 */
CLH::ResourceUsage resourceUsage(const boost::program_options::variables_map &vm);

/**
 * Check that a CL device can safely be used.
 *
 * @param device      Device to check.
 * @param totalUsage  Resource usage for the device, as returned by @ref resourceUsage.
 * @throw CLH::invalid_device if the device is unusable.
 */
void validateDevice(const cl::Device &device, const CLH::ResourceUsage &totalUsage);

/**
 * Put the input files named in @a vm into @a files.
 *
 * @throw boost::exception   if there was a problem reading the files.
 * @throw std::runtime_error if there are too many files or splats.
 */
void prepareInputs(SplatSet::FileSet &files, const boost::program_options::variables_map &vm, float smooth, float maxRadius);

/**
 * Dump an error to stderr.
 */
void reportException(std::exception &e);

#endif /* !MLSGPU_CORE_H */
