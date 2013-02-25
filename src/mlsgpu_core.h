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
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/smart_ptr/scoped_ptr.hpp>
#include <boost/function.hpp>
#include <ostream>
#include <exception>
#include <vector>
#include <utility>
#include "splat_set.h"
#include "workers.h"
#include "bucket.h"
#include "bucket_loader.h"
#include "splat_set.h"
#include "grid.h"
#include "progress.h"
#include "timeplot.h"
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
    const char * const tmpDir = "tmp-dir";

    const char * const fitSmooth = "fit-smooth";
    const char * const maxRadius = "max-radius";
    const char * const fitGrid = "fit-grid";
    const char * const fitPrune = "fit-prune";
    const char * const fitBoundaryLimit = "fit-boundary-limit";
    const char * const fitShape = "fit-shape";

    const char * const inputFile = "input-file";
    const char * const outputFile = "output-file";
    const char * const split = "split";
    const char * const splitSize = "split-size";

    const char * const statistics = "statistics";
    const char * const statisticsFile = "statistics-file";
    const char * const statisticsCL = "statistics-cl";
    const char * const timeplot = "timeplot";

    const char * const maxSplit = "max-split";
    const char * const levels = "levels";
    const char * const subsampling = "subsampling";
    const char * const leafCells = "leaf-cells";
    const char * const deviceThreads = "device-threads";
    const char * const reader = "reader";
    const char * const writer = "writer";
    const char * const ompThreads = "omp-threads";
    const char * const decache = "decache";
    const char * const checkpoint = "checkpoint";
    const char * const resume = "resume";

    const char * const memLoadSplats = "mem-load-splats";
    const char * const memHostSplats = "mem-host-splats";
    const char * const memBucketSplats = "mem-bucket-splats";
    const char * const memMesh = "mem-mesh";
    const char * const memReorder = "mem-reorder";
    const char * const memGather = "mem-gather";
};

/**
 * Write usage information to an output stream.
 */
void usage(std::ostream &o, const boost::program_options::options_description desc);

/**
 * Process the argv array to produce command-line options.
 */
boost::program_options::variables_map processOptions(int argc, char **argv, bool isMPI);

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
 * @param isMPI Whether MPI-related options are expected.
 *
 * @throw invalid_option if any of the options were invalid.
 */
void validateOptions(const boost::program_options::variables_map &vm, bool isMPI);

/**
 * Set the logging level based on the command-line options.
 */
void setLogLevel(const boost::program_options::variables_map &vm);

/**
 * Maximum number of splats to load as a batch.
 */
std::size_t getMaxLoadSplats(const boost::program_options::variables_map &vm);

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

/**
 * Load the inputs and compute the blobs and chunk size.
 *
 * @param tworker          Worker to attribute time for bounding box calculation
 * @param vm               Command-line options
 * @param[out] splats      The input files (must be initially empty)
 * @param computeBlobs     Callback to do the low-level computation
 *
 * @throw boost::exception   if there was a problem reading the files.
 * @throw std::runtime_error if there are too many or too few files or splats.
 */
void doComputeBlobs(
    Timeplot::Worker &tworker,
    const boost::program_options::variables_map &vm,
    SplatSet::FileSet &splats,
    boost::function<void(float, unsigned int)> computeBlobs);

/**
 * Validate the grid size and compute the chunk size.
 * @param vm               Command-line options
 * @param grid             Bounding box grid
 * @return Chunk size for output, in cells
 * @throw std::runtime_error if the grid is too large
 */
unsigned int postprocessGrid(
    const boost::program_options::variables_map &vm,
    const Grid &grid);

/**
 * An all-in-one helper to call @ref Bucket::bucket with appropriate parameters.
 *
 * @param tworker          Worker to which the bucketing time is allocated
 * @param vm               Command-line options
 * @param splats           Splats to bucket
 * @param grid             Bounding box grid from @ref doComputeBlobs
 * @param chunkCells       Chunk side length from @ref postprocessGrid
 * @param collector        Bucket processor passed to @ref Bucket::bucket
 */
void doBucket(
    Timeplot::Worker &tworker,
    const boost::program_options::variables_map &vm,
    const SplatSet::FastBlobSet<SplatSet::FileSet> &splats,
    const Grid &grid,
    Grid::size_type chunkCells,
    BucketCollector &collector);

/**
 * Set comments on the writer showing provenance of the file.
 */
void setWriterComments(const boost::program_options::variables_map &vm, FastPly::Writer &writer);

/**
 * Set mesher options based on command-line options.
 */
void setMesherOptions(const boost::program_options::variables_map &vm, MesherBase &mesher);

/**
 * Generate a file name from command-line options.
 */
MesherBase::Namer getNamer(const boost::program_options::variables_map &vm, const std::string &out);

/**
 * Collects together the workers that run on the slave side in MPI, without
 * using any MPI-specific code.
 */
class SlaveWorkers
{
public:
    Timeplot::Worker &tworker;
    boost::ptr_vector<DeviceWorkerGroup> deviceWorkerGroups;
    boost::scoped_ptr<CopyGroup> copyGroup;
    boost::scoped_ptr<BucketLoader> loader;

    SlaveWorkers(
        Timeplot::Worker &tworker,
        const boost::program_options::variables_map &vm,
        const std::vector<std::pair<cl::Context, cl::Device> > &devices,
        const DeviceWorkerGroup::OutputGenerator &outputGenerator);

    void start(SplatSet::FileSet &splats, const Grid &grid, ProgressMeter *progress);

    void stop();
};

#endif /* !MLSGPU_CORE_H */
