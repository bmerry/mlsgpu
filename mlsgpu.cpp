/**
 * @file
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif

#ifndef __CL_ENABLE_EXCEPTIONS
# define __CL_ENABLE_EXCEPTIONS
#endif

#include <boost/program_options.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/foreach.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/smart_ptr/scoped_ptr.hpp>
#include <boost/smart_ptr/shared_ptr.hpp>
#include <boost/smart_ptr/make_shared.hpp>
#include <boost/thread/thread.hpp>
#include <boost/array.hpp>
#include <boost/progress.hpp>
#include <boost/io/ios_state.hpp>
#include <boost/exception/all.hpp>
#include <boost/system/error_code.hpp>
#include "src/tr1_unordered_map.h"
#include <iostream>
#include <map>
#include <vector>
#include <algorithm>
#include <stxxl.h>
#include "src/misc.h"
#include "src/clh.h"
#include "src/logging.h"
#include "src/timer.h"
#include "src/fast_ply.h"
#include "src/splat.h"
#include "src/grid.h"
#include "src/splat_tree_cl.h"
#include "src/marching.h"
#include "src/mls.h"
#include "src/mesher.h"
#include "src/options.h"
#include "src/splat_set.h"
#include "src/bucket.h"
#include "src/provenance.h"
#include "src/statistics.h"
#include "src/statistics_cl.h"
#include "src/work_queue.h"
#include "src/workers.h"
#include "src/progress.h"
#include "src/clip.h"
#include "src/mesh_filter.h"
#include "src/timeplot.h"

namespace po = boost::program_options;
using namespace std;

namespace Option
{
    const char * const help = "help";
    const char * const quiet = "quiet";
    const char * const debug = "debug";
    const char * const responseFile = "response-file";

    const char * const fitSmooth = "fit-smooth";
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

    const char * const maxHostSplats = "max-host-splats";
    const char * const maxDeviceSplats = "max-device-splats";
    const char * const maxSplit = "max-split";
    const char * const levels = "levels";
    const char * const subsampling = "subsampling";
    const char * const bucketThreads = "bucket-threads";
    const char * const deviceThreads = "device-threads";
    const char * const reader = "reader";
    const char * const writer = "writer";
};

static void addCommonOptions(po::options_description &opts)
{
    opts.add_options()
        ("help,h",                "Show help")
        ("quiet,q",               "Do not show informational messages")
        (Option::debug,           "Show debug messages")
        (Option::responseFile,    po::value<string>(), "Read options from file");
}

static void addFitOptions(po::options_description &opts)
{
    opts.add_options()
        (Option::fitSmooth,       po::value<double>()->default_value(4.0),  "Smoothing factor")
        (Option::fitGrid,         po::value<double>()->default_value(0.01), "Spacing of grid cells")
        (Option::fitPrune,        po::value<double>()->default_value(0.02), "Minimum fraction of vertices per component")
        (Option::fitKeepBoundary,                                           "Do not remove boundaries")
        (Option::fitBoundaryLimit, po::value<double>()->default_value(1.5), "Tuning factor for boundary detection")
        (Option::fitShape,        po::value<Choice<MlsShapeWrapper> >()->default_value(MLS_SHAPE_SPHERE),
                                                                            "Model shape (sphere | plane)");
}

static void addStatisticsOptions(po::options_description &opts)
{
    po::options_description statistics("Statistics options");
    statistics.add_options()
        (Option::statistics,                          "Print information about internal statistics")
        (Option::statisticsFile, po::value<string>(), "Direct statistics to file instead of stdout (implies --statistics)");
    opts.add(statistics);
}

static void addAdvancedOptions(po::options_description &opts)
{
    po::options_description advanced("Advanced options");
    advanced.add_options()
        (Option::levels,       po::value<int>()->default_value(6), "Levels in octree")
        (Option::subsampling,  po::value<int>()->default_value(3), "Subsampling of octree")
        (Option::maxDeviceSplats, po::value<int>()->default_value(1000000), "Maximum splats per block on the device")
        (Option::maxHostSplats, po::value<std::size_t>()->default_value(8000000), "Maximum splats per block on the CPU")
        (Option::maxSplit,     po::value<int>()->default_value(2097152), "Maximum fan-out in partitioning")
        (Option::bucketThreads, po::value<int>()->default_value(4), "Number of threads for bucketing splats")
        (Option::deviceThreads, po::value<int>()->default_value(1), "Number of threads per device for submitting OpenCL work")
        (Option::reader,       po::value<Choice<FastPly::ReaderTypeWrapper> >()->default_value(FastPly::SYSCALL_READER), "File reader class (mmap | syscall)")
        (Option::writer,       po::value<Choice<FastPly::WriterTypeWrapper> >()->default_value(FastPly::STREAM_WRITER), "File writer class (mmap | stream)");
    opts.add(advanced);
}

string makeOptions(const po::variables_map &vm)
{
    ostringstream opts;
    for (po::variables_map::const_iterator i = vm.begin(); i != vm.end(); ++i)
    {
        if (i->first == Option::inputFile)
            continue; // these are not output because some programs choke
        if (i->first == Option::responseFile)
            continue; // this is not relevant to reproducing the results
        const po::variable_value &param = i->second;
        const boost::any &value = param.value();
        if (param.empty()
            || (value.type() == typeid(string) && param.as<string>().empty()))
            opts << " --" << i->first;
        else if (value.type() == typeid(vector<string>))
        {
            BOOST_FOREACH(const string &j, param.as<vector<string> >())
            {
                opts << " --" << i->first << '=' << j;
            }
        }
        else
        {
            opts << " --" << i->first << '=';
            if (value.type() == typeid(string))
                opts << param.as<string>();
            else if (value.type() == typeid(double))
                opts << param.as<double>();
            else if (value.type() == typeid(int))
                opts << param.as<int>();
            else if (value.type() == typeid(unsigned int))
                opts << param.as<unsigned int>();
            else if (value.type() == typeid(std::size_t))
                opts << param.as<std::size_t>();
            else if (value.type() == typeid(Choice<MesherTypeWrapper>))
                opts << param.as<Choice<MesherTypeWrapper> >();
            else if (value.type() == typeid(Choice<FastPly::WriterTypeWrapper>))
                opts << param.as<Choice<FastPly::WriterTypeWrapper> >();
            else if (value.type() == typeid(Choice<FastPly::ReaderTypeWrapper>))
                opts << param.as<Choice<FastPly::ReaderTypeWrapper> >();
            else if (value.type() == typeid(Choice<MlsShapeWrapper>))
                opts << param.as<Choice<MlsShapeWrapper> >();
            else
                assert(!"Unhandled parameter type");
        }
    }
    return opts.str();
}

void writeStatistics(const boost::program_options::variables_map &vm, bool force = false)
{
    if (force || vm.count(Option::statistics) || vm.count(Option::statisticsFile))
    {
        ostream *out;
        ofstream outf;
        if (vm.count(Option::statisticsFile))
        {
            const string &name = vm[Option::statisticsFile].as<string>();
            outf.open(name.c_str());
            out = &outf;
        }
        else
        {
            out = &std::cout;
        }

        boost::io::ios_exception_saver saver(*out);
        out->exceptions(ios::failbit | ios::badbit);
        *out << "mlsgpu version: " << provenanceVersion() << '\n';
        *out << "mlsgpu variant: " << provenanceVariant() << '\n';
        *out << "mlsgpu options:" << makeOptions(vm) << '\n';
        {
            boost::io::ios_precision_saver saver2(*out);
            out->precision(15);
            *out << Statistics::Registry::getInstance();
        }
        *out << *stxxl::stats::get_instance();
    }
}

static void usage(ostream &o, const po::options_description desc)
{
    o << "Usage: mlsgpu [options] -o output.ply input.ply [input.ply...]\n\n";
    o << desc;
}

static po::variables_map processOptions(int argc, char **argv)
{
    po::positional_options_description positional;
    positional.add(Option::inputFile, -1);

    po::options_description desc("General options");
    addCommonOptions(desc);
    addFitOptions(desc);
    addStatisticsOptions(desc);
    addAdvancedOptions(desc);
    desc.add_options()
        ("output-file,o",   po::value<string>()->required(), "output file")
        (Option::split,     "split output across multiple files")
        (Option::splitSize, po::value<unsigned int>()->default_value(100), "approximate size of output chunks (MiB)");

    po::options_description clopts("OpenCL options");
    CLH::addOptions(clopts);
    desc.add(clopts);

    po::options_description hidden("Hidden options");
    hidden.add_options()
        (Option::inputFile, po::value<vector<string> >()->composing(), "input files");

    po::options_description all("All options");
    all.add(desc);
    all.add(hidden);

    try
    {
        po::variables_map vm;
        po::store(po::command_line_parser(argc, argv)
                  .style(po::command_line_style::default_style & ~po::command_line_style::allow_guessing)
                  .options(all)
                  .positional(positional)
                  .run(), vm);
        if (vm.count(Option::responseFile))
        {
            const string &fname = vm[Option::responseFile].as<string>();
            ifstream in(fname.c_str());
            if (!in)
            {
                Log::log[Log::warn] << "Could not open `" << fname << "', ignoring\n";
            }
            else
            {
                vector<string> args;
                copy(istream_iterator<string>(in), istream_iterator<string>(), back_inserter(args));
                if (in.bad())
                {
                    Log::log[Log::warn] << "Error while reading from `" << fname << "'\n";
                }
                in.close();
                po::store(po::command_line_parser(args)
                          .style(po::command_line_style::default_style & ~po::command_line_style::allow_guessing)
                          .options(all)
                          .positional(positional)
                          .run(), vm);
            }
        }

        po::notify(vm);

        if (vm.count(Option::help))
        {
            usage(cout, desc);
            exit(0);
        }
        /* Using ->required() on the option gives an unhelpful message */
        if (!vm.count(Option::inputFile))
        {
            cerr << "At least one input file must be specified.\n\n";
            usage(cerr, desc);
            exit(1);
        }

        return vm;
    }
    catch (po::error &e)
    {
        cerr << e.what() << "\n\n";
        usage(cerr, desc);
        exit(1);
    }
}

static void prepareInputs(SplatSet::FileSet &files, const po::variables_map &vm, float smooth)
{
    const vector<string> &names = vm[Option::inputFile].as<vector<string> >();
    const FastPly::ReaderType readerType = vm[Option::reader].as<Choice<FastPly::ReaderTypeWrapper> >();
    if (names.size() > SplatSet::FileSet::maxFiles)
    {
        cerr << "Too many input files (" << names.size() << " > " << SplatSet::FileSet::maxFiles << ")\n";
        exit(1);
    }
    BOOST_FOREACH(const string &name, names)
    {
        std::auto_ptr<FastPly::ReaderBase> reader(FastPly::createReader(readerType, name, smooth));
        if (reader->size() > SplatSet::FileSet::maxFileSplats)
        {
            cerr << "Too many samples in " << name << " ("
                << reader->size() << " > " << SplatSet::FileSet::maxFileSplats << ")\n";
            exit(1);
        }
        files.addFile(reader.get());
        reader.release();
    }
}

/**
 * Handles coarse-level bucketing from external storage. Unlike @ref
 * DeviceWorkerGroupBase::Worker and @ref FineBucketGroupBase::Worker, there
 * is only expected to be one of these, and it does not run in a separate
 * thread. It produces coarse buckets, read the splats into memory and pushes
 * the results to a @ref FineBucketGroup.
 */
template<typename Splats>
class HostBlock : public boost::noncopyable
{
public:
    void operator()(
        const typename SplatSet::Traits<Splats>::subset_type &splatSet,
        const Grid &grid,
        const Bucket::Recursion &recursionState);

    explicit HostBlock(FineBucketGroup &outGroup);

    /// Prepares for a pass
    void start(const Grid &fullGrid);

    /// Ends a pass
    void stop();
private:
    ChunkId curChunkId;
    FineBucketGroup &outGroup;
    Grid fullGrid;
    Timeplot::Worker tworker;
};

template<typename Splats>
HostBlock<Splats>::HostBlock(FineBucketGroup &outGroup)
: outGroup(outGroup), tworker("bucket.coarse")
{
}

template<typename Splats>
void HostBlock<Splats>::operator()(
    const typename SplatSet::Traits<Splats>::subset_type &splats,
    const Grid &grid, const Bucket::Recursion &recursionState)
{
    if (recursionState.chunk != curChunkId.coords)
    {
        curChunkId.gen++;
        curChunkId.coords = recursionState.chunk;
    }

    Statistics::Registry &registry = Statistics::Registry::getInstance();

    boost::shared_ptr<FineBucketGroup::WorkItem> item = outGroup.get(tworker);
    item->chunkId = curChunkId;
    item->grid = grid;
    item->recursionState = recursionState;
    item->splats.clear();
    float invSpacing = 1.0f / fullGrid.getSpacing();

    {
        Timeplot::Action timer("load", tworker, "host.block.load");
        assert(splats.numSplats() <= item->splats.capacity());

        boost::scoped_ptr<SplatSet::SplatStream> splatStream(splats.makeSplatStream());
        while (!splatStream->empty())
        {
            Splat splat = **splatStream;
            /* Transform the splats into the grid's coordinate system */
            fullGrid.worldToVertex(splat.position, splat.position);
            splat.radius *= invSpacing;
            item->splats.push_back(splat);
            ++*splatStream;
        }

        registry.getStatistic<Statistics::Variable>("host.block.splats").add(splats.numSplats());
        registry.getStatistic<Statistics::Variable>("host.block.ranges").add(splats.numRanges());
        registry.getStatistic<Statistics::Variable>("host.block.size").add
            (double(grid.numCells(0)) * grid.numCells(1) * grid.numCells(2));
    }

    outGroup.push(item, tworker);
}

template<typename Splats>
void HostBlock<Splats>::start(const Grid &fullGrid)
{
    this->fullGrid = fullGrid;
}

template<typename Splats>
void HostBlock<Splats>::stop()
{
}

/**
 * Main execution.
 *
 * @param devices         List of OpenCL devices to use
 * @param out             Output filename or basename
 * @param vm              Command-line options
 */
static void run(const std::vector<std::pair<cl::Context, cl::Device> > &devices,
                 const string &out,
                 const po::variables_map &vm)
{
    typedef SplatSet::FastBlobSet<SplatSet::FileSet, stxxl::VECTOR_GENERATOR<SplatSet::BlobData>::result > Splats;

    const float spacing = vm[Option::fitGrid].as<double>();
    const float smooth = vm[Option::fitSmooth].as<double>();
    const int subsampling = vm[Option::subsampling].as<int>();
    const int levels = vm[Option::levels].as<int>();
    const FastPly::WriterType writerType = vm[Option::writer].as<Choice<FastPly::WriterTypeWrapper> >();
    const MesherType mesherType = STXXL_MESHER;
    const std::size_t maxDeviceSplats = vm[Option::maxDeviceSplats].as<int>();
    const std::size_t maxHostSplats = vm[Option::maxHostSplats].as<std::size_t>();
    const std::size_t maxSplit = vm[Option::maxSplit].as<int>();
    const double pruneThreshold = vm[Option::fitPrune].as<double>();
    const bool keepBoundary = vm.count(Option::fitKeepBoundary);
    const float boundaryLimit = vm[Option::fitBoundaryLimit].as<double>();
    const MlsShape shape = vm[Option::fitShape].as<Choice<MlsShapeWrapper> >();
    const bool split = vm.count(Option::split);
    const unsigned int splitSize = vm[Option::splitSize].as<unsigned int>();

    const unsigned int block = 1U << (levels + subsampling - 1);
    const unsigned int blockCells = block - 1;

    const unsigned int numBucketThreads = vm[Option::bucketThreads].as<int>();
    const unsigned int numDeviceThreads = vm[Option::deviceThreads].as<int>();

    {
        Statistics::Timer grandTotalTimer("run.time");

        MesherBase::Namer namer;
        if (split)
            namer = ChunkNamer(out);
        else
            namer = TrivialNamer(out);

        boost::scoped_ptr<FastPly::WriterBase> writer(FastPly::createWriter(writerType));
        writer->addComment("mlsgpu version: " + provenanceVersion());
        writer->addComment("mlsgpu variant: " + provenanceVariant());
        writer->addComment("mlsgpu options:" + makeOptions(vm));
        boost::scoped_ptr<MesherBase> mesher(createMesher(mesherType, *writer, namer));
        mesher->setPruneThreshold(pruneThreshold);

        Log::log[Log::info] << "Initializing...\n";
        MesherGroup mesherGroup(devices.size() * numDeviceThreads);
        DeviceWorkerGroup deviceWorkerGroup(
            numDeviceThreads, numBucketThreads, mesherGroup,
            devices, maxDeviceSplats, blockCells, levels, subsampling,
            keepBoundary, boundaryLimit, shape);
        FineBucketGroup fineBucketGroup(
            numBucketThreads, 1, deviceWorkerGroup,
            maxHostSplats, maxDeviceSplats, blockCells, maxSplit);
        HostBlock<Splats> hostBlock(fineBucketGroup);

        Splats splats;
        prepareInputs(splats, vm, smooth);
        try
        {
            Statistics::Timer timer("bbox.time");
            splats.computeBlobs(spacing, blockCells, &Log::log[Log::info]);
            Log::log[Log::debug] << "Bbox time: " << timer.getElapsed() << std::endl;
        }
        catch (std::length_error &e) // TODO: should be a subclass of runtime_error
        {
            cerr << "At least one input point is required.\n";
            exit(1);
        }
        Grid grid = splats.getBoundingGrid();
        for (unsigned int i = 0; i < 3; i++)
            if (grid.numVertices(i) > Marching::MAX_GLOBAL_DIMENSION)
            {
                cerr << "The bounding box is too big (" << grid.numVertices(i) << " grid units).\n"
                    << "Perhaps you have used the wrong units for --fit-grid?\n";
                exit(1);
            }

        unsigned int chunkCells = 0;
        if (split)
        {
            /* Determine a chunk size from splitSize. We assume that a chunk will be
             * sliced by an axis-aligned plane. This plane will cut each vertical and
             * each diagonal edge ones, thus generating 2x^2 vertices. We then
             * apply a fudge factor of 10 to account for the fact that the real
             * world is not a simple plane, and will have walls, noise, etc, giving
             * 20x^2 vertices.
             *
             * A manifold with genus 0 has two triangles per vertex; vertices take
             * 12 bytes (3 floats) and triangles take 13 (count plus 3 uints in
             * PLY), giving 38 bytes per vertex. So there are 760x^2 bytes.
             */
            chunkCells = (unsigned int) ceil(sqrt((1024.0 * 1024.0 / 760.0) * splitSize));
            if (chunkCells == 0) chunkCells = 1;
        }

        for (unsigned int pass = 0; pass < mesher->numPasses(); pass++)
        {
            Log::log[Log::info] << "\nPass " << pass + 1 << "/" << mesher->numPasses() << endl;
            ostringstream passName;
            passName << "pass" << pass + 1 << ".time";
            Statistics::Timer timer(passName.str());

            ProgressDisplay progress(grid.numCells(), Log::log[Log::info]);

            mesherGroup.setInputFunctor(mesher->functor(pass));
            deviceWorkerGroup.setProgress(&progress);
            fineBucketGroup.setProgress(&progress);

            // Start threads
            hostBlock.start(grid);
            fineBucketGroup.start(grid);
            deviceWorkerGroup.start(grid);
            mesherGroup.start();

            try
            {
                Statistics::Timer bucketTimer("host.block.exec");
                Bucket::bucket(splats, grid, maxHostSplats, blockCells, chunkCells, true, maxSplit,
                               boost::ref(hostBlock), &progress);
            }
            catch (...)
            {
                // This can't be handled using unwinding, because that would operate in
                // the wrong order
                hostBlock.stop();
                fineBucketGroup.stop();
                deviceWorkerGroup.stop();
                mesherGroup.stop();
                throw;
            }

            /* Shut down threads. Note that it has to be done in forward order to
             * satisfy the requirement that stop() is only called after producers
             * are terminated.
             */
            hostBlock.stop();
            fineBucketGroup.stop();
            deviceWorkerGroup.stop();
            mesherGroup.stop();
        }

        {
            Statistics::Timer timer("finalize.time");
            mesher->write(&Log::log[Log::info]);
        }
    } // ends scope for grandTotalTimer

    Statistics::finalizeEventTimes();
    writeStatistics(vm);
}

static void validateOptions(const po::variables_map &vm)
{
    const int levels = vm[Option::levels].as<int>();
    const int subsampling = vm[Option::subsampling].as<int>();
    const std::size_t maxDeviceSplats = vm[Option::maxDeviceSplats].as<int>();
    const std::size_t maxHostSplats = vm[Option::maxHostSplats].as<std::size_t>();
    const std::size_t maxSplit = vm[Option::maxSplit].as<int>();
    const int bucketThreads = vm[Option::bucketThreads].as<int>();
    const int deviceThreads = vm[Option::deviceThreads].as<int>();
    const double pruneThreshold = vm[Option::fitPrune].as<double>();

    int maxLevels = std::min(
            std::size_t(Marching::MAX_DIMENSION_LOG2 + 1),
            std::size_t(SplatTreeCL::MAX_LEVELS));
    if (levels < 1 || levels > maxLevels)
    {
        cerr << "Value of --levels must be in the range 1 to " << maxLevels << ".\n";
        exit(1);
    }
    if (subsampling < MlsFunctor::subsamplingMin)
    {
        cerr << "Value of --subsampling must be at least " << MlsFunctor::subsamplingMin << ".\n";
        exit(1);
    }
    if (maxDeviceSplats < 1)
    {
        cerr << "Value of --max-device-splats must be positive.\n";
        exit(1);
    }
    if (maxHostSplats < maxDeviceSplats)
    {
        cerr << "Value of --max-host-splats must be at least that of --max-device-splats.\n";
        exit(1);
    }
    if (maxSplit < 8)
    {
        cerr << "Value of --max-split must be at least 8.\n";
        exit(1);
    }
    if (subsampling > Marching::MAX_DIMENSION_LOG2 + 1 - levels)
    {
        cerr << "Sum of --subsampling and --levels is too large.\n";
        exit(1);
    }
    const std::size_t treeVerts = std::size_t(1) << (subsampling + levels - 1);
    if (treeVerts < MlsFunctor::wgs[0] || treeVerts < MlsFunctor::wgs[1])
    {
        cerr << "Sum of --subsampling and --levels is too small.\n";
        exit(1);
    }

    if (bucketThreads < 1)
    {
        cerr << "Value of --bucket-threads must be at least 1\n";
        exit(1);
    }
    if (deviceThreads < 1)
    {
        cerr << "Value of --device-threads must be at least 1\n";
        exit(1);
    }
    if (!(pruneThreshold >= 0.0 && pruneThreshold <= 1.0))
    {
        cerr << "Value of --fit-prune must be in [0, 1]\n";
        exit(1);
    }
}

static CLH::ResourceUsage resourceUsage(const po::variables_map &vm)
{
    const int levels = vm[Option::levels].as<int>();
    const int subsampling = vm[Option::subsampling].as<int>();
    const std::size_t maxDeviceSplats = vm[Option::maxDeviceSplats].as<int>();
    const int bucketThreads = vm[Option::bucketThreads].as<int>();
    const int deviceThreads = vm[Option::deviceThreads].as<int>();
    const bool keepBoundary = vm.count(Option::fitKeepBoundary);

    /* Check that we have enough memory on the device. This is no guarantee against OOM, but
     * we can at least turn down silly requests before wasting any time.
     */
    const Grid::size_type maxCells = (Grid::size_type(1U) << (levels + subsampling - 1)) - 1;
    // TODO: get rid of device parameter
    CLH::ResourceUsage totalUsage = DeviceWorkerGroup::resourceUsage(
        deviceThreads, bucketThreads, cl::Device(), maxDeviceSplats, maxCells, levels, keepBoundary);
    return totalUsage;
}

// TODO: do some of these checks during findDevices
static void validateDevice(const cl::Device &device, const CLH::ResourceUsage &totalUsage)
{
    const std::string deviceName = "OpenCL device `" + device.getInfo<CL_DEVICE_NAME>() + "'";
    if (!Marching::validateDevice(device)
        || !SplatTreeCL::validateDevice(device))
    {
        cerr << deviceName << " is not supported.\n";
        exit(1);
    }

    /* Check that we have enough memory on the device. This is no guarantee against OOM, but
     * we can at least turn down silly requests before wasting any time.
     */
    const std::size_t deviceTotalMemory = device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
    const std::size_t deviceMaxMemory = device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
    if (totalUsage.getMaxMemory() > deviceMaxMemory)
    {
        cerr << "Arguments require an allocation of " << totalUsage.getMaxMemory() << ",\n"
            << "but " << deviceName << " only supports up to " << deviceMaxMemory << ".\n"
            << "Try reducing --levels or increasing --subsampling.\n";
        exit(1);
    }
    if (totalUsage.getTotalMemory() > deviceTotalMemory)
    {
        cerr << "Arguments require device memory of " << totalUsage.getTotalMemory() << ",\n"
            << "but " << deviceName << " only has " << deviceTotalMemory << ".\n"
            << "Try reducing --levels, increasing --subsampling or decreasing --max-device-splats.\n";
        exit(1);
    }

    if (totalUsage.getTotalMemory() > deviceTotalMemory * 0.8)
    {
        Log::log[Log::warn] << "WARNING: More than 80% of the memory on " << deviceName << " will be used.\n";
    }
}

static void reportException(std::exception &e)
{
    cerr << '\n';

    std::string *file_name = boost::get_error_info<boost::errinfo_file_name>(e);
    int *err = boost::get_error_info<boost::errinfo_errno>(e);
    if (file_name != NULL)
        cerr << *file_name << ": ";
    if (err != NULL && *err != 0)
        cerr << boost::system::errc::make_error_code((boost::system::errc::errc_t) *err).message() << std::endl;
    else
        cerr << e.what() << std::endl;
}

int main(int argc, char **argv)
{
    Log::log.setLevel(Log::info);

    po::variables_map vm = processOptions(argc, argv);
    if (vm.count(Option::quiet))
        Log::log.setLevel(Log::warn);
    else if (vm.count(Option::debug))
        Log::log.setLevel(Log::debug);

    Timeplot::init();

    std::vector<cl::Device> devices = CLH::findDevices(vm);
    if (devices.empty())
    {
        cerr << "No suitable OpenCL device found\n";
        exit(1);
    }

    validateOptions(vm);
    CLH::ResourceUsage totalUsage = resourceUsage(vm);
    Log::log[Log::info] << "About " << totalUsage.getTotalMemory() / (1024 * 1024) << "MiB of device memory will be used per device.\n";
    BOOST_FOREACH(const cl::Device &device, devices)
    {
        validateDevice(device, totalUsage);
        Log::log[Log::info] << "Using device " << device.getInfo<CL_DEVICE_NAME>() << "\n";
    }

    std::vector<std::pair<cl::Context, cl::Device> > cd;
    cd.reserve(devices.size());
    for (std::size_t i = 0; i < devices.size(); i++)
    {
        cd.push_back(std::make_pair(CLH::makeContext(devices[i]), devices[i]));
    }

    try
    {
        run(cd, vm[Option::outputFile].as<string>(), vm);
        unsigned long long filesWritten = Statistics::getStatistic<Statistics::Counter>("output.files").getTotal();
        if (filesWritten == 0)
            Log::log[Log::warn] << "Warning: no output files written!\n";
        else if (filesWritten == 1)
            Log::log[Log::info] << "1 output file written.\n";
        else
            Log::log[Log::info] << filesWritten << " output files written.\n";
    }
    catch (cl::Error &e)
    {
        cerr << "\nOpenCL error in " << e.what() << " (" << e.err() << ")\n";
        return 1;
    }
    catch (std::ios::failure &e)
    {
        reportException(e);
        return 1;
    }
    catch (std::runtime_error &e)
    {
        reportException(e);
        return 1;
    }

    return 0;
}
