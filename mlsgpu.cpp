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
#include <tr1/unordered_map>
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
#include "src/work_queue.h"
#include "src/workers.h"
#include "src/progress.h"
#include "src/clip.h"
#include "src/mesh_filter.h"

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
    const char * const mesher = "mesher";
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
        (Option::fitBoundaryLimit, po::value<double>()->default_value(1.5), "Tuning factor for boundary detection");
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
        (Option::levels,       po::value<int>()->default_value(7), "Levels in octree")
        (Option::subsampling,  po::value<int>()->default_value(2), "Subsampling of octree")
        (Option::maxDeviceSplats, po::value<int>()->default_value(1000000), "Maximum splats per block on the device")
        (Option::maxHostSplats, po::value<std::size_t>()->default_value(8000000), "Maximum splats per block on the CPU")
        (Option::maxSplit,     po::value<int>()->default_value(2097152), "Maximum fan-out in partitioning")
        (Option::bucketThreads, po::value<int>()->default_value(4), "Number of threads for bucketing splats")
        (Option::deviceThreads, po::value<int>()->default_value(1), "Number of threads for submitting OpenCL work")
        (Option::mesher,       po::value<Choice<MesherTypeWrapper> >()->default_value(STXXL_MESHER), "Mesher (weld | big | stxxl)")
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
        *out << Statistics::Registry::getInstance();
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
        (Option::splitSize, po::value<unsigned int>()->default_value(100), "approximate size of output chunks (MB)");

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
    BOOST_FOREACH(const string &name, names)
    {
        FastPly::Reader *reader = new FastPly::Reader(name, smooth);
        files.addFile(reader);
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
class HostBlock
{
public:
    void operator()(
        const typename SplatSet::Traits<Splats>::subset_type &splatSet,
        const Grid &grid,
        const Bucket::Recursion &recursionState);

    HostBlock(FineBucketGroup &outGroup, const Grid &fullGrid);
private:
    ChunkId curChunkId;
    FineBucketGroup &outGroup;
    const Grid &fullGrid;
};

template<typename Splats>
HostBlock<Splats>::HostBlock(FineBucketGroup &outGroup, const Grid &fullGrid)
: curChunkId(), outGroup(outGroup), fullGrid(fullGrid)
{
    outGroup.producerStart(curChunkId);
}

template<typename Splats>
void HostBlock<Splats>::operator()(
    const typename SplatSet::Traits<Splats>::subset_type &splats,
    const Grid &grid, const Bucket::Recursion &recursionState)
{
    if (recursionState.chunk != curChunkId.coords)
    {
        ChunkId old = curChunkId;
        curChunkId.gen++;
        curChunkId.coords = recursionState.chunk;
        outGroup.producerNext(old, curChunkId);
    }

    Statistics::Registry &registry = Statistics::Registry::getInstance();

    boost::shared_ptr<FineBucketGroup::WorkItem> item = outGroup.get();
    item->grid = grid;
    item->recursionState = recursionState;
    item->splats.clear();
    float invSpacing = 1.0f / fullGrid.getSpacing();

    {
        Statistics::Timer timer("host.block.load");
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

    outGroup.push(curChunkId, item);
}

/**
 * Second phase of execution, which is templated on the collection type
 * (which in turn depends on whether --sort was given or not).
 *
 * @todo --sort is gone for now.
 */
template<typename Splats>
static void run2(const cl::Context &context, const cl::Device &device, const string &out,
                 const po::variables_map &vm,
                 const Splats &splats,
                 const Grid &grid)
{
    const int subsampling = vm[Option::subsampling].as<int>();
    const int levels = vm[Option::levels].as<int>();
    const FastPly::WriterType writerType = vm[Option::writer].as<Choice<FastPly::WriterTypeWrapper> >();
    const MesherType mesherType = vm[Option::mesher].as<Choice<MesherTypeWrapper> >();
    const std::size_t maxDeviceSplats = vm[Option::maxDeviceSplats].as<int>();
    const std::size_t maxHostSplats = vm[Option::maxHostSplats].as<std::size_t>();
    const std::size_t maxSplit = vm[Option::maxSplit].as<int>();
    const double pruneThreshold = vm[Option::fitPrune].as<double>();
    const bool keepBoundary = vm.count(Option::fitKeepBoundary);
    const float boundaryLimit = vm[Option::fitBoundaryLimit].as<double>();
    const bool split = vm.count(Option::split);
    const unsigned int splitSize = vm[Option::splitSize].as<unsigned int>();

    const unsigned int block = 1U << (levels + subsampling - 1);
    const unsigned int blockCells = block - 1;

    const unsigned int numBucketThreads = vm[Option::bucketThreads].as<int>();
    const unsigned int numDeviceThreads = vm[Option::deviceThreads].as<int>();

    unsigned int chunkCells = 0;
    boost::array<Grid::size_type, 3> numChunks = {{1, 1, 1}};
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

        std::tr1::uint64_t totalChunks = 1;
        for (unsigned int i = 0; i < 3; i++)
        {
            numChunks[i] = divUp(grid.numCells(i), chunkCells);
            totalChunks *= numChunks[i];
        }
        if (totalChunks > 1000)
        {
            Log::log[Log::warn] << totalChunks << " output files will be produced. This may fail." << endl;
        }
    }

    MesherGroup mesherGroup(1);
    DeviceWorkerGroup deviceWorkerGroup(
        numDeviceThreads, numDeviceThreads + numBucketThreads, mesherGroup,
        grid, context, device, maxDeviceSplats, blockCells, levels, subsampling,
        keepBoundary, boundaryLimit);
    FineBucketGroup fineBucketGroup(
        numBucketThreads, numBucketThreads + 1, deviceWorkerGroup,
        grid, context, device, maxHostSplats, maxDeviceSplats, blockCells, maxSplit);
    HostBlock<Splats> hostBlock(fineBucketGroup, grid);

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
        fineBucketGroup.start();
        deviceWorkerGroup.start();
        mesherGroup.start();

        Bucket::bucket(splats, grid, maxHostSplats, blockCells, chunkCells, true, maxSplit,
                       boost::ref(hostBlock), &progress);

        /* Shut down threads. Note that it has to be done in forward order to
         * satisfy the requirement that stop() is only called after producers
         * are terminated.
         */
        fineBucketGroup.stop();
        deviceWorkerGroup.stop();
        mesherGroup.stop();
    }


    {
        Statistics::Timer timer("finalize.time");

        mesher->finalize(&Log::log[Log::info]);
        mesher->write(*writer, namer, &Log::log[Log::info]);
    }
}

static void run(const cl::Context &context, const cl::Device &device, const string &out,
                const po::variables_map &vm)
{
    const float spacing = vm[Option::fitGrid].as<double>();
    const float smooth = vm[Option::fitSmooth].as<double>();
    const int subsampling = vm[Option::subsampling].as<int>();
    const int levels = vm[Option::levels].as<int>();
    const unsigned int block = 1U << (levels + subsampling - 1);
    const unsigned int blockCells = block - 1;

    typedef SplatSet::FastBlobSet<SplatSet::FileSet, stxxl::VECTOR_GENERATOR<SplatSet::BlobData>::result > Splats;
    Splats splats;

    boost::ptr_vector<FastPly::Reader> files;
    prepareInputs(splats, vm, smooth);
    Grid grid;

    try
    {
        Statistics::Timer timer("bbox.time");
        splats.computeBlobs(spacing, blockCells, &Log::log[Log::info]);
    }
    catch (std::length_error &e)
    {
        cerr << "At least one input point is required.\n";
        exit(1);
    }

    run2(context, device, out, vm, splats, splats.getBoundingGrid());
    writeStatistics(vm);
}

static void validateOptions(const cl::Device &device, const po::variables_map &vm)
{
    if (!Marching::validateDevice(device)
        || !SplatTreeCL::validateDevice(device))
    {
        cerr << "This OpenCL device is not supported.\n";
        exit(1);
    }

    const int levels = vm[Option::levels].as<int>();
    const int subsampling = vm[Option::subsampling].as<int>();
    const std::size_t maxDeviceSplats = vm[Option::maxDeviceSplats].as<int>();
    const std::size_t maxHostSplats = vm[Option::maxHostSplats].as<std::size_t>();
    const std::size_t maxSplit = vm[Option::maxSplit].as<int>();
    const int bucketThreads = vm[Option::bucketThreads].as<int>();
    const int deviceThreads = vm[Option::deviceThreads].as<int>();
    const double pruneThreshold = vm[Option::fitPrune].as<double>();
    const bool keepBoundary = vm.count(Option::fitKeepBoundary);

    int maxLevels = std::min(std::size_t(Marching::MAX_DIMENSION_LOG2 + 1), SplatTreeCL::MAX_LEVELS);
    /* TODO make dynamic, considering maximum image sizes etc */
    if (levels < 1 || levels > maxLevels)
    {
        cerr << "Value of --levels must be in the range 1 to " << maxLevels << ".\n";
        exit(1);
    }
    if (subsampling < 0)
    {
        cerr << "Value of --subsampling must be non-negative.\n";
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
        cerr << "Sum of --subsampling and --levels it too small.\n";
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

    /* Check that we have enough memory on the device. This is no guarantee against OOM, but
     * we can at least turn down silly requests before wasting any time.
     */
    const Grid::size_type maxCells = (Grid::size_type(1U) << (levels + subsampling - 1)) - 1;
    CLH::ResourceUsage totalUsage = DeviceWorkerGroup::resourceUsage(
        deviceThreads, deviceThreads + bucketThreads, device, maxDeviceSplats, maxCells, levels, keepBoundary);

    const std::size_t deviceTotalMemory = device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
    const std::size_t deviceMaxMemory = device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
    if (totalUsage.getMaxMemory() > deviceMaxMemory)
    {
        cerr << "Arguments require an allocation of " << totalUsage.getMaxMemory() << ",\n"
            << "but the OpenCL device only supports up to " << deviceMaxMemory << ".\n"
            << "Try reducing --levels or --subsampling.\n";
        exit(1);
    }
    if (totalUsage.getTotalMemory() > deviceTotalMemory)
    {
        cerr << "Arguments require device memory of " << totalUsage.getTotalMemory() << ",\n"
            << "but the OpenCL device has " << deviceTotalMemory << ".\n"
            << "Try reducing --levels or --subsampling.\n";
        exit(1);
    }

    Log::log[Log::info] << "About " << totalUsage.getTotalMemory() / (1024 * 1024) << "MiB of device memory will be used.\n";
    if (totalUsage.getTotalMemory() > deviceTotalMemory * 0.8)
    {
        Log::log[Log::warn] << "WARNING: More than 80% of the device memory will be used.\n";
    }
}

int main(int argc, char **argv)
{
    Log::log.setLevel(Log::debug);

    po::variables_map vm = processOptions(argc, argv);
    if (vm.count(Option::quiet))
        Log::log.setLevel(Log::warn);
    else if (vm.count(Option::debug))
        Log::log.setLevel(Log::debug);

    cl::Device device = CLH::findDevice(vm);
    if (!device())
    {
        cerr << "No suitable OpenCL device found\n";
        exit(1);
    }
    Log::log[Log::info] << "Using device " << device.getInfo<CL_DEVICE_NAME>() << "\n";

    validateOptions(device, vm);

    cl::Context context = CLH::makeContext(device);

    try
    {
        run(context, device, vm[Option::outputFile].as<string>(), vm);
    }
    catch (ios::failure &e)
    {
        cerr << e.what() << '\n';
        return 1;
    }
    catch (PLY::FormatError &e)
    {
        cerr << e.what() << '\n';
        return 1;
    }
    catch (cl::Error &e)
    {
        cerr << "OpenCL error in " << e.what() << " (" << e.err() << ")\n";
        return 1;
    }
    catch (Bucket::DensityError &e)
    {
        cerr << "The splats were too dense. Try passing a higher value for --max-device-splats.\n";
        return 1;
    }

    return 0;
}
