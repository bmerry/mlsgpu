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
#include <boost/array.hpp>
#include <boost/progress.hpp>
#include <tr1/unordered_map>
#include <iostream>
#include <map>
#include <vector>
#include <algorithm>
#include "src/clh.h"
#include "src/logging.h"
#include "src/timer.h"
#include "src/fast_ply.h"
#include "src/splat.h"
#include "src/grid.h"
#include "src/splat_tree_cl.h"
#include "src/marching.h"
#include "src/mls.h"
#include "src/mesh.h"
#include "src/misc.h"
#include "src/bucket.h"

namespace po = boost::program_options;
using namespace std;

namespace Option
{
    const char * const help = "help";
    const char * const quiet = "quiet";

    const char * const fitSmooth = "fit-smooth";
    const char * const fitGrid = "fit-grid";

    const char * const inputFile = "input-file";
    const char * const outputFile = "output-file";

    const char * const levels = "levels";
    const char * const subsampling = "subsampling";
    const char * const mesh = "mesh";
    const char * const writer = "writer";
};

static void addCommonOptions(po::options_description &opts)
{
    opts.add_options()
        ("help,h",                "Show help")
        ("quiet,q",               "Do not show informational messages");
}

static void addFitOptions(po::options_description &opts)
{
    opts.add_options()
        (Option::fitSmooth,       po::value<double>()->default_value(4.0),  "Smoothing factor")
        (Option::fitGrid,         po::value<double>()->default_value(0.01), "Spacing of grid cells");
}

static void addAdvancedOptions(po::options_description &opts)
{
    po::options_description advanced("Advanced options");
    advanced.add_options()
        (Option::levels,       po::value<int>()->default_value(7), "Levels in octree")
        (Option::subsampling,  po::value<int>()->default_value(2), "Subsampling of octree")
        (Option::mesh,         po::value<Choice<MeshTypeWrapper> >()->default_value(BIG_MESH), "Mesh collector (simple | weld | big)")
        (Option::writer,       po::value<Choice<FastPly::WriterTypeWrapper> >()->default_value(FastPly::STREAM_WRITER), "File writer class (mmap | stream)");
    opts.add(advanced);
}

static void validateOptions(const cl::Device &device, const po::variables_map &vm)
{
    if (!Marching::validateDevice(device)
        || !SplatTreeCL::validateDevice(device))
    {
        cerr << "This OpenCL device is not supported.\n";
        exit(1);
    }

    int levels = vm[Option::levels].as<int>();
    int subsampling = vm[Option::subsampling].as<int>();

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
    if (subsampling > Marching::MAX_DIMENSION_LOG2 + 1 - levels)
    {
        cerr << "Sum of --subsampling and --levels is too large.\n";
        exit(1);
    }

    /* Check that we have enough memory on the device. This is no guarantee against OOM, but
     * we can at least turn down silly requests before wasting any time.
     *
     * TODO: get an actual value for maxSplats once we implement splat partitioning.
     */
    const std::size_t maxSplats = 1000000;
    const std::size_t block = std::size_t(1U) << (levels + subsampling - 1);
    std::pair<std::tr1::uint64_t, std::tr1::uint64_t> marchingMemory = Marching::deviceMemory(device, block, block);
    std::pair<std::tr1::uint64_t, std::tr1::uint64_t> splatTreeMemory = SplatTreeCL::deviceMemory(device, levels, maxSplats);
    const std::tr1::uint64_t total = marchingMemory.first + splatTreeMemory.first;
    const std::tr1::uint64_t max = std::max(marchingMemory.second, splatTreeMemory.second);

    const std::size_t deviceTotal = device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
    const std::size_t deviceMax = device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
    if (max > deviceMax)
    {
        cerr << "Arguments require an allocation of " << max << ",\n"
            << "but the OpenCL device only supports up to " << deviceMax << ".\n"
            << "Try reducing --levels or --subsampling.\n";
        exit(1);
    }
    if (total > deviceTotal)
    {
        cerr << "Arguments require device memory of " << total << ",\n"
            << "but the OpenCL device has " << deviceTotal << ".\n"
            << "Try reducing --levels or --subsampling.\n";
        exit(1);
    }

    Log::log[Log::info] << "About " << total / (1024 * 1024) << "MiB of device memory will be used.\n";
    if (total > deviceTotal * 0.8)
    {
        Log::log[Log::warn] << "More than 80% of the device memory will be used.\n";
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
    addAdvancedOptions(desc);
    desc.add_options()
        ("output-file,o",   po::value<string>()->required(), "output file");

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

static void prepareInputs(boost::ptr_vector<FastPly::Reader> &files, const po::variables_map &vm, float smooth)
{
    const vector<string> &names = vm[Option::inputFile].as<vector<string> >();
    files.clear();
    files.reserve(names.size());
    BOOST_FOREACH(const string &name, names)
    {
        FastPly::Reader *reader = new FastPly::Reader(name, smooth);
        files.push_back(reader);
    }
}

class BlockRun
{
private:
    const cl::CommandQueue queue;
    SplatTreeCL tree;
    MlsFunctor input;
    Marching marching;
    Marching::OutputFunctor output;
    Grid fullGrid;
    int subsampling;

public:
    BlockRun(const cl::Context &context, const cl::Device &device,
             std::size_t maxSplats, std::size_t maxCells,
             int levels, int subsampling);
    void operator()(const boost::ptr_vector<FastPly::Reader> &files, SplatRange::index_type numSplats, SplatRangeConstIterator first, SplatRangeConstIterator last, const Grid &grid);

    void setGrid(const Grid &grid) { this->fullGrid = grid; }
    void setOutput(const Marching::OutputFunctor &output) { this->output = output; }
};

BlockRun::BlockRun(
    const cl::Context &context, const cl::Device &device,
    std::size_t maxSplats, std::size_t maxCells,
    int levels, int subsampling)
    : queue(context, device),
    tree(context, levels, maxSplats),
    input(context),
    marching(context, device, maxCells + 1, maxCells + 1),
    subsampling(subsampling)
{
}

void BlockRun::operator()(const boost::ptr_vector<FastPly::Reader> &files, SplatRange::index_type numSplats, SplatRangeConstIterator first, SplatRangeConstIterator last, const Grid &grid)
{
    cl_uint3 keyOffset;
    for (int i = 0; i < 3; i++)
        keyOffset.s[i] = grid.getExtent(i).first - fullGrid.getExtent(i).first;

    // TODO: use mapping to transfer the data directly into a buffer
    vector<Splat> splats(numSplats);
    std::size_t pos = 0;
    for (SplatRangeConstIterator i = first; i != last; i++)
    {
        files[i->scan].readVertices(i->start, i->size, &splats[pos]);
        pos += i->size;
    }

    {
        Timer timer;
        tree.enqueueBuild(queue, &splats[0], numSplats, grid, subsampling, CL_FALSE);
        queue.finish();
        Log::log[Log::debug] << "build: " << timer.getElapsed() << '\n';
    }

    input.set(grid, tree, subsampling);

    {
        Timer timer;
        marching.generate(queue, input, output, grid, keyOffset, NULL);
        Log::log[Log::debug] << "process: " << timer.getElapsed() << endl;
    }
}

static void run(const cl::Context &context, const cl::Device &device, const string &out, const po::variables_map &vm)
{
    const int subsampling = vm[Option::subsampling].as<int>();
    const int levels = vm[Option::levels].as<int>();
    const float spacing = vm[Option::fitGrid].as<double>();
    const float smooth = vm[Option::fitSmooth].as<double>();
    const FastPly::WriterType writerType = vm[Option::writer].as<Choice<FastPly::WriterTypeWrapper> >();
    const MeshType meshType = vm[Option::mesh].as<Choice<MeshTypeWrapper> >();
    const std::size_t maxSplats = 1000000;
    const std::size_t maxSplit = 1000000;

    const unsigned int block = 1U << (levels + subsampling - 1);
    const unsigned int blockCells = block - 1;

    boost::ptr_vector<FastPly::Reader> files;
    prepareInputs(files, vm, smooth);

    BlockRun blockRun(context, device, maxSplats, blockCells, levels, subsampling);
    const Grid grid = makeGrid(files, spacing);
    blockRun.setGrid(grid);

    boost::scoped_ptr<FastPly::WriterBase> writer(FastPly::createWriter(writerType));
    boost::scoped_ptr<MeshBase> mesh(createMesh(meshType, *writer, out));
    for (unsigned int pass = 0; pass < mesh->numPasses(); pass++)
    {
        Log::log[Log::info] << "\nPass " << pass + 1 << "/" << mesh->numPasses() << endl;

        blockRun.setOutput(mesh->outputFunctor(pass));
        // TODO: blockCells will be just less than a power of 2, so the
        // actual calls will end up at almost half
        bucket(files, grid, maxSplats, blockCells, maxSplit, boost::ref(blockRun));
    }

    mesh->finalize();
    mesh->write(*writer, out);
}

int main(int argc, char **argv)
{
    Log::log.setLevel(Log::info);

    po::variables_map vm = processOptions(argc, argv);
    if (vm.count(Option::quiet))
        Log::log.setLevel(Log::warn);

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

    return 0;
}
