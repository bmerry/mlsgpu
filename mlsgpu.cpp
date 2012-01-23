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
#include <boost/numeric/conversion/converter.hpp>
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

typedef boost::numeric::converter<
    int,
    float,
    boost::numeric::conversion_traits<int, float>,
    boost::numeric::def_overflow_handler,
    boost::numeric::Ceil<float> > RoundUp;
typedef boost::numeric::converter<
    int,
    float,
    boost::numeric::conversion_traits<int, float>,
    boost::numeric::def_overflow_handler,
    boost::numeric::Floor<float> > RoundDown;

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
    const std::size_t maxSplats = 3000000;
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

template<typename InputIterator>
static void loadInputSplats(InputIterator first, InputIterator last, std::vector<Splat> &out, float smooth)
{
    out.clear();
    for (InputIterator in = first; in != last; ++in)
    {
        try
        {
            FastPly::Reader reader(*in, smooth);
            size_t pos = out.size();
            out.resize(pos + reader.numVertices());
            reader.readVertices(0, reader.numVertices(), &out[pos]);
        }
        catch (FastPly::FormatError &e)
        {
            throw FastPly::FormatError(*in + ": " + e.what());
        }
    }
}

static void loadInputSplats(const po::variables_map &vm, std::vector<Splat> &out, float smooth)
{
    const vector<string> &inputs = vm[Option::inputFile].as<vector<string> >();
    loadInputSplats(inputs.begin(), inputs.end(), out, smooth);
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

/**
 * Grid that encloses the bounding spheres of all the input splats.
 *
 * The grid is constructed as follows:
 *  -# The bounding box of the sample points is found, ignoring influence regions.
 *  -# The lower bound is used as the grid reference point.
 *  -# The grid extends are set to cover the full bounding box.
 *
 * @param first, last   Iterator range for the splats.
 * @param spacing       The spacing between grid vertices.
 *
 * @pre The iterator range is not empty.
 */
template<typename ForwardIterator>
static Grid makeGrid(ForwardIterator first, ForwardIterator last, float spacing)
{
    MLSGPU_ASSERT(first != last, std::invalid_argument);

    float low[3];
    float bboxMin[3];
    float bboxMax[3];

    // Load the first splat
    {
        const float radius = first->radius;
        for (unsigned int i = 0; i < 3; i++)
        {
            low[i] = first->position[i];
            bboxMin[i] = low[i] - radius;
            bboxMax[i] = low[i] + radius;
        }
    }
    first++;

    for (ForwardIterator i = first; i != last; ++i)
    {
        const float radius = i->radius;
        for (unsigned int j = 0; j < 3; j++)
        {
            float p = i->position[j];
            low[j] = min(low[j], p);
            bboxMin[j] = min(bboxMin[j], p - radius);
            bboxMax[j] = max(bboxMax[j], p + radius);
        }
    }

    int extents[3][2];
    for (unsigned int i = 0; i < 3; i++)
    {
        float l = (bboxMin[i] - low[i]) / spacing;
        float h = (bboxMax[i] - low[i]) / spacing;
        extents[i][0] = RoundDown::convert(l);
        extents[i][1] = RoundUp::convert(h);
    }
    return Grid(low, spacing,
                extents[0][0], extents[0][1], extents[1][0], extents[1][1], extents[2][0], extents[2][1]);
}

static void showBucket(const boost::ptr_vector<FastPly::Reader> &files, SplatRange::index_type numSplats, SplatRangeConstIterator first, SplatRangeConstIterator last, const Grid &grid)
{
    for (int i = 0; i < 3; i++)
    {
        const pair<int, int> e = grid.getExtent(i);
        if (i > 0) cout << " x ";
        cout << "[" << e.first << "," << e.second << "]";
    }
    cout << ": " << numSplats << " splats in " << last - first << " ranges\n";
}

static void run(const cl::Context &context, const cl::Device &device, const string &out, const po::variables_map &vm)
{
    const int subsampling = vm[Option::subsampling].as<int>();
    const int levels = vm[Option::levels].as<int>();
    const unsigned int block = 1U << (levels + subsampling - 1);
    const unsigned int blockCells = block - 1;

    float spacing = vm[Option::fitGrid].as<double>();
    float smooth = vm[Option::fitSmooth].as<double>();
    vector<Splat> splats;
    loadInputSplats(vm, splats, smooth);
    Grid grid = makeGrid(splats.begin(), splats.end(), spacing);

    boost::ptr_vector<FastPly::Reader> files;
    prepareInputs(files, vm, smooth);
    // TODO: blockCells will be just less than a power of 2, so the
    // actual calls will end up at almost half
    bucket(files, grid, 1000000, blockCells, 1000000, showBucket);

    /* Round up to multiple of block size
     */
    unsigned int cells[3];
    unsigned int chunks[3];
    for (unsigned int i = 0; i < 3; i++)
    {
        std::pair<int, int> extent = grid.getExtent(i);
        chunks[i] = (extent.second - extent.first + blockCells - 1) / blockCells;
        cells[i] = chunks[i] * blockCells;
        grid.setExtent(i, extent.first, extent.first + cells[i]);
    }
    Log::log[Log::info] << "Octree cells: " << cells[0] << " x " << cells[1] << " x " << cells[2] << endl;

    cl::CommandQueue queue(context, device);
    SplatTreeCL tree(context, levels, splats.size());
    Marching marching(context, device, block, block);

    MlsFunctor input(context);

    FastPly::WriterType writerType = vm[Option::writer].as<Choice<FastPly::WriterTypeWrapper> >();
    MeshType meshType = vm[Option::mesh].as<Choice<MeshTypeWrapper> >();
    boost::scoped_ptr<FastPly::WriterBase> writer(FastPly::createWriter(writerType));
    boost::scoped_ptr<MeshBase> mesh(createMesh(meshType, *writer, out));

    /* TODO: partition splats */
    for (unsigned int pass = 0; pass < mesh->numPasses(); pass++)
    {
        Log::log[Log::info] << "\nPass " << pass + 1 << "/" << mesh->numPasses() << endl;
        boost::progress_display progress(chunks[0] * chunks[1] * chunks[2], Log::log[Log::info]);

        Marching::OutputFunctor output = mesh->outputFunctor(pass);
        for (unsigned int bz = 0; bz < cells[2]; bz += blockCells)
            for (unsigned int by = 0; by < cells[1]; by += blockCells)
                for (unsigned int bx = 0; bx < cells[0]; bx += blockCells)
                {
                    cl_uint3 keyOffset = {{ bx, by, bz }};
                    Grid sub = grid.subGrid(bx, bx + blockCells,
                                            by, by + blockCells,
                                            bz, bz + blockCells);
                    {
                        Timer timer;
                        tree.enqueueBuild(queue, &splats[0], splats.size(), sub, subsampling, CL_FALSE);
                        queue.finish();
                        Log::log[Log::debug] << "Build: " << timer.getElapsed() << '\n';
                    }

                    input.set(sub, tree, subsampling);

                    {
                        Timer timer;
                        marching.generate(queue, input, output, sub, keyOffset,
                                          NULL);
                        Log::log[Log::debug] << "Process: " << timer.getElapsed() << endl;
                    }
                    ++progress;
                }
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
