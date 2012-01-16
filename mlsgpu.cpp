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
#include <boost/ref.hpp>
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
};

static void addCommonOptions(po::options_description &opts)
{
    opts.add_options()
        ("help,h",                "show help")
        (Option::quiet,           "do not show informational messages");
}

static void addFitOptions(po::options_description &opts)
{
    opts.add_options()
        (Option::fitSmooth,       po::value<double>()->default_value(4.0),  "smoothing factor")
        (Option::fitGrid,         po::value<double>()->default_value(0.01), "spacing of grid cells");
}

static po::variables_map processOptions(int argc, char **argv)
{
    po::positional_options_description positional;
    positional.add(Option::inputFile, -1);

    po::options_description desc("General options");
    addCommonOptions(desc);
    addFitOptions(desc);
    desc.add_options()
        ("output-file,o",   po::value<string>(), "output file");

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
            cout << desc << '\n';
            exit(0);
        }

        if (!vm.count(Option::inputFile))
        {
            cerr << "At least one input file must be specified\n\n" << desc << '\n';
            exit(1);
        }
        if (!vm.count(Option::outputFile))
        {
            cerr << "An output file must be specified\n\n" << desc << '\n';
            exit(1);
        }

        return vm;
    }
    catch (po::error &e)
    {
        cerr << e.what() << "\n\n" << desc << '\n';
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
            FastPly::Reader reader(*in);
            size_t pos = out.size();
            out.resize(pos + reader.numVertices());
            reader.readVertices(0, reader.numVertices(), &out[pos]);
        }
        catch (FastPly::FormatError &e)
        {
            throw FastPly::FormatError(*in + ": " + e.what());
        }
    }
    BOOST_FOREACH(Splat &splat, out)
    {
        splat.radius *= smooth;
        splat.quality = 1.0 / (splat.radius * splat.radius);
    }
}

static void loadInputSplats(const po::variables_map &vm, std::vector<Splat> &out, float smooth)
{
    const vector<string> &inputs = vm[Option::inputFile].as<vector<string> >();
    loadInputSplats(inputs.begin(), inputs.end(), out, smooth);
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

/**
 * Output functor for @ref Marching which does global welding.
 *
 * @warning This functor contains internal state, so it must be passed wrapped using @c boost::ref.
 */
class OutputFunctor
{
private:
    vector<boost::array<cl_float, 3> > hInternalVertices;
    vector<boost::array<cl_float, 3> > hExternalVertices;
    vector<cl_ulong> hExternalKeys;
    vector<boost::array<cl_uint, 3> > hIndices;

public:
    OutputFunctor() {}

    inline size_t numVertices() const
    {
        return hInternalVertices.size() + hExternalVertices.size();
    }

    void validate() const;

    void weld();

    void operator()(const cl::CommandQueue &queue,
                    const cl::Buffer &vertices,
                    const cl::Buffer &vertexKeys,
                    const cl::Buffer &indices,
                    std::size_t numVertices,
                    std::size_t numInternalVertices,
                    std::size_t numIndices,
                    cl::Event *event);

    void write(const std::string &out);
};

void OutputFunctor::validate() const
{
    for (size_t i = 0; i < hIndices.size(); i++)
        for (size_t j = 0; j < 3; j++)
            assert(hIndices[i][j] < hInternalVertices.size()
                   || ~hIndices[i][j] < hExternalVertices.size());
}

void OutputFunctor::operator()(const cl::CommandQueue &queue,
                               const cl::Buffer &vertices,
                               const cl::Buffer &vertexKeys,
                               const cl::Buffer &indices,
                               std::size_t numVertices,
                               std::size_t numInternalVertices,
                               std::size_t numIndices,
                               cl::Event *event)
{
    cl::Event indicesEvent, last;
    std::vector<cl::Event> wait(1);

    std::size_t oldIV = hInternalVertices.size();
    std::size_t oldEV = hExternalVertices.size();
    std::size_t oldV = oldIV + oldEV;
    std::size_t oldTriangles = hIndices.size();
    std::size_t numExternalVertices = numVertices - numInternalVertices;
    std::size_t numTriangles = numIndices / 3;
    hInternalVertices.resize(oldIV + numInternalVertices);
    hExternalVertices.resize(oldEV + numExternalVertices);
    hExternalKeys.resize(oldEV + numExternalVertices);
    hIndices.resize(oldTriangles + numTriangles);
    // TODO: revisit dependency tracking to allow overlaps
    queue.enqueueReadBuffer(indices, CL_FALSE, 0, numIndices * sizeof(cl_uint),
                            &hIndices[oldTriangles][0],
                            NULL, &indicesEvent);
    last = indicesEvent;
    wait[0] = last;
    queue.flush(); // Start the read in the background
    if (numInternalVertices > 0)
    {
        queue.enqueueReadBuffer(vertices, CL_FALSE, 0, 3 * numInternalVertices * sizeof(cl_float),
                                &hInternalVertices[oldIV][0],
                                &wait, &last);
        wait[0] = last;
    }
    if (numExternalVertices > 0)
    {
        queue.enqueueReadBuffer(vertices, CL_FALSE,
                                3 * numInternalVertices * sizeof(cl_float),
                                3 * numExternalVertices * sizeof(cl_float),
                                &hExternalVertices[oldEV][0],
                                &wait, &last);
        wait[0] = last;
        queue.enqueueReadBuffer(vertexKeys, CL_FALSE,
                                numInternalVertices * sizeof(cl_ulong),
                                numExternalVertices * sizeof(cl_ulong),
                                &hExternalKeys[oldEV],
                                &wait, &last);
        wait[0] = last;
    }
    if (event != NULL)
        *event = last;

    /* Rewrite the indices to split between internal and external.
     * TODO: do this on the GPU as part of remapping.
     */
    indicesEvent.wait();
    // note: these will both wrap around
    cl_uint iOffset = oldIV - oldV;
    cl_uint eOffset = oldEV - oldV - numInternalVertices;
    for (size_t i = oldTriangles; i < hIndices.size(); i++)
        for (int j = 0; j < 3; j++)
            if (hIndices[i][j] < oldV + numInternalVertices)
                hIndices[i][j] += iOffset;
            else
                hIndices[i][j] = ~(hIndices[i][j] + eOffset);
}

void OutputFunctor::weld()
{
    /* TODO: weld as we go? */
    std::size_t unwelded = hExternalVertices.size();
    std::size_t welded = 0;

    std::tr1::unordered_map<cl_ulong, cl_uint> keyMap;
    vector<cl_uint> remap(unwelded);
    for (size_t i = 0; i < unwelded; i++)
    {
        std::tr1::unordered_map<cl_ulong, cl_uint>::iterator pos;
        pos = keyMap.find(hExternalKeys[i]);
        if (pos == keyMap.end())
        {
            keyMap[hExternalKeys[i]] = welded;
            hExternalVertices[welded] = hExternalVertices[i];
            remap[i] = welded;
            welded++;
        }
        else
        {
            remap[i] = pos->second;
        }
    }
    hExternalVertices.resize(welded);
    for (size_t i = 0; i < hIndices.size(); i++)
        for (int j = 0; j < 3; j++)
        {
            cl_uint &index = hIndices[i][j];
            if (~index < unwelded)
                index = ~remap[~index];
        }
}

void OutputFunctor::write(const std::string &out)
{
    FastPly::Writer writer;
    writer.setNumVertices(numVertices());
    writer.setNumTriangles(hIndices.size());
    writer.open(out);
    writer.writeVertices(0, hInternalVertices.size(), &hInternalVertices[0][0]);
    writer.writeVertices(hInternalVertices.size(), hExternalVertices.size(), &hExternalVertices[0][0]);
    /* Rewrite indices to reflect that external vertices are following the internal ones */
    for (size_t i = 0; i < hIndices.size(); i++)
        for (int j = 0; j < 3; j++)
        {
            cl_uint &index = hIndices[i][j];
            if (index >= hInternalVertices.size())
                index = ~index + hInternalVertices.size();
        }
    writer.writeTriangles(0, hIndices.size(), &hIndices[0][0]);
}

static void run(const cl::Context &context, const cl::Device &device, const string &out, const po::variables_map &vm)
{
    const unsigned int subsampling = 2;
    const unsigned int maxLevels = 8;
    const unsigned int maxBlock = 1U << (maxLevels + subsampling - 1);
    const unsigned int maxCells = maxBlock - 1;

    float spacing = vm[Option::fitGrid].as<double>();
    float smooth = vm[Option::fitSmooth].as<double>();
    vector<Splat> splats;
    loadInputSplats(vm, splats, smooth);
    Grid grid = makeGrid(splats.begin(), splats.end(), spacing);

    /* Round up to multiple of block size
     */
    unsigned int cells[3];
    for (unsigned int i = 0; i < 3; i++)
    {
        std::pair<int, int> extent = grid.getExtent(i);
        cells[i] = (extent.second - extent.first + maxCells - 1) / maxCells * maxCells;
        grid.setExtent(i, extent.first, extent.first + cells[i]);
    }
    cout << "Octree cells: " << cells[0] << " x " << cells[1] << " x " << cells[2] << "\n";

    cl::CommandQueue queue(context, device);
    SplatTreeCL tree(context, maxLevels, splats.size());
    Marching marching(context, device, maxBlock, maxBlock);

    MlsFunctor input(context);

    OutputFunctor output;

    /* TODO: partition splats */
    for (unsigned int bz = 0; bz < cells[2]; bz += maxCells)
        for (unsigned int by = 0; by < cells[1]; by += maxCells)
            for (unsigned int bx = 0; bx < cells[0]; bx += maxCells)
            {
                cl_uint3 keyOffset = {{ bx, by, bz }};
                Grid sub = grid.subGrid(bx, bx + maxCells,
                                        by, by + maxCells,
                                        bz, bz + maxCells);
                {
                    Timer timer;
                    tree.enqueueBuild(queue, &splats[0], splats.size(), sub, subsampling, CL_FALSE);
                    queue.finish();
                    cout << "Build: " << timer.getElapsed() << '\n';
                }

                input.set(sub, tree, subsampling);

                {
                    Timer timer;
                    marching.generate(queue, input, boost::ref(output), sub, keyOffset,
                                      output.numVertices(), NULL);
                    cout << "Process: " << timer.getElapsed() << endl;
                }
            }

    output.validate();
    output.weld();
    output.write(out);
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
