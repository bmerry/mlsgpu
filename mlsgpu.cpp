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
#include <iostream>
#include <map>
#include <vector>
#include <algorithm>
#include "src/clh.h"
#include "src/logging.h"
#include "src/timer.h"
#include "src/ply.h"
#include "src/ply_mesh.h"
#include "src/splat.h"
#include "src/files.h"
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
        return vm;
    }
    catch (po::error &e)
    {
        cerr << e.what() << "\n\n" << desc << '\n';
        exit(1);
    }
}

static void makeInputFiles(boost::ptr_vector<InputFile> &inFiles, const po::variables_map &vm)
{
    if (vm.count(Option::inputFile))
    {
        vector<string> inFilenames = vm[Option::inputFile].as<vector<string> >();
        BOOST_FOREACH(const string &filename, inFilenames)
        {
            inFiles.push_back(new InputFile(filename));
        }
    }
    else
    {
        inFiles.push_back(new InputFile());
    }
}

template<typename InputIterator, typename OutputIterator>
static OutputIterator loadInputSplats(InputIterator first, InputIterator last, OutputIterator out, float smooth)
{
    for (InputIterator in = first; in != last; ++in)
    {
        try
        {
            PLY::Reader reader(in->buffer);
            reader.addBuilder("vertex", SplatBuilder(smooth));
            reader.readHeader();
            PLY::ElementRangeReader<SplatBuilder> &rangeReader = reader.skipTo<SplatBuilder>("vertex");
            copy(rangeReader.begin(), rangeReader.end(), out);
        }
        catch (PLY::FormatError &e)
        {
            throw PLY::FormatError(in->filename + ": " + e.what());
        }
    }
    return out;
}

template<typename OutputIterator>
static OutputIterator loadInputSplats(const po::variables_map &vm, OutputIterator out, float smooth)
{
    boost::ptr_vector<InputFile> inFiles;
    makeInputFiles(inFiles, vm);
    return loadInputSplats(inFiles.begin(), inFiles.end(), out, smooth);
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

    const float dir[3][3] = { {spacing, 0, 0}, {0, spacing, 0}, {0, 0, spacing} };
    int extents[3][2];
    for (unsigned int i = 0; i < 3; i++)
    {
        float l = (bboxMin[i] - low[i]) / spacing;
        float h = (bboxMax[i] - low[i]) / spacing;
        extents[i][0] = RoundDown::convert(l);
        extents[i][1] = RoundUp::convert(h);
    }
    return Grid(low, dir[0], dir[1], dir[2],
                extents[0][0], extents[0][1], extents[1][0], extents[1][1], extents[2][0], extents[2][1]);
}

class OutputFunctor
{
private:
    vector<cl_float3> &hVertices;
    vector<boost::array<cl_uint, 3> > &hIndices;

public:
    OutputFunctor(vector<cl_float3> &hVertices, vector<boost::array<cl_uint, 3> > &hIndices)
        : hVertices(hVertices), hIndices(hIndices) {}

    void operator()(const cl::CommandQueue &queue,
                    const cl::Buffer &vertices,
                    const cl::Buffer &indices,
                    std::size_t numVertices,
                    std::size_t numIndices,
                    cl::Event *event) const;
};

void OutputFunctor::operator()(const cl::CommandQueue &queue,
                               const cl::Buffer &vertices,
                               const cl::Buffer &indices,
                               std::size_t numVertices,
                               std::size_t numIndices,
                               cl::Event *event) const
{
    cl::Event last;
    std::vector<cl::Event> wait(1);

    std::size_t oldVertices = hVertices.size();
    std::size_t oldIndices = hIndices.size();
    hVertices.resize(oldVertices + numVertices);
    hIndices.resize(oldIndices + numIndices / 3);
    queue.enqueueReadBuffer(vertices, CL_FALSE, 0, numVertices * sizeof(cl_float3), &hVertices[oldVertices],
                            NULL, &last);
    wait[0] = last;
    queue.enqueueReadBuffer(indices, CL_FALSE, 0, numIndices * sizeof(cl_uint), &hIndices[oldIndices],
                            &wait, &last);
    if (event != NULL)
        *event = last;
}

static void run(const cl::Context &context, const cl::Device &device, streambuf *out, const po::variables_map &vm)
{
    const unsigned int subsampling = 2;
    const unsigned int maxLevels = 8;
    const unsigned int maxBlock = 1U << (maxLevels + subsampling - 1);
    const unsigned int maxCells = maxBlock - 1;

    float spacing = vm[Option::fitGrid].as<double>();
    float smooth = vm[Option::fitSmooth].as<double>();
    vector<Splat> splats;
    loadInputSplats(vm, back_inserter(splats), smooth);
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

    std::vector<cl_float3> hVertices;
    std::vector<boost::array<cl_uint, 3> > hIndices;
    OutputFunctor output(hVertices, hIndices);

    /* TODO: partition splats */
    for (unsigned int bz = 0; bz <= cells[2]; bz += maxCells)
        for (unsigned int by = 0; by <= cells[1]; by += maxCells)
            for (unsigned int bx = 0; bx <= cells[0]; bx += maxCells)
            {
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
                    marching.generate(queue, input, output, sub, hVertices.size(), NULL);
                    cout << "Process: " << timer.getElapsed() << endl;
                }
            }

    PLY::Writer writer(PLY::FILE_FORMAT_LITTLE_ENDIAN, out);
    writer.addElement(PLY::makeElementRangeWriter(
            hVertices.begin(), hVertices.end(), hVertices.size(),
            PLY::VertexFetcher()));
    writer.addElement(PLY::makeElementRangeWriter(
            hIndices.begin(), hIndices.end(), hIndices.size(),
            PLY::TriangleFetcher()));
    writer.write();
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
        boost::scoped_ptr<OutputFile> outFile;
        if (vm.count(Option::outputFile))
        {
            const string &outFilename = vm[Option::outputFile].as<string>();
            outFile.reset(new OutputFile(outFilename));
        }
        else
            outFile.reset(new OutputFile());
        run(context, device, outFile->buffer, vm);
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
