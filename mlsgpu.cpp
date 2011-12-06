/**
 * @file
 */

#ifndef __CL_ENABLE_EXCEPTIONS
#define __CL_ENABLE_EXCEPTIONS
#endif

#include <boost/program_options.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/foreach.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/smart_ptr/scoped_ptr.hpp>
#include <iostream>
#include <map>
#include <vector>
#include "src/clh.h"
#include "src/logging.h"
#include "src/timer.h"
#include "src/ply.h"
#include "src/splat.h"
#include "src/files.h"

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
static OutputIterator loadInputSplats(InputIterator first, InputIterator last, OutputIterator out)
{
    for (InputIterator in = first; in != last; ++in)
    {
        try
        {
            PLY::Reader reader(in->buffer);
            reader.addBuilder("vertex", SplatBuilder());
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
static OutputIterator loadInputSplats(const po::variables_map &vm, OutputIterator out)
{
    boost::ptr_vector<InputFile> inFiles;
    makeInputFiles(inFiles, vm);
    return loadInputSplats(inFiles.begin(), inFiles.end(), out);
}

static void run(const po::variables_map &vm)
{
    vector<Splat> splats;
    loadInputSplats(vm, back_inserter(splats));
}

static void benchmarking(const cl::Context &context, const cl::Device &device)
{
    cl::CommandQueue queue(context, device);

    {
        // Benchmark copies
        const size_t elems = 1 << 21;
        const unsigned int workGroupSize = 128;
        const unsigned int iterations = 2;
        typedef cl_uint2 element_t;

        cl::Buffer in(context, CL_MEM_READ_WRITE, elems * sizeof(element_t));
        cl::Buffer out(context, CL_MEM_READ_WRITE, elems * sizeof(element_t));

        map<string, string> defines;
        defines["ELEMENT_T"] = "uint2";
        defines["WORK_GROUP_SIZE"] = boost::lexical_cast<string>(workGroupSize);
        defines["ITERATIONS"] = boost::lexical_cast<string>(iterations);
        cl::Program copyProgram = CLH::build(context, "kernels/copy.cl", defines);
        cl::Kernel copyKernel(copyProgram, "copy");
        copyKernel.setArg(0, out);
        copyKernel.setArg(1, in);
        copyKernel.setArg(2, (cl_uint) elems);

        const size_t tile = iterations * workGroupSize;
        const size_t groups = (elems + tile - 1) / tile;

        // warmup
        queue.enqueueNDRangeKernel(copyKernel, cl::NullRange,
                                   cl::NDRange(groups * workGroupSize), cl::NDRange(workGroupSize));
        queue.finish();

        Timer timer;
        queue.enqueueNDRangeKernel(copyKernel, cl::NullRange,
                                   cl::NDRange(groups * workGroupSize), cl::NDRange(workGroupSize));
        queue.finish();
        double rate = elems * 2 * sizeof(element_t) / timer.getElapsed();
        cout << rate * 1e-9 << " GB/s\n";
    }
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
    benchmarking(context, device);

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
        run(vm);
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

    return 0;
}
