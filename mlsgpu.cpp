/**
 * @file
 */

#ifndef __CL_ENABLE_EXCEPTIONS
#define __CL_ENABLE_EXCEPTIONS
#endif

#include <boost/program_options.hpp>
#include <boost/lexical_cast.hpp>
#include <iostream>
#include <map>
#include "src/clh.h"
#include "src/logging.h"
#include "src/timer.h"

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
    return 0;
}
