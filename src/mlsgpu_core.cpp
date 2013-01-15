/**
 * @file
 *
 * Utility functions only used in the main program.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif

#ifndef __CL_ENABLE_EXCEPTIONS
# define __CL_ENABLE_EXCEPTIONS
#endif

#include <boost/program_options.hpp>
#include <boost/foreach.hpp>
#include <boost/io/ios_state.hpp>
#include <boost/exception/all.hpp>
#include <boost/system/error_code.hpp>
#include <boost/filesystem.hpp>
#include <memory>
#include <string>
#include <iterator>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cassert>
#include <stxxl.h>
#include "mlsgpu_core.h"
#include "options.h"
#include "mls.h"
#include "mesher.h"
#include "fast_ply.h"
#include "clh.h"
#include "statistics.h"
#include "statistics_cl.h"
#include "logging.h"
#include "provenance.h"
#include "marching.h"
#include "splat_tree_cl.h"
#include "workers.h"
#include "splat_set.h"
#include "decache.h"

namespace po = boost::program_options;

static void addCommonOptions(po::options_description &opts)
{
    opts.add_options()
        ("help,h",                "Show help")
        ("quiet,q",               "Do not show informational messages")
        (Option::debug,           "Show debug messages")
        (Option::responseFile,    po::value<std::string>(), "Read options from file")
        (Option::tmpDir,          po::value<boost::filesystem::path::string_type>(), "Directory to store temporary files");
}

static void addFitOptions(po::options_description &opts)
{
    opts.add_options()
        (Option::fitSmooth,       po::value<double>()->default_value(4.0),  "Smoothing factor")
        (Option::maxRadius,       po::value<double>(),                      "Limit influence radii")
        (Option::fitGrid,         po::value<double>()->default_value(0.01), "Spacing of grid cells")
        (Option::fitPrune,        po::value<double>()->default_value(0.02), "Minimum fraction of vertices per component")
        (Option::fitBoundaryLimit, po::value<double>()->default_value(1.0), "Tuning factor for boundary detection")
        (Option::fitShape,        po::value<Choice<MlsShapeWrapper> >()->default_value(MLS_SHAPE_SPHERE),
                                                                            "Model shape (sphere | plane)");
}

static void addStatisticsOptions(po::options_description &opts)
{
    po::options_description statistics("Statistics options");
    statistics.add_options()
        (Option::statistics,                          "Print information about internal statistics")
        (Option::statisticsFile, po::value<std::string>(), "Direct statistics to file instead of stdout (implies --statistics)")
        (Option::statisticsCL,                             "Collect timings for OpenCL commands")
        (Option::timeplot, po::value<std::string>(),       "Write timing data to file");
    opts.add(statistics);
}

static void addAdvancedOptions(po::options_description &opts)
{
    po::options_description advanced("Advanced options");
    advanced.add_options()
        (Option::levels,       po::value<int>()->default_value(6), "Levels in octree")
        (Option::subsampling,  po::value<int>()->default_value(3), "Subsampling of octree")
        (Option::maxDeviceSplats, po::value<int>()->default_value(1000000), "Maximum splats per block on the device")
        (Option::maxHostSplats, po::value<std::size_t>()->default_value(10000000), "Maximum splats per block on the CPU")
        (Option::memHostSplats, po::value<std::size_t>(),          "Total splats kept on the CPU")
        (Option::maxSplit,     po::value<int>()->default_value(2097152), "Maximum fan-out in partitioning")
        (Option::bucketThreads, po::value<int>()->default_value(2), "Number of threads for bucketing splats")
        (Option::deviceThreads, po::value<int>()->default_value(1), "Number of threads per device for submitting OpenCL work")
        (Option::reader,       po::value<Choice<FastPly::ReaderTypeWrapper> >()->default_value(FastPly::SYSCALL_READER), "File reader class (mmap | syscall)")
        (Option::writer,       po::value<Choice<FastPly::WriterTypeWrapper> >()->default_value(FastPly::STREAM_WRITER), "File writer class (mmap | stream)")
        (Option::decache,      "Try to evict input files from OS cache for benchmarking");
    opts.add(advanced);
}

void usage(std::ostream &o, const po::options_description desc)
{
    o << "Usage: mlsgpu [options] -o output.ply input.ply [input.ply...]\n\n";
    o << desc;
}

po::variables_map processOptions(int argc, char **argv)
{
    // TODO: replace cerr with thrown exception
    po::positional_options_description positional;
    positional.add(Option::inputFile, -1);

    po::options_description desc("General options");
    addCommonOptions(desc);
    addFitOptions(desc);
    addStatisticsOptions(desc);
    addAdvancedOptions(desc);
    desc.add_options()
        ("output-file,o",   po::value<std::string>()->required(), "output file")
        (Option::split,     "split output across multiple files")
        (Option::splitSize, po::value<unsigned int>()->default_value(100), "approximate size of output chunks (MiB)");

    po::options_description clopts("OpenCL options");
    CLH::addOptions(clopts);
    desc.add(clopts);

    po::options_description hidden("Hidden options");
    hidden.add_options()
        (Option::inputFile, po::value<std::vector<std::string> >()->composing(), "input files");

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
            const std::string &fname = vm[Option::responseFile].as<std::string>();
            std::ifstream in(fname.c_str());
            if (!in)
            {
                Log::log[Log::warn] << "Could not open `" << fname << "', ignoring\n";
            }
            else
            {
                std::vector<std::string> args;
                std::copy(std::istream_iterator<std::string>(in),
                          std::istream_iterator<std::string>(), std::back_inserter(args));
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
            usage(std::cout, desc);
            std::exit(0);
        }
        /* Using ->required() on the option gives an unhelpful message */
        if (!vm.count(Option::inputFile))
        {
            std::cerr << "At least one input file must be specified.\n\n";
            usage(std::cerr, desc);
            std::exit(1);
        }

        if (vm.count(Option::statisticsCL))
        {
            Statistics::enableEventTiming();
        }
        if (vm.count(Option::tmpDir))
        {
            setTmpFileDir(vm[Option::tmpDir].as<boost::filesystem::path::string_type>());
        }

        return vm;
    }
    catch (po::error &e)
    {
        std::cerr << e.what() << "\n\n";
        usage(std::cerr, desc);
        std::exit(1);
    }
}

std::string makeOptions(const po::variables_map &vm)
{
    std::ostringstream opts;
    for (po::variables_map::const_iterator i = vm.begin(); i != vm.end(); ++i)
    {
        if (i->first == Option::inputFile)
            continue; // these are not output because some programs choke
        if (i->first == Option::responseFile)
            continue; // this is not relevant to reproducing the results
        const po::variable_value &param = i->second;
        const boost::any &value = param.value();
        if (param.empty()
            || (value.type() == typeid(std::string) && param.as<std::string>().empty()))
            opts << " --" << i->first;
        else if (value.type() == typeid(std::vector<std::string>))
        {
            BOOST_FOREACH(const std::string &j, param.as<std::vector<std::string> >())
            {
                opts << " --" << i->first << '=' << j;
            }
        }
        else
        {
            opts << " --" << i->first << '=';
            if (value.type() == typeid(std::string))
                opts << param.as<std::string>();
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

void writeStatistics(const po::variables_map &vm, bool force)
{
    if (force || vm.count(Option::statistics) || vm.count(Option::statisticsFile))
    {
        std::string name = "<stdout>";
        try
        {
            std::ostream *out;
            std::ofstream outf;
            if (vm.count(Option::statisticsFile))
            {
                name = vm[Option::statisticsFile].as<std::string>();
                outf.open(name.c_str());
                out = &outf;
            }
            else
            {
                out = &std::cout;
            }

            boost::io::ios_exception_saver saver(*out);
            out->exceptions(std::ios::failbit | std::ios::badbit);
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
        catch (std::ios::failure &e)
        {
            throw boost::enable_error_info(e)
                << boost::errinfo_file_name(name)
                << boost::errinfo_errno(errno);
        }
    }
}

void validateOptions(const po::variables_map &vm)
{
    const int levels = vm[Option::levels].as<int>();
    const int subsampling = vm[Option::subsampling].as<int>();
    const std::size_t maxDeviceSplats = vm[Option::maxDeviceSplats].as<int>();
    const std::size_t maxHostSplats = vm[Option::maxHostSplats].as<std::size_t>();
    const std::size_t memHostSplats = getMemHostSplats(vm);
    const std::size_t maxSplit = vm[Option::maxSplit].as<int>();
    const int bucketThreads = vm[Option::bucketThreads].as<int>();
    const int deviceThreads = vm[Option::deviceThreads].as<int>();
    const double pruneThreshold = vm[Option::fitPrune].as<double>();

    int maxLevels = std::min(
            std::size_t(Marching::MAX_DIMENSION_LOG2 + 1),
            std::size_t(SplatTreeCL::MAX_LEVELS));
    if (levels < 1 || levels > maxLevels)
    {
        std::ostringstream msg;
        msg << "Value of --levels must be in the range 1 to " << maxLevels;
        throw invalid_option(msg.str());
    }
    if (subsampling < MlsFunctor::subsamplingMin)
    {
        std::ostringstream msg;
        msg << "Value of --subsampling must be at least " << MlsFunctor::subsamplingMin;
        throw invalid_option(msg.str());
    }
    if (maxDeviceSplats < 1)
        throw invalid_option("Value of --max-device-splats must be positive");
    if (maxHostSplats < maxDeviceSplats)
        throw invalid_option("Value of --max-host-splats must be at least that of --max-device-splats");
    if (memHostSplats < maxHostSplats)
        throw invalid_option("Value of --mem-host-splats must be at least that of --max-host-splats");
    if (maxSplit < 8)
        throw invalid_option("Value of --max-split must be at least 8");
    if (subsampling > Marching::MAX_DIMENSION_LOG2 + 1 - levels)
        throw invalid_option("Sum of --subsampling and --levels is too large");
    const std::size_t treeVerts = std::size_t(1) << (subsampling + levels - 1);
    if (treeVerts < MlsFunctor::wgs[0] || treeVerts < MlsFunctor::wgs[1])
        throw invalid_option("Sum of --subsampling and --levels is too small");

    if (bucketThreads < 1)
        throw invalid_option("Value of --bucket-threads must be at least 1");
    if (deviceThreads < 1)
        throw invalid_option("Value of --device-threads must be at least 1");
    if (!(pruneThreshold >= 0.0 && pruneThreshold <= 1.0))
        throw invalid_option("Value of --fit-prune must be in [0, 1]");
}

void setLogLevel(const po::variables_map &vm)
{
    if (vm.count(Option::quiet))
        Log::log.setLevel(Log::warn);
    else if (vm.count(Option::debug))
        Log::log.setLevel(Log::debug);
    else
        Log::log.setLevel(Log::info);
}

int deviceWorkerSpare(const po::variables_map &vm)
{
    const int bucketThreads = vm[Option::bucketThreads].as<int>();
    return std::max(bucketThreads, 6);
}

std::size_t getMemHostSplats(const po::variables_map &vm)
{
    if (vm.count(Option::memHostSplats))
        return vm[Option::memHostSplats].as<std::size_t>();
    else
        return vm[Option::maxHostSplats].as<std::size_t>() * 4;
}

std::size_t meshMemory(const po::variables_map &vm)
{
    const int levels = vm[Option::levels].as<int>();
    const int subsampling = vm[Option::subsampling].as<int>();
    const Grid::size_type maxCells = (Grid::size_type(1U) << (levels + subsampling - 1)) - 1;
    return maxCells * maxCells * 2 * Marching::MAX_CELL_BYTES;
}

CLH::ResourceUsage resourceUsage(const po::variables_map &vm)
{
    const int levels = vm[Option::levels].as<int>();
    const int subsampling = vm[Option::subsampling].as<int>();
    const std::size_t maxDeviceSplats = vm[Option::maxDeviceSplats].as<int>();
    const int deviceThreads = vm[Option::deviceThreads].as<int>();
    const int deviceSpare = deviceWorkerSpare(vm);

    const Grid::size_type maxCells = (Grid::size_type(1U) << (levels + subsampling - 1)) - 1;
    // TODO: get rid of device parameter
    CLH::ResourceUsage totalUsage = DeviceWorkerGroup::resourceUsage(
        deviceThreads, deviceSpare, cl::Device(), maxDeviceSplats,
        maxCells, meshMemory(vm), levels);
    return totalUsage;
}

void validateDevice(const cl::Device &device, const CLH::ResourceUsage &totalUsage)
{
    const std::string deviceName = "OpenCL device `" + device.getInfo<CL_DEVICE_NAME>() + "'";
    Marching::validateDevice(device);
    SplatTreeCL::validateDevice(device);

    /* Check that we have enough memory on the device. This is no guarantee against OOM, but
     * we can at least turn down silly requests before wasting any time.
     */
    const std::size_t deviceTotalMemory = device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
    const std::size_t deviceMaxMemory = device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
    if (totalUsage.getMaxMemory() > deviceMaxMemory)
    {
        std::ostringstream msg;
        msg << "Arguments require an allocation of " << totalUsage.getMaxMemory() << ",\n"
            << "but only " << deviceMaxMemory << " is supported.\n"
            << "Try reducing --levels or increasing --subsampling.";
        throw CLH::invalid_device(device, msg.str());
    }
    if (totalUsage.getTotalMemory() > deviceTotalMemory)
    {
        std::ostringstream msg;
        msg << "Arguments require device memory of " << totalUsage.getTotalMemory() << ",\n"
            << "but only " << deviceTotalMemory << " available.\n"
            << "Try reducing --levels or --max-device-splats, or increasing --subsampling.";
        throw CLH::invalid_device(device, msg.str());
    }

    if (totalUsage.getTotalMemory() > deviceTotalMemory * 0.8)
    {
        Log::log[Log::warn] << "WARNING: More than 80% of the memory on " << deviceName << " will be used.\n";
    }
}

void prepareInputs(SplatSet::FileSet &files, const po::variables_map &vm, float smooth, float maxRadius)
{
    const std::vector<std::string> &names = vm[Option::inputFile].as<std::vector<std::string> >();
    std::vector<boost::filesystem::path> paths;
    BOOST_FOREACH(const std::string &name, names)
    {
        boost::filesystem::path base(name);
        if (is_directory(base))
        {
            boost::filesystem::directory_iterator it(base);
            while (it != boost::filesystem::directory_iterator())
            {
                if (it->path().extension() == ".ply" && !is_directory(it->status()))
                    paths.push_back(it->path());
                ++it;
            }
        }
        else
            paths.push_back(name);
    }

    const FastPly::ReaderType readerType = vm[Option::reader].as<Choice<FastPly::ReaderTypeWrapper> >();
    if (paths.size() > SplatSet::FileSet::maxFiles)
    {
        std::ostringstream msg;
        msg << "Too many input files (" << paths.size() << " > " << SplatSet::FileSet::maxFiles << ")";
        throw std::runtime_error(msg.str());
    }
    std::tr1::uint64_t totalSplats = 0;
    std::tr1::uint64_t totalBytes = 0;
    BOOST_FOREACH(const boost::filesystem::path &path, paths)
    {
        if (vm.count(Option::decache))
            decache(path.string());
        std::auto_ptr<FastPly::ReaderBase> reader(FastPly::createReader(readerType, path.string(), smooth, maxRadius));
        if (reader->size() > SplatSet::FileSet::maxFileSplats)
        {
            std::ostringstream msg;
            msg << "Too many samples in " << path << " ("
                << reader->size() << " > " << SplatSet::FileSet::maxFileSplats << ")";
            throw std::runtime_error(msg.str());
        }
        totalSplats += reader->size();
        totalBytes += reader->size() * reader->getVertexSize();
        files.addFile(reader.get());
        reader.release();
    }

    Statistics::getStatistic<Statistics::Counter>("files.scans").add(paths.size());
    Statistics::getStatistic<Statistics::Counter>("files.splats").add(totalSplats);
    Statistics::getStatistic<Statistics::Counter>("files.bytes").add(totalBytes);
}

void reportException(std::exception &e)
{
    std::cerr << '\n';

    std::string *file_name = boost::get_error_info<boost::errinfo_file_name>(e);
    int *err = boost::get_error_info<boost::errinfo_errno>(e);
    if (file_name != NULL)
        std::cerr << *file_name << ": ";
    if (err != NULL && *err != 0)
        std::cerr << boost::system::errc::make_error_code((boost::system::errc::errc_t) *err).message() << std::endl;
    else
        std::cerr << e.what() << std::endl;
}
