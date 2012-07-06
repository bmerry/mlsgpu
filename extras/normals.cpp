/**
 * @file
 *
 * Computes improved normals for samples based on either bucketing or a sweep plane.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif

#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <memory>
#include <iterator>
#include <cmath>
#include <boost/program_options.hpp>
#include <boost/io/ios_state.hpp>
#include <boost/exception/all.hpp>
#include <boost/foreach.hpp>
#include <boost/smart_ptr/scoped_ptr.hpp>
#include <boost/smart_ptr/shared_ptr.hpp>
#include <boost/smart_ptr/make_shared.hpp>
#include <boost/ref.hpp>
#include <stxxl.h>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include "../src/statistics.h"
#include "../src/fast_ply.h"
#include "../src/logging.h"
#include "../src/options.h"
#include "../src/provenance.h"
#include "../src/decache.h"
#include "../src/splat_set.h"
#include "normals.h"
#include "normals_bucket.h"
#include "normals_sweep.h"

namespace po = boost::program_options;

enum Mode
{
    MODE_BUCKET,
    MODE_SWEEP,
    MODE_SLICE
};

class ModeWrapper
{
public:
    typedef Mode type;
    static std::map<std::string, Mode> getNameMap()
    {
        std::map<std::string, Mode> nameMap;
        nameMap["bucket"] = MODE_BUCKET;
        nameMap["sweep"] = MODE_SWEEP;
        nameMap["slice"] = MODE_SLICE;
        return nameMap;
    }
};

static void addCommonOptions(po::options_description &opts)
{
    opts.add_options()
        ("help,h",                  "Show help")
        ("quiet,q",                 "Do not show informational messages")
        (Option::bufferSize(),      po::value<std::size_t>()->default_value(SplatSet::FileSet::DEFAULT_BUFFER_SIZE), "File reader buffer size")
        (Option::reader(),          po::value<Choice<FastPly::ReaderTypeWrapper> >()->default_value(FastPly::SYSCALL_READER), "File reader class (mmap | syscall)");
        (Option::debug(),           "Show debug messages");
}

static void addSolveOptions(po::options_description &opts)
{
    po::options_description solve("Solver options");
    solve.add_options()
        (Option::maxHostSplats(),   po::value<std::size_t>()->default_value(10000000), "Maximum splats per bin/slice")
        (Option::radius(),          po::value<double>()->default_value(100),  "Maximum radius to search")
        (Option::neighbors(),       po::value<int>()->default_value(16),      "Neighbors to find")
        (Option::mode(),            po::value<Choice<ModeWrapper> >()->default_value(MODE_BUCKET), "Out-of-core mode (bucket | sweep)");
    opts.add(solve);
}

static void addStatisticsOptions(po::options_description &opts)
{
    po::options_description statistics("Statistics options");
    statistics.add_options()
        (Option::statistics(),     "Print information about internal statistics")
        (Option::statisticsFile(), po::value<std::string>(), "Direct statistics to file instead of stdout (implies --statistics)");
    opts.add(statistics);
}

std::string makeOptions(const po::variables_map &vm)
{
    std::ostringstream opts;
    for (po::variables_map::const_iterator i = vm.begin(); i != vm.end(); ++i)
    {
        if (i->first == Option::inputFile())
            continue; // these are not output because some programs choke
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
            else if (value.type() == typeid(Choice<FastPly::ReaderTypeWrapper>))
                opts << param.as<Choice<FastPly::ReaderTypeWrapper> >();
            else if (value.type() == typeid(Choice<ModeWrapper>))
                opts << param.as<Choice<ModeWrapper> >();
            else
                assert(!"Unhandled parameter type");
        }
    }
    return opts.str();
}

void writeStatistics(const boost::program_options::variables_map &vm, bool force = false)
{
    if (force || vm.count(Option::statistics()) || vm.count(Option::statisticsFile()))
    {
        std::ostream *out;
        std::ofstream outf;
        if (vm.count(Option::statisticsFile()))
        {
            const std::string &name = vm[Option::statisticsFile()].as<std::string>();
            outf.open(name.c_str());
            out = &outf;
        }
        else
        {
            out = &std::cout;
        }

        boost::io::ios_exception_saver saver(*out);
        out->exceptions(std::ios::failbit | std::ios::badbit);
        out->precision(9);
        *out << "normals version: " << provenanceVersion() << '\n';
        *out << "normals variant: " << provenanceVariant() << '\n';
        *out << "normals options:" << makeOptions(vm) << '\n';
        *out << Statistics::Registry::getInstance();
        *out << *stxxl::stats::get_instance();
    }
}

static void usage(std::ostream &o, const po::options_description desc)
{
    o << "Usage: normals [options] input.ply [input.ply...]\n\n";
    o << desc;
}

static po::variables_map processOptions(int argc, char **argv)
{
    po::positional_options_description positional;
    positional.add(Option::inputFile(), -1);

    po::options_description desc("General options");
    addCommonOptions(desc);
    addSolveOptions(desc);
    addBucketOptions(desc);
    addSweepOptions(desc);
    addStatisticsOptions(desc);

    po::options_description hidden("Hidden options");
    hidden.add_options()
        (Option::inputFile(), po::value<std::vector<std::string> >()->composing(), "input files");

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

        if (vm.count(Option::help()))
        {
            usage(std::cout, desc);
            std::exit(0);
        }
        /* Using ->required() on the option gives an unhelpful message */
        if (!vm.count(Option::inputFile()))
        {
            std::cerr << "At least one input file must be specified.\n\n";
            usage(std::cerr, desc);
            std::exit(1);
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

void NormalStats::computeNormal(
    const Splat &s,
    const std::vector<Eigen::Vector3f> &neighbors,
    unsigned int K)
{
    bool full = (neighbors.size() == K);
    if (!full)
        outlierStat.add(1);
    splatsStat.add(1);
    if (full)
    {
        Eigen::Vector3f oldNormal(s.normal[0], s.normal[1], s.normal[2]);
        oldNormal.normalize();

        Eigen::Vector3f centroid;
        centroid.setZero();
        for (std::size_t k = 0; k < neighbors.size(); k++)
        {
            centroid += neighbors[k];
        }
        centroid /= neighbors.size();

        Eigen::Matrix3f cov;
        cov.setZero();
        for (std::size_t k = 0; k < neighbors.size(); k++)
        {
            Eigen::Vector3f delta = neighbors[k] - centroid;
            cov += delta * delta.transpose();
        }
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(cov);
        Eigen::Vector3f normal = solver.eigenvectors().col(0);
        float quality = 1.0f - solver.eigenvalues()[0] / solver.eigenvalues()[1];
        float dot = normal.dot(oldNormal);
        if (dot < 0.0f)
        {
            normal = -normal;
            dot = -dot;
        }
        float angle = std::acos(std::min(dot, 1.0f)) * 57.2957795130823f;

        qualityStat.add(quality);
        angleStat.add(angle);
    }
}

static void run(const po::variables_map &vm)
{
    const std::vector<std::string> &names = vm[Option::inputFile()].as<std::vector<std::string> >();
    BOOST_FOREACH(const std::string &name, names)
    {
        decache(name);
    }

    Statistics::Timer timer("run.time");

    Mode mode = vm[Option::mode()].as<Choice<ModeWrapper> >();
    switch (mode)
    {
    case MODE_BUCKET:
        runBucket(vm);
        break;
    case MODE_SWEEP:
        runSweep(vm, true);
        break;
    case MODE_SLICE:
        runSweep(vm, false);
        break;
    }
}

static void reportException(std::exception &e)
{
    std::cerr << '\n';

    std::string *file_name = boost::get_error_info<boost::errinfo_file_name>(e);
    if (file_name != NULL)
        std::cerr << *file_name << ": ";
    std::cerr << e.what() << std::endl;
}

int main(int argc, char **argv)
{
    Log::log.setLevel(Log::info);

    po::variables_map vm = processOptions(argc, argv);
    if (vm.count(Option::quiet()))
        Log::log.setLevel(Log::warn);
    else if (vm.count(Option::debug()))
        Log::log.setLevel(Log::debug);

    try
    {
        run(vm);
        writeStatistics(vm);
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
}
