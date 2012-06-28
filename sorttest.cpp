/**
 * @file
 *
 * Artificial benchmark that sorts input PLY files by one coordinate and
 * measures the maximum active set.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif

#include <iostream>
#include <cstdlib>
#include <limits>
#include <deque>
#include <algorithm>
#include <memory>
#include <boost/program_options.hpp>
#include <boost/io/ios_state.hpp>
#include <boost/exception/all.hpp>
#include <boost/foreach.hpp>
#include <boost/smart_ptr/scoped_ptr.hpp>
#include <stxxl.h>
#include "src/statistics.h"
#include "src/splat_set.h"
#include "src/fast_ply.h"
#include "src/logging.h"
#include "src/progress.h"
#include "src/options.h"
#include "src/provenance.h"
#include "src/decache.h"

namespace po = boost::program_options;

namespace Option
{
    const char * const help = "help";
    const char * const quiet = "quiet";
    const char * const debug = "debug";

    const char * const window = "window";

    const char * const inputFile = "input-file";

    const char * const statistics = "statistics";
    const char * const statisticsFile = "statistics-file";

    const char * const reader = "reader";
};

static void addCommonOptions(po::options_description &opts)
{
    opts.add_options()
        ("help,h",                "Show help")
        ("quiet,q",               "Do not show informational messages")
        (Option::window, po::value<double>()->default_value(0.1), "Window thickness")
        (Option::debug,           "Show debug messages");
}

static void addStatisticsOptions(po::options_description &opts)
{
    po::options_description statistics("Statistics options");
    statistics.add_options()
        (Option::statistics,                          "Print information about internal statistics")
        (Option::statisticsFile, po::value<std::string>(), "Direct statistics to file instead of stdout (implies --statistics)");
    opts.add(statistics);
}

static void addAdvancedOptions(po::options_description &opts)
{
    po::options_description advanced("Advanced options");
    advanced.add_options()
        (Option::reader,       po::value<Choice<FastPly::ReaderTypeWrapper> >()->default_value(FastPly::SYSCALL_READER), "File reader class (mmap | syscall)");
    opts.add(advanced);
}

std::string makeOptions(const po::variables_map &vm)
{
    std::ostringstream opts;
    for (po::variables_map::const_iterator i = vm.begin(); i != vm.end(); ++i)
    {
        if (i->first == Option::inputFile)
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
        std::ostream *out;
        std::ofstream outf;
        if (vm.count(Option::statisticsFile))
        {
            const std::string &name = vm[Option::statisticsFile].as<std::string>();
            outf.open(name.c_str());
            out = &outf;
        }
        else
        {
            out = &std::cout;
        }

        boost::io::ios_exception_saver saver(*out);
        out->exceptions(std::ios::failbit | std::ios::badbit);
        *out << "sorttest version: " << provenanceVersion() << '\n';
        *out << "sorttest variant: " << provenanceVariant() << '\n';
        *out << "sorttest options:" << makeOptions(vm) << '\n';
        *out << Statistics::Registry::getInstance();
        *out << *stxxl::stats::get_instance();
    }
}

static void usage(std::ostream &o, const po::options_description desc)
{
    o << "Usage: sorttest [options] input.ply [input.ply...]\n\n";
    o << desc;
}

static po::variables_map processOptions(int argc, char **argv)
{
    po::positional_options_description positional;
    positional.add(Option::inputFile, -1);

    po::options_description desc("General options");
    addCommonOptions(desc);
    addStatisticsOptions(desc);
    addAdvancedOptions(desc);

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

        return vm;
    }
    catch (po::error &e)
    {
        std::cerr << e.what() << "\n\n";
        usage(std::cerr, desc);
        std::exit(1);
    }
}

struct CompareSplats
{
    bool operator()(const Splat &a, const Splat &b) const
    {
        return a.position[2] < b.position[2];
    }

    Splat min_value() const
    {
        Splat ans;
        ans.position[2] = -std::numeric_limits<float>::max();
        return ans;
    }

    Splat max_value() const
    {
        Splat ans;
        ans.position[2] = std::numeric_limits<float>::max();
        return ans;
    }
};

static void run(const po::variables_map &vm)
{
    SplatSet::FileSet splats;
    Timer total;
    Timer latency;
    long long nSplats = 0;
    std::size_t maxActive = 0;

    const std::vector<std::string> &names = vm[Option::inputFile].as<std::vector<std::string> >();
    const FastPly::ReaderType readerType = vm[Option::reader].as<Choice<FastPly::ReaderTypeWrapper> >();
    const float window = vm[Option::window].as<double>();

    BOOST_FOREACH(const std::string &name, names)
    {
        decache(name);
        std::auto_ptr<FastPly::ReaderBase> reader(FastPly::createReader(readerType, name, 1.0f));
        splats.addFile(reader.get());
        reader.release();
    }

    boost::scoped_ptr<SplatSet::SplatStream> splatStream(splats.makeSplatStream());
    stxxl::stream::sort<SplatSet::SplatStream, CompareSplats, 8 * 1024 * 1024> sortStream(*splatStream, CompareSplats(), 1024 * 1024 * 1024);
    std::deque<Splat> active;

    while (!sortStream.empty())
    {
        Splat s = *sortStream;

        active.push_back(s);
        while (active.front().position[2] < s.position[2] - window)
            active.pop_front();
        maxActive = std::max(maxActive, active.size());

        if (nSplats == 0)
            Statistics::getStatistic<Statistics::Variable>("latency").add(latency.getElapsed());
        ++nSplats;
        ++sortStream;
    }

    Statistics::getStatistic<Statistics::Variable>("time").add(total.getElapsed());
    Statistics::getStatistic<Statistics::Counter>("splats").add(nSplats);
    Statistics::getStatistic<Statistics::Counter>("active.max").add(maxActive);
    writeStatistics(vm);
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
    if (vm.count(Option::quiet))
        Log::log.setLevel(Log::warn);
    else if (vm.count(Option::debug))
        Log::log.setLevel(Log::debug);

    try
    {
        run(vm);
        // TODO: report sorting rate
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
