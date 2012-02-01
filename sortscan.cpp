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
#include <boost/io/ios_state.hpp>
#include <tr1/unordered_map>
#include <iostream>
#include <map>
#include <vector>
#include <algorithm>
#include "src/misc.h"
#include "src/logging.h"
#include "src/timer.h"
#include "src/fast_ply.h"
#include "src/ply.h"
#include "src/ply_mesh.h"
#include "src/splat.h"
#include "src/grid.h"
#include "src/bucket.h"
#include "src/provenance.h"
#include "src/statistics.h"
#include "src/collection.h"

namespace po = boost::program_options;
using namespace std;

namespace Option
{
    const char * const help = "help";
    const char * const quiet = "quiet";
    const char * const debug = "debug";
    const char * const responseFile = "response-file";

    const char * const inputFile = "input-file";
    const char * const outputFile = "output-file";

    const char * const statistics = "statistics";
    const char * const statisticsFile = "statistics-file";
};

static void addCommonOptions(po::options_description &opts)
{
    opts.add_options()
        ("help,h",                "Show help")
        ("quiet,q",               "Do not show informational messages")
        (Option::debug,           "Show debug messages")
        (Option::responseFile,    "Read options from file");
}

static void addStatisticsOptions(po::options_description &opts)
{
    po::options_description statistics("Statistics options");
    statistics.add_options()
        (Option::statistics,                          "Print information about internal statistics")
        (Option::statisticsFile, po::value<string>(), "Direct statistics to file instead of stdout (implies --statistics)");
    opts.add(statistics);
}

string makeOptions(const po::variables_map &vm)
{
    ostringstream opts;
    for (po::variables_map::const_iterator i = vm.begin(); i != vm.end(); ++i)
    {
        if (i->first == Option::inputFile)
            continue; // these are output last
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
            else
                assert(!"Unhandled parameter type");
        }
    }
    BOOST_FOREACH(const string &j, vm[Option::inputFile].as<vector<string> >())
    {
        opts << ' ' << j;
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
    }
}

static void usage(ostream &o, const po::options_description desc)
{
    o << "Usage: sortscan [options] -o output.ply input.ply [input.ply...]\n\n";
    o << desc;
}

static po::variables_map processOptions(int argc, char **argv)
{
    po::positional_options_description positional;
    positional.add(Option::inputFile, -1);

    po::options_description desc("General options");
    addCommonOptions(desc);
    addStatisticsOptions(desc);
    desc.add_options()
        ("output-file,o",   po::value<string>()->required(), "output file");

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

static void run(const string &out, const po::variables_map &vm)
{
    boost::ptr_vector<FastPly::Reader> files;
    prepareInputs(files, vm, 1.0f);

    Grid grid;
    typedef StxxlVectorCollection<Splat>::vector_type SplatVector;
    SplatVector splatData;
    try
    {
        Bucket::loadSplats(files, 1.0f, true, splatData, grid);
    }
    catch (std::length_error &e)
    {
        cerr << "At least one input point is required.\n";
        exit(1);
    }
    files.clear();

    filebuf outf;
    outf.open(out.c_str(), ios::out);
    PLY::Writer writer(PLY::FILE_FORMAT_LITTLE_ENDIAN, &outf);
    writer.addElement(makeElementRangeWriter(splatData.begin(), splatData.end(), splatData.size(), PLY::SplatFetcher()));
    writer.write();
    writeStatistics(vm);
}

int main(int argc, char **argv)
{
    Log::log.setLevel(Log::debug);

    po::variables_map vm = processOptions(argc, argv);
    if (vm.count(Option::quiet))
        Log::log.setLevel(Log::warn);
    else if (vm.count(Option::debug))
        Log::log.setLevel(Log::debug);

    try
    {
        run(vm[Option::outputFile].as<string>(), vm);
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
