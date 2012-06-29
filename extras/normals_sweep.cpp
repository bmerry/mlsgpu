/**
 * @file
 *
 * Implementation of the normals tool using a sweep plane.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif

#include <cstdlib>
#include <limits>
#include <deque>
#include <algorithm>
#include <memory>
#include <boost/program_options.hpp>
#include <boost/foreach.hpp>
#include <boost/smart_ptr/scoped_ptr.hpp>
#include <stxxl.h>
#include "../src/statistics.h"
#include "../src/splat_set.h"
#include "../src/fast_ply.h"
#include "../src/logging.h"
#include "../src/progress.h"
#include "../src/options.h"
#include "normals.h"
#include "normals_sweep.h"

namespace po = boost::program_options;

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

void runSweep(const po::variables_map &vm)
{
    SplatSet::FileSet splats;
    Timer latency;
    long long nSplats = 0;
    std::size_t maxActive = 0;

    const std::vector<std::string> &names = vm[Option::inputFile()].as<std::vector<std::string> >();
    const FastPly::ReaderType readerType = vm[Option::reader()].as<Choice<FastPly::ReaderTypeWrapper> >();
    const float radius = vm[Option::radius()].as<double>();
    const float window = 2.0f * radius;

    BOOST_FOREACH(const std::string &name, names)
    {
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

    Statistics::getStatistic<Statistics::Counter>("splats").add(nSplats);
    Statistics::getStatistic<Statistics::Counter>("active.max").add(maxActive);
}
