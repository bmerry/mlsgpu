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
#include <boost/smart_ptr/shared_ptr.hpp>
#include <boost/smart_ptr/make_shared.hpp>
#include <boost/tr1/cmath.hpp>
#include <stxxl.h>
#include <nabo/nabo.h>
#include "../src/statistics.h"
#include "../src/splat_set.h"
#include "../src/fast_ply.h"
#include "../src/logging.h"
#include "../src/progress.h"
#include "../src/options.h"
#include "normals.h"
#include "normals_sweep.h"

namespace po = boost::program_options;

namespace Option
{
    static const char *axis() { return "axis"; }
};

void addSweepOptions(po::options_description &opts)
{
    opts.add_options()
        (Option::axis(), po::value<int>()->default_value(2), "Sort axis (0 = X, 1 = Y, 2 = Z");
}

class CompareSplats
{
private:
    int axis;

public:
    bool operator()(const Splat &a, const Splat &b) const
    {
        return a.position[axis] < b.position[axis];
    }

    Splat min_value() const
    {
        Splat ans;
        ans.position[axis] = -std::numeric_limits<float>::max();
        return ans;
    }

    Splat max_value() const
    {
        Splat ans;
        ans.position[axis] = std::numeric_limits<float>::max();
        return ans;
    }

    explicit CompareSplats(int axis) : axis(axis) {}
};

struct Slice
{
    std::vector<Splat> splats;
    Eigen::MatrixXf points;
    boost::scoped_ptr<Nabo::NNSearchF> tree;

    void addSplat(const Splat &s)
    {
        splats.push_back(s);
    }

    void makeTree()
    {
        points.resize(3, splats.size());
        for (std::size_t i = 0; i < splats.size(); i++)
        {
            for (int j = 0; j < 3; j++)
                points(j, i) = splats[i].position[j];
        }
        tree.reset(Nabo::NNSearchF::createKDTreeLinearHeap(points));
    }
};

struct Neighbors
{
    std::vector<float> dist2;
    std::vector<Eigen::Vector3f> elements;

    void merge(const Eigen::VectorXf &ndist2, const Eigen::VectorXi &nindices, const Eigen::MatrixXf &points, unsigned int K)
    {
        std::vector<float> mdist2;
        std::vector<Eigen::Vector3f> melements;

        unsigned int F = 0;
        while (F < K && (std::tr1::isfinite)(ndist2[F]))
            F++;
        unsigned int T = std::min(K, (unsigned int) (dist2.size() + F));
        mdist2.reserve(T);
        melements.reserve(T);

        unsigned int p = 0;
        unsigned int q = 0;
        while (mdist2.size() < T)
        {
            bool useOld;
            if (p < dist2.size() && q < F)
                useOld = dist2[p] < ndist2[q];
            else
                useOld = p < dist2.size();
            if (useOld)
            {
                mdist2.push_back(dist2[p]);
                melements.push_back(elements[p]);
                p++;
            }
            else
            {
                mdist2.push_back(ndist2[q]);
                Eigen::VectorXf col = points.col(nindices[q]);
                melements.push_back(col.head<3>());
                q++;
            }
        }
        mdist2.swap(dist2);
        melements.swap(elements);
    }
};

void processSlice(Slice *slice, const std::deque<boost::shared_ptr<Slice> > &active, unsigned int K, float maxRadius)
{
    Statistics::Registry &registry = Statistics::Registry::getInstance();
    Statistics::Variable &neighborStat = registry.getStatistic<Statistics::Variable>("neighbors");
    Statistics::Variable &qualityStat = registry.getStatistic<Statistics::Variable>("quality");
    Statistics::Variable &angleStat = registry.getStatistic<Statistics::Variable>("angle");
    Statistics::Timer timer("normal.worker.time");

    Eigen::VectorXf dist2(K);
    Eigen::VectorXi indices(K);
#pragma omp parallel for firstprivate(dist2, indices) schedule(static)
    for (std::size_t i = 0; i < slice->splats.size(); i++)
    {
        Neighbors nn;
        Eigen::VectorXf query = slice->points.col(i);
        for (std::size_t j = 0; j < active.size(); j++)
        {
            active[j]->tree->knn(query, indices, dist2, K, 0.0f, Nabo::NNSearchF::SORT_RESULTS, maxRadius);
            nn.merge(dist2, indices, active[j]->points, K);
        }

        std::vector<Eigen::Vector3f> neighbors;
        neighborStat.add(nn.elements.size() == std::size_t(K));

        if (nn.elements.size() == std::size_t(K))
        {
            float angle, quality;
            Eigen::Vector3f normal;
            normal = computeNormal(slice->splats[i], nn.elements, angle, quality);
            angleStat.add(angle);
            qualityStat.add(quality);
        }
    }
}

void runSweep(const po::variables_map &vm)
{
    SplatSet::FileSet splats;
    Timer latency;
    long long nSplats = 0;

    const std::vector<std::string> &names = vm[Option::inputFile()].as<std::vector<std::string> >();
    const FastPly::ReaderType readerType = vm[Option::reader()].as<Choice<FastPly::ReaderTypeWrapper> >();
    const int numNeighbors = vm[Option::neighbors()].as<int>();
    const float radius = vm[Option::radius()].as<double>();
    const int axis = vm[Option::axis()].as<int>();
    if (axis < 0 || axis > 2)
    {
        Log::log[Log::error] << "Invalid axis (should be 0, 1 or 2)\n";
        std::exit(1);
    }

    BOOST_FOREACH(const std::string &name, names)
    {
        std::auto_ptr<FastPly::ReaderBase> reader(FastPly::createReader(readerType, name, 1.0f));
        splats.addFile(reader.get());
        reader.release();
    }

    boost::scoped_ptr<SplatSet::SplatStream> splatStream(splats.makeSplatStream());
    stxxl::stream::sort<SplatSet::SplatStream, CompareSplats, 8 * 1024 * 1024> sortStream(*splatStream, CompareSplats(axis), 1024 * 1024 * 1024);
    std::deque<boost::shared_ptr<Slice> > active;

    ProgressDisplay progress(splats.maxSplats(), Log::log[Log::info]);

    while (!sortStream.empty())
    {
        // Fill up the next slice
        float z0 = sortStream->position[axis];
        boost::shared_ptr<Slice> curSlice(boost::make_shared<Slice>());
        while (!sortStream.empty() && sortStream->position[axis] < z0 + radius)
        {
            curSlice->addSplat(*sortStream);
            ++sortStream;
            ++nSplats;
        }
        curSlice->makeTree();
        progress += curSlice->splats.size();

        active.push_front(curSlice);
        if (active.size() >= 2)
        {
            processSlice(active[1].get(), active, numNeighbors, radius);
            if (active.size() == 3)
                active.pop_back();
        }
    }
    if (active.size() >= 2)
        processSlice(active[1].get(), active, numNeighbors, radius);

    Statistics::getStatistic<Statistics::Counter>("splats").add(nSplats);
    progress += splats.maxSplats() - nSplats; // ensures the progress bar completes
}
