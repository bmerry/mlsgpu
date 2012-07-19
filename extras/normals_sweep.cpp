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
#include <boost/noncopyable.hpp>
#include <boost/program_options.hpp>
#include <boost/foreach.hpp>
#include <boost/smart_ptr/scoped_ptr.hpp>
#include <boost/smart_ptr/shared_ptr.hpp>
#include <boost/smart_ptr/make_shared.hpp>
#include <boost/tr1/cmath.hpp>
#include <boost/thread/future.hpp>
#include <stxxl.h>
#include <nabo/nabo.h>
#include <sl/kdtree.hpp>
#include "../src/statistics.h"
#include "../src/splat_set.h"
#include "../src/fast_ply.h"
#include "../src/logging.h"
#include "../src/progress.h"
#include "../src/options.h"
#include "../src/worker_group.h"
#include "../src/tr1_cstdint.h"
#include "normals.h"
#include "normals_sweep.h"

namespace po = boost::program_options;

namespace Option
{
    static inline const char *axis() { return "axis"; }
};

void addSweepOptions(po::options_description &opts)
{
    po::options_description opts2("Sweep mode options");
    opts2.add_options()
        (Option::axis(), po::value<int>()->default_value(2), "Sort axis (0 = X, 1 = Y, 2 = Z)");
    opts.add(opts2);
}

namespace
{

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

struct Slice : public boost::noncopyable
{
    std::tr1::uint64_t index; ///< Sequential number of the slice
    Statistics::Peak<std::tr1::uint64_t> &activeStat;
    Statistics::Container::vector<Splat> splats;

    Eigen::MatrixXf points;
    boost::scoped_ptr<Nabo::NNSearchF> tree;
    boost::mutex treeMutex; ///< Guards @c tree and @c points
    boost::condition_variable treeCondition; ///< Signaled when @c makeTree completes

    float minCut, maxCut;

    void addSplat(const Splat &s)
    {
        splats.push_back(s);
    }

    void finishSplats()
    {
        activeStat += splats.size();
    }

    void makeTree()
    {
        boost::lock_guard<boost::mutex> lock(treeMutex);
        assert(!tree);

        points.resize(3, splats.size());
        for (std::size_t i = 0; i < splats.size(); i++)
        {
            for (int j = 0; j < 3; j++)
                points(j, i) = splats[i].position[j];
        }
        tree.reset(Nabo::NNSearchF::createKDTreeLinearHeap(points));
        treeCondition.notify_all();
    }

    void waitTree()
    {
        boost::unique_lock<boost::mutex> lock(treeMutex);
        while (!tree)
            treeCondition.wait(lock);
    }

    const Nabo::NNSearchF *getTree() const
    {
        assert(tree);
        return tree.get();
    }

    Slice() :
        activeStat(Statistics::getStatistic<Statistics::Peak<std::tr1::uint64_t> >("active.peak")),
        splats("mem.splats")
    {}

    ~Slice()
    {
        activeStat -= splats.size();
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
        while (F < ndist2.size() && (std::tr1::isfinite)(ndist2[F]))
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

struct NormalItem
{
    int axis;
    unsigned int K;
    float maxRadius;
    boost::shared_ptr<Slice> slice;
    std::size_t nactive;
    boost::shared_ptr<Slice> active[3];
    std::tr1::uint64_t needsTree; ///< Index of the first slice whose tree is incomplete
    ProgressDisplay *progress;
};

class NormalWorker : public NormalStats
{
public:
    void start() {}
    void stop() {}

    void operator()(int gen, NormalItem &item)
    {
        (void) gen;
        Statistics::Timer timer(computeStat);
        const int axis = item.axis;
        const unsigned int K = item.K;
        const float maxRadius = item.maxRadius;
        const std::size_t NS = item.nactive;
        const Slice * const slice = item.slice.get();
        const Slice * const active[3] =
        {
            item.active[0].get(),
            item.active[1].get(),
            item.active[2].get()
        };
        assert(NS >= 1 && NS <= 3);

        // Run tree generation for future slices
        for (std::size_t i = 0; i < NS; i++)
            if (item.active[i]->index >= item.needsTree)
                item.active[i]->makeTree();

        // Wait for tree generation from previous slices
        for (std::size_t i = 0; i < NS; i++)
            if (item.active[i]->index < item.needsTree)
                item.active[i]->waitTree();

        Eigen::VectorXf dist2(K);
        Eigen::VectorXi indices(K);
        for (std::size_t i = 0; i < slice->splats.size(); i++)
        {
            const Eigen::VectorXf query = slice->points.col(i);
            const float z = query[axis];

            std::pair<float, const Slice *> nslices[3];
            for (std::size_t j = 0; j < NS; j++)
            {
                nslices[j].second = active[j];
                if (active[j]->maxCut < z)
                    nslices[j].first = z - active[j]->maxCut;
                else if (active[j]->minCut > z)
                    nslices[j].first = active[j]->minCut - z;
                else
                    nslices[j].first = 0.0f;
            }
            std::sort(nslices, nslices + NS);

            Neighbors nn;
            for (std::size_t j = 0; j < NS; j++)
            {
                float d2 = nslices[j].first * nslices[j].first;
                float limit = maxRadius;
                std::size_t skip = 0;
                if (nn.dist2.size() == K)
                    limit = std::sqrt(nn.dist2.back());
                while (skip < nn.dist2.size() && nn.dist2[skip] < d2)
                    skip++;

                if (skip < K && nslices[j].first <= limit)
                {
                    nslices[j].second->getTree()->knn(query, indices, dist2, K - skip, 0.0f, Nabo::NNSearchF::SORT_RESULTS, limit);
                    nn.merge(dist2, indices, nslices[j].second->points, K);
                }
            }

            computeNormal(slice->splats[i], nn.elements, K);
        }
        if (item.progress != NULL)
            *item.progress += slice->splats.size();
        // Recover the memory as soon as possible
        item.slice.reset();
        for (int i = 0; i < 3; i++)
            item.active[i].reset();
    }
};

class NormalWorkerGroup : public WorkerGroup<NormalItem, int, NormalWorker, NormalWorkerGroup>
{
public:
    NormalWorkerGroup(std::size_t numWorkers, std::size_t spare)
        : WorkerGroup<NormalItem, int, NormalWorker, NormalWorkerGroup>(
            numWorkers, spare,
            Statistics::getStatistic<Statistics::Variable>("normal.worker.push"),
            Statistics::getStatistic<Statistics::Variable>("normal.worker.pop.first"),
            Statistics::getStatistic<Statistics::Variable>("normal.worker.pop"),
            Statistics::getStatistic<Statistics::Variable>("normal.worker.get"))
    {
        for (std::size_t i = 0; i < numWorkers; i++)
            addWorker(new NormalWorker);
        for (std::size_t i = 0; i < numWorkers + spare; i++)
            addPoolItem(boost::make_shared<NormalItem>());
    }
};

void processSlice(
    NormalWorkerGroup &outGroup,
    int axis, boost::shared_ptr<Slice> slice,
    const std::deque<boost::shared_ptr<Slice> > &active,
    std::tr1::uint64_t needsTree,
    unsigned int K, float maxRadius,
    ProgressDisplay *progress)
{
    boost::shared_ptr<NormalItem> item = outGroup.get();
    item->axis = axis;
    item->K = K;
    item->maxRadius = maxRadius;
    item->slice = slice;
    item->nactive = active.size();
    for (std::size_t i = 0; i < item->nactive; i++)
        item->active[i] = active[i];
    item->needsTree = needsTree;
    item->progress = progress;
    outGroup.push(0, item);
}

typedef stxxl::stream::sort<SplatSet::SplatStream, CompareSplats, 2 * 1024 * 1024> SortStream;

void runSweepDiscrete(SplatSet::SplatStream *splatStream, ProgressDisplay *progress,
                      bool compute, int axis, unsigned int K, float radius)
{
    Statistics::Variable &loadTime = Statistics::getStatistic<Statistics::Variable>("load.time");
    std::tr1::uint64_t nSplats = 0;
    Timer latency;

    SortStream sortStream(*splatStream, CompareSplats(axis), 1024 * 1024 * 1024);
    std::deque<boost::shared_ptr<Slice> > active;

    NormalWorkerGroup normalGroup(8, 2);
    normalGroup.producerStart(0);
    normalGroup.start();

    std::tr1::uint64_t sliceIdx = 0;
    std::tr1::uint64_t needsTree = 0;
    while (!sortStream.empty())
    {
        if (active.empty())
            Statistics::getStatistic<Statistics::Variable>("latency").add(latency.getElapsed());

        // Fill up the next slice
        float z0 = sortStream->position[axis];
        boost::shared_ptr<Slice> curSlice(boost::make_shared<Slice>());
        curSlice->minCut = z0;
        {
            Statistics::Timer loadTimer(loadTime);
            while (!sortStream.empty()
                   && (sortStream->position[axis] < z0 + radius))
            {
                curSlice->addSplat(*sortStream);
                ++sortStream;
                ++nSplats;
            }
            curSlice->finishSplats();
        }
        curSlice->maxCut = curSlice->splats.back().position[axis];
        curSlice->index = sliceIdx++;

        active.push_front(curSlice);
        if (active.size() >= 2)
        {
            if (compute)
                processSlice(normalGroup, axis, active[1], active, needsTree, K, radius, progress);
            needsTree = sliceIdx;
            if (active.size() == 3)
                active.pop_back();
        }
    }
    if (!active.empty() && compute)
        processSlice(normalGroup, axis, active.front(), active, needsTree, K, radius, progress);

    Statistics::getStatistic<Statistics::Counter>("slices").add(sliceIdx);

    Statistics::Timer spindownTimer("spindown.time");
    normalGroup.producerStop(0);
    normalGroup.stop();

    // ensures the progress bar completes even if there were non-finite splats
    if (progress != NULL)
        *progress += progress->expected_count() - progress->count();
}

class NormalCompute : public NormalStats
{
public:
    typedef sl::kdtree<3, float, Eigen::Vector3f> tree_type;

    explicit NormalCompute(unsigned int K, float maxRadius)
        : knn_it(K), knn_dist2(K), knn(K), K(K), maxRadius(maxRadius)
    {
    }

    void computeOneNormal(const tree_type &tree, const Splat &splat)
    {
        Statistics::Timer timer(computeStat);
        Eigen::Vector3f query(splat.position[0], splat.position[1], splat.position[2]);
        std::size_t knn_size;
        tree.k_nearest_neighbors_in(knn_size, &knn_it[0], &knn_dist2[0], query, K, maxRadius);

        knn.resize(knn_size);
        for (std::size_t i = 0; i < knn_size; i++)
            knn[i] = *knn_it[i];
        computeNormal(splat, knn, K);
    }

private:
    std::vector<tree_type::const_iterator> knn_it;
    std::vector<tree_type::float_t> knn_dist2;
    std::vector<Eigen::Vector3f> knn;
    unsigned int K;
    float maxRadius;
};

void runSweepContinuous(SplatSet::SplatStream *splatStream, ProgressDisplay *progress,
                        bool doCompute, int axis, unsigned int K, float radius)
{
    std::tr1::uint64_t nSplats = 0;
    Timer latency;

    SortStream sortStream(*splatStream, CompareSplats(axis), 1024 * 1024 * 1024);
    std::deque<Splat> active;
    NormalCompute compute(K, radius);
    NormalCompute::tree_type tree;

    std::size_t front = 0; // number for the first splat in the deque
    std::size_t next = 0;  // number of the next splat which needs a normal computed
    while (!sortStream.empty())
    {
        const unsigned int chunk = 65536;
        for (unsigned int i = 0; i < chunk && !sortStream.empty(); i++)
        {
            Splat s = *sortStream;
            active.push_back(s);
            if (doCompute)
            {
                Eigen::Vector3f pos(s.position[0], s.position[1], s.position[2]);
                tree.insert(pos);
            }
            ++sortStream;
            ++nSplats;
        }
        float end = active.back().position[axis] - radius;

        std::size_t next2 = next;
        while (active[next2 - front].position[axis] < end)
            next2++;

        if (doCompute)
        {
            for (std::size_t i = next; i < next2; i++)
            {
                compute.computeOneNormal(tree, active[i - front]);
            }
        }

        while (active[next2 - front].position[axis] > active[0].position[axis] + radius)
        {
            if (doCompute)
            {
                const Splat &rm = active[0];
                Eigen::Vector3f q(rm.position[0], rm.position[1], rm.position[2]);
                tree.erase_exact(q);
            }

            active.pop_front();
            front++;
        }

        if (progress != NULL)
            *progress += next2 - next;
        next = next2;
    }

    while (next < nSplats)
    {
        if (doCompute)
            compute.computeOneNormal(tree, active[next - front]);
        next++;
        if (progress != NULL)
            ++*progress;
    }

    Statistics::getStatistic<Statistics::Counter>("splats").add(nSplats);
    // ensures the progress bar completes even if there were non-finite splats
    if (progress != NULL)
        *progress += progress->expected_count() - progress->count();
}

} // anonymous namespace

void runSweep(const po::variables_map &vm, bool continuous)
{
    SplatSet::FileSet splats;

    const std::vector<std::string> &names = vm[Option::inputFile()].as<std::vector<std::string> >();
    const FastPly::ReaderType readerType = vm[Option::reader()].as<Choice<FastPly::ReaderTypeWrapper> >();
    const int K = vm[Option::neighbors()].as<int>();
    const float radius = vm[Option::radius()].as<double>();
    const int axis = vm[Option::axis()].as<int>();
    const bool compute = !vm.count(Option::noCompute());

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
    splats.setBufferSize(vm[Option::bufferSize()].as<std::size_t>());

    ProgressDisplay progress(splats.maxSplats(), Log::log[Log::info]);
    boost::scoped_ptr<SplatSet::SplatStream> splatStream(splats.makeSplatStream());
    if (continuous)
        runSweepContinuous(splatStream.get(), &progress, compute, axis, K, radius);
    else
        runSweepDiscrete(splatStream.get(), &progress, compute, axis, K, radius);
}
