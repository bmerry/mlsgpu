/**
 * @file
 *
 * Implementation of normal-computing using bucketing to handle large inputs OOC.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif

#define KNN_NABO 1
// #define KNN_INTERNAL 1

#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <memory>
#include <cmath>
#include <stdexcept>
#include <boost/program_options.hpp>
#include <boost/foreach.hpp>
#include <boost/smart_ptr/scoped_ptr.hpp>
#include <boost/smart_ptr/shared_ptr.hpp>
#include <boost/smart_ptr/make_shared.hpp>
#include <boost/ref.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <stxxl.h>
#include <Eigen/Core>
#ifdef _OPENMP
# include <omp.h>
#endif
#include "../src/bucket.h"
#include "../src/statistics.h"
#include "../src/splat_set.h"
#include "../src/fast_ply.h"
#include "../src/logging.h"
#include "../src/progress.h"
#include "../src/options.h"
#include "../src/worker_group.h"
#include "normals.h"
#include "normals_bucket.h"

#if KNN_NABO
# include <nabo/nabo.h>
#elif KNN_INTERNAL
# include "knng.h"
#endif

namespace po = boost::program_options;

namespace Option
{
    static inline const char *maxSplit()      { return "max-split"; }
    static inline const char *leafSize()      { return "leaf-size"; }
    static inline const char *maxHostSplats() { return "max-host-splats"; }
};

void addBucketOptions(po::options_description &opts)
{
    po::options_description opts2("Bucket mode options");
    opts2.add_options()
        (Option::maxHostSplats(), po::value<std::size_t>()->default_value(10000000), "Maximum splats per bin/slice")
        (Option::maxSplit(),      po::value<int>()->default_value(40000000), "Maximum fan-out in partitioning")
        (Option::leafSize(),      po::value<double>()->default_value(1000.0), "Size of top-level octree leaves");
    opts.add(opts2);
}

namespace
{

template<typename S, typename T>
class TransformSplatSet : public S
{
public:
    typedef T Transform;

    SplatSet::SplatStream *makeSplatStream() const
    {
        std::auto_ptr<SplatSet::SplatStream> child(S::makeSplatStream());
        SplatSet::SplatStream *stream = new MySplatStream(child.get(), transform);
        child.release();
        return stream;
    }

    template<typename RangeIterator>
    SplatSet::SplatStream *makeSplatStream(RangeIterator first, RangeIterator last) const
    {
        std::auto_ptr<SplatSet::SplatStream> child(S::makeSplatStream(first, last));
        SplatSet::SplatStream *stream = new MySplatStream(child.get(), transform);
        child.release();
        return stream;
    }

    SplatSet::BlobStream *makeBlobStream(const Grid &grid, Grid::size_type bucketSize) const
    {
        return new SplatSet::SimpleBlobStream(makeSplatStream(), grid, bucketSize);
    }

    void setTransform(const Transform &transform)
    {
        this->transform = transform;
    }

private:
    Transform transform;

    class MySplatStream : public SplatSet::SplatStream
    {
    private:
        boost::scoped_ptr<SplatSet::SplatStream> child;
        Transform transform;

    public:
        MySplatStream(SplatSet::SplatStream *child, const Transform &transform)
            : child(child), transform(transform) {}

        virtual SplatStream &operator++()
        {
            ++*child;
            return *this;
        }

        virtual Splat operator*() const
        {
            return boost::unwrap_ref(transform)(**child);
        }

        virtual bool empty() const
        {
            return child->empty();
        }

        virtual SplatSet::splat_id currentId() const
        {
            return child->currentId();
        }
    };
};

class TransformSetRadius
{
private:
    float radius;

public:
    explicit TransformSetRadius(float radius = 0.0) : radius(radius) {}

    Splat operator()(Splat s) const
    {
        s.radius = radius;
        return s;
    }
};


struct NormalItem
{
    Grid binGrid;
    int numNeighbors;
    float maxDistance2;
    bool compute;
    ProgressDisplay *progress;

    Statistics::Container::vector<Splat> splats;

    NormalItem(std::size_t maxSplats) : splats("mem.splats")
    {
        splats.reserve(maxSplats);
    }
};

/// Transformation used to transform the splats for @ref KDTree
class SplatToEigen
{
public:
    typedef Eigen::Vector3f result_type;

    Eigen::Vector3f operator()(const Splat &s) const
    {
        return Eigen::Vector3f(s.position[0], s.position[1], s.position[2]);
    }
};

class NormalWorker : public NormalStats
{
public:
    void start() {}
    void stop() {}

    void operator()(int gen, NormalItem &item)
    {
        Statistics::Timer timer(computeStat);
        (void) gen;

        if (item.compute)
        {
#if KNN_INTERNAL
            typedef std::tr1::uint32_t size_type;
            boost::scoped_ptr<KNNG<float, size_type> > knng;

            {
                KDTree<float, 3, size_type> tree(
                    boost::make_transform_iterator(item.splats.begin(), SplatToEigen()),
                    boost::make_transform_iterator(item.splats.end(), SplatToEigen()));
                knng.reset(tree.knn(item.numNeighbors, item.maxDistance2));
            }
#elif KNN_NABO
            Eigen::VectorXi indices(item.numNeighbors);
            Eigen::VectorXf dists2(item.numNeighbors);
            Eigen::MatrixXf M(3, item.splats.size());
            for (std::size_t i = 0; i < item.splats.size(); i++)
            {
                for (int j = 0; j < 3; j++)
                    M(j, i) = item.splats[i].position[j];
            }
            boost::scoped_ptr<Nabo::NNSearchF> tree(Nabo::NNSearchF::createKDTreeLinearHeap(M));
#endif

            std::vector<Eigen::Vector3f> neighbors;
            neighbors.reserve(item.numNeighbors);
#ifdef _OPENMP
# pragma omp parallel for firstprivate(neighbors, indices, dists2) schedule(dynamic,1024)
#endif
            for (std::size_t i = 0; i < item.splats.size(); i++)
            {
                const Splat &s = item.splats[i];
                Grid::difference_type vertexCoords[3];
                item.binGrid.worldToCell(s.position, vertexCoords);
                bool inside = true;
                for (int j = 0; j < 3; j++)
                    inside &= vertexCoords[j] >= 0 && Grid::size_type(vertexCoords[j]) < item.binGrid.numCells(j);
                if (inside)
                {
                    neighbors.clear();

#if KNN_INTERNAL
                    std::vector<std::pair<float, size_type> > nn = (*knng)[i];
                    for (std::size_t j = 0; j < nn.size(); j++)
                    {
                        int idx = nn[j].second;
                        const Splat &sn = item.splats[idx]; // TODO: just have KNNG give us Eigen::Vector3f?
                        neighbors.push_back(Eigen::Vector3f(sn.position[0], sn.position[1], sn.position[2]));
                    }
#elif KNN_NABO
                    Eigen::VectorXf query(3);
                    for (int j = 0; j < 3; j++)
                        query[j] = s.position[j];
                    tree->knn(query, indices, dists2, item.numNeighbors, 0.0f, 0, std::sqrt(item.maxDistance2));
                    for (std::size_t j = 0; j < std::size_t(item.numNeighbors); j++)
                    {
                        if ((std::tr1::isfinite)(dists2(j)))
                        {
                            int idx = indices(j);
                            neighbors.push_back(M.col(idx).head<3>());
                        }
                    }
#endif
                    computeNormal(s, neighbors, item.numNeighbors);
                }
            }
        }
        if (item.progress != NULL)
            *item.progress += item.binGrid.numCells();
    }
};

class NormalWorkerGroup : public WorkerGroup<NormalItem, int, NormalWorker, NormalWorkerGroup>
{
public:
    NormalWorkerGroup(std::size_t numWorkers, std::size_t spare, std::size_t maxSplats)
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
            addPoolItem(boost::make_shared<NormalItem>(maxSplats));
    }
};

template<typename Splats>
class BinProcessor
{
private:
    NormalWorkerGroup &outGroup;

    bool compute;
    int numNeighbors;
    float maxDistance2;

    ProgressDisplay *progress;
    Timer histoTimer;
    bool first;

    Statistics::Variable &loadStat;
    Statistics::Variable &binStat;

public:
    BinProcessor(
        NormalWorkerGroup &outGroup,
        bool compute,
        int numNeighbors,
        float maxDistance,
        ProgressDisplay *progress = NULL)
    :
        outGroup(outGroup),
        compute(compute),
        numNeighbors(numNeighbors), maxDistance2(maxDistance * maxDistance),
        progress(progress),
        first(true),
        loadStat(Statistics::getStatistic<Statistics::Variable>("load.time")),
        binStat(Statistics::getStatistic<Statistics::Variable>("load.bin.size"))
    {}

    void operator()(const typename SplatSet::Traits<Splats>::subset_type &subset,
                    const Grid &binGrid, const Bucket::Recursion &recursionState)
    {
        (void) recursionState;
        Log::log[Log::debug] << binGrid.numCells(0) << " x " << binGrid.numCells(1) << " x " << binGrid.numCells(2) << '\n';
        if (first)
            Statistics::getStatistic<Statistics::Variable>("histogram.time").add(histoTimer.getElapsed());
        first = false;

        boost::shared_ptr<NormalItem> item = outGroup.get();

        {
            Statistics::Timer timer(loadStat);
            boost::scoped_ptr<SplatSet::SplatStream> stream(subset.makeSplatStream());
            item->splats.clear();
            while (!stream->empty())
            {
                Splat s = **stream;
                item->splats.push_back(s);
                ++*stream;
            }
            item->compute = compute;
            item->binGrid = binGrid;
            item->numNeighbors = numNeighbors;
            item->maxDistance2 = maxDistance2;
            item->progress = progress;
        }
        binStat.add(item->splats.size());
        outGroup.push(0, item);
    }
};

} // anonymous namespace

void runBucket(const po::variables_map &vm)
{
    const int bucketSize = 256;
    const float leafSize = vm[Option::leafSize()].as<double>();
    const float spacing = leafSize / bucketSize;
    const float radius = vm[Option::radius()].as<double>();
    const int numNeighbors = vm[Option::neighbors()].as<int>();
    bool compute = !vm.count(Option::noCompute());

    const std::size_t maxHostSplats = vm[Option::maxHostSplats()].as<std::size_t>();
    const std::size_t maxSplit = vm[Option::maxSplit()].as<int>();
    const std::vector<std::string> &names = vm[Option::inputFile()].as<std::vector<std::string> >();
    const FastPly::ReaderType readerType = vm[Option::reader()].as<Choice<FastPly::ReaderTypeWrapper> >();

    typedef TransformSplatSet<SplatSet::FileSet, TransformSetRadius> Set0;
    typedef SplatSet::FastBlobSet<Set0, stxxl::VECTOR_GENERATOR<SplatSet::BlobData>::result > Splats;
    Splats splats;
    splats.setTransform(TransformSetRadius(radius));

    BOOST_FOREACH(const std::string &name, names)
    {
        std::auto_ptr<FastPly::ReaderBase> reader(FastPly::createReader(readerType, name, 1.0f));
        splats.addFile(reader.get());
        reader.release();
    }
    splats.setBufferSize(vm[Option::bufferSize()].as<std::size_t>());

#ifdef _OPENMP
    omp_set_num_threads(4);
#endif
    NormalWorkerGroup normalGroup(compute ? 2 : 1, compute ? 2 : 0, maxHostSplats);
    normalGroup.producerStart(0);
    normalGroup.start();

    try
    {
        Statistics::Timer timer("bbox.time");
        splats.computeBlobs(spacing, bucketSize, &Log::log[Log::info]);
    }
    catch (std::length_error &e)
    {
        std::cerr << "At least one input point is required.\n";
        std::exit(1);
    }

    Grid grid = splats.getBoundingGrid();
    {
        // Provided early, in case this is all that is desired
        Log::log[Log::info]
            << "Bounding box size: "
            << Statistics::getStatistic<Statistics::Variable>("blobset.bboxX").getMean() << " x "
            << Statistics::getStatistic<Statistics::Variable>("blobset.bboxY").getMean() << " x "
            << Statistics::getStatistic<Statistics::Variable>("blobset.bboxZ").getMean() << '\n';
    }

    ProgressDisplay progress(grid.numCells(), Log::log[Log::info]);

    BinProcessor<Splats> binProcessor(normalGroup, compute, numNeighbors, radius, &progress);

    Bucket::bucket(splats, grid, maxHostSplats, bucketSize, 0, true, maxSplit,
                   boost::ref(binProcessor), &progress);
    Statistics::Timer spindownTimer("spindown.time");
    normalGroup.producerStop(0);
    normalGroup.stop();
}
