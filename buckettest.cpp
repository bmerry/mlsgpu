/**
 * @file
 *
 * Artificial benchmark to measure bucketing performance.
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
#include <boost/ref.hpp>
#include <stxxl.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Kd_tree.h>
#include <CGAL/Search_traits_3.h>
#include <CGAL/Fuzzy_sphere.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include "src/bucket.h"
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

    const char * const inputFile = "input-file";

    const char * const fitSmooth = "fit-smooth";
    const char * const fitGrid = "fit-grid";
    const char * const radius = "radius";
    const char * const neighbors = "neighbors";

    const char * const maxHostSplats = "max-host-splats";
    const char * const maxSplit = "max-split";
    const char * const leafSize = "leaf-size";

    const char * const statistics = "statistics";
    const char * const statisticsFile = "statistics-file";

    const char * const reader = "reader";
};

static void addCommonOptions(po::options_description &opts)
{
    opts.add_options()
        ("help,h",                "Show help")
        ("quiet,q",               "Do not show informational messages")
        (Option::debug,           "Show debug messages");
}

static void addFitOptions(po::options_description &opts)
{
    opts.add_options()
        (Option::fitSmooth,       po::value<double>()->default_value(4.0),  "Smoothing factor")
        (Option::fitGrid,         po::value<double>()->default_value(0.01), "Spacing of grid cells")
        (Option::radius,          po::value<double>()->default_value(0.1),  "Maximum radius to search")
        (Option::neighbors,       po::value<int>()->default_value(16),      "Neighbors to find");
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
        (Option::maxHostSplats, po::value<std::size_t>()->default_value(8000000), "Maximum splats per block on the CPU")
        (Option::maxSplit,     po::value<int>()->default_value(2097152), "Maximum fan-out in partitioning")
        (Option::leafSize,     po::value<int>()->default_value(256), "Size of top-level octree leaves, in cells")
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
        *out << "buckettest version: " << provenanceVersion() << '\n';
        *out << "buckettest variant: " << provenanceVariant() << '\n';
        *out << "buckettest options:" << makeOptions(vm) << '\n';
        *out << Statistics::Registry::getInstance();
        *out << *stxxl::stats::get_instance();
    }
}

static void usage(std::ostream &o, const po::options_description desc)
{
    o << "Usage: buckettest [options] input.ply [input.ply...]\n\n";
    o << desc;
}

static po::variables_map processOptions(int argc, char **argv)
{
    po::positional_options_description positional;
    positional.add(Option::inputFile, -1);

    po::options_description desc("General options");
    addCommonOptions(desc);
    addFitOptions(desc);
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

template<typename Splats>
class BinProcessor
{
private:
    int numNeighbors;
    float maxDistance2;

    ProgressDisplay *progress;

    Statistics::Variable &neighborStat;
    Statistics::Variable &loadStat;
    Statistics::Variable &computeStat;
    Statistics::Variable &qualityStat;
    Statistics::Variable &angleStat;

public:
    BinProcessor(int numNeighbors, float maxDistance, ProgressDisplay *progress = NULL)
        : numNeighbors(numNeighbors), maxDistance2(maxDistance * maxDistance),
        progress(progress),
        neighborStat(Statistics::getStatistic<Statistics::Variable>("neighbors")),
        loadStat(Statistics::getStatistic<Statistics::Variable>("load.time")),
        computeStat(Statistics::getStatistic<Statistics::Variable>("compute.time")),
        qualityStat(Statistics::getStatistic<Statistics::Variable>("quality")),
        angleStat(Statistics::getStatistic<Statistics::Variable>("angle"))
    {}

    void operator()(const typename SplatSet::Traits<Splats>::subset_type &subset,
                    const Grid &binGrid, const Bucket::Recursion &recursionState)
    {
        (void) recursionState;
        Log::log[Log::debug] << binGrid.numCells(0) << " x " << binGrid.numCells(1) << " x " << binGrid.numCells(2) << '\n';

        typedef CGAL::Simple_cartesian<float> Kernel;
        typedef Kernel::Point_3 Point;
        typedef CGAL::Search_traits_3<Kernel> SearchTraits;
        typedef CGAL::Orthogonal_k_neighbor_search<SearchTraits> Search;
        typedef Search::Tree Tree;

        Tree tree;
        std::vector<Splat> splats;
        splats.reserve(subset.maxSplats());
        boost::scoped_ptr<SplatSet::SplatStream> stream(subset.makeSplatStream());
        while (!stream->empty())
        {
            Splat s = **stream;
            Point p(s.position[0], s.position[1], s.position[2]);
            tree.insert(p);
            splats.push_back(s);
            ++*stream;
        }

        std::vector<Point> neighbors;
        neighbors.reserve(numNeighbors);
        for (std::size_t i = 0; i < splats.size(); i++)
        {
            const Splat &s = splats[i];
            float vertexCoords[3];
            binGrid.worldToVertex(s.position, vertexCoords);
            bool inside = true;
            for (int j = 0; j < 3; j++)
                inside &= vertexCoords[j] >= 0.0f && vertexCoords[j] < binGrid.numVertices(j);
            if (inside)
            {
                neighbors.clear();
                Point p(s.position[0], s.position[1], s.position[2]);
                // + 1 because we will find the point itself
                Search search(tree, p, numNeighbors + 1);

                float maxN2 = 0.0f;
                for (Search::iterator j = search.begin(); j != search.end(); ++j)
                    if (j->second != 0.0f && j->second <= maxDistance2)
                    {
                        neighbors.push_back(j->first);
                        maxN2 = std::max(maxN2, j->second);
                    }
                neighborStat.add(neighbors.size());

                if (neighbors.size() == std::size_t(numNeighbors))
                {
                    Eigen::Vector3f oldNormal(s.normal[0], s.normal[1], s.normal[2]);
                    oldNormal.normalize();

                    Eigen::Matrix3f cov;
                    cov.setZero();
                    BOOST_FOREACH(const Point &n, neighbors)
                    {
                        Kernel::Vector_3 delta = n - p;
                        float w = 1.0 - delta.squared_length() / maxN2;
                        w *= w;
                        w *= w;
                        for (int i = 0; i < 3; i++)
                            for (int j = 0; j <= i; j++)
                                cov(i, j) += delta[i] * delta[j];
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

                    float ang = std::acos(std::min(dot, 1.0f));
                    angleStat.add(ang * 57.2957795130823f);
                    qualityStat.add(quality);
                }
            }
        }

        if (progress != NULL)
            *progress += binGrid.numCells();
    }
};

static void run(const po::variables_map &vm)
{
    const float spacing = vm[Option::fitGrid].as<double>();
    const float smooth = vm[Option::fitSmooth].as<double>();
    const float radius = vm[Option::radius].as<double>();
    const int numNeighbors = vm[Option::neighbors].as<int>();

    const std::size_t maxHostSplats = vm[Option::maxHostSplats].as<std::size_t>();
    const std::size_t maxSplit = vm[Option::maxSplit].as<int>();
    const int leafSize = vm[Option::leafSize].as<int>();
    const std::vector<std::string> &names = vm[Option::inputFile].as<std::vector<std::string> >();
    const FastPly::ReaderType readerType = vm[Option::reader].as<Choice<FastPly::ReaderTypeWrapper> >();

    typedef TransformSplatSet<SplatSet::FileSet, TransformSetRadius> Set0;
    typedef SplatSet::FastBlobSet<Set0, stxxl::VECTOR_GENERATOR<SplatSet::BlobData>::result > Splats;
    Splats splats;
    splats.setTransform(TransformSetRadius(radius));

    BOOST_FOREACH(const std::string &name, names)
    {
        decache(name);
        std::auto_ptr<FastPly::ReaderBase> reader(FastPly::createReader(readerType, name, smooth));
        splats.addFile(reader.get());
        reader.release();
    }

    try
    {
        Statistics::Timer timer("bbox.time");
        splats.computeBlobs(spacing, leafSize, &Log::log[Log::info]);
    }
    catch (std::length_error &e)
    {
        std::cerr << "At least one input point is required.\n";
        std::exit(1);
    }

    Grid grid = splats.getBoundingGrid();
    ProgressDisplay progress(grid.numCells(), Log::log[Log::info]);

    BinProcessor<Splats> binProcessor(numNeighbors, radius, &progress);
    Bucket::bucket(splats, grid, maxHostSplats, leafSize, 0, true, maxSplit,
                   boost::ref(binProcessor), &progress);

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
