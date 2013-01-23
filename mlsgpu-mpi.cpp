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
#include <boost/foreach.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/smart_ptr/scoped_ptr.hpp>
#include <boost/smart_ptr/scoped_array.hpp>
#include <boost/thread/thread.hpp>
#include <boost/progress.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>
#include "src/tr1_unordered_map.h"
#include <iostream>
#include <map>
#include <vector>
#include <algorithm>
#include <numeric>
#include <stxxl.h>
#include <mpi.h>
#include "src/misc.h"
#include "src/clh.h"
#include "src/logging.h"
#include "src/timer.h"
#include "src/fast_ply.h"
#include "src/splat.h"
#include "src/grid.h"
#include "src/splat_tree_cl.h"
#include "src/marching.h"
#include "src/mls.h"
#include "src/mesher.h"
#include "src/options.h"
#include "src/splat_set.h"
#include "src/bucket.h"
#include "src/provenance.h"
#include "src/statistics.h"
#include "src/statistics_cl.h"
#include "src/work_queue.h"
#include "src/circular_buffer.h"
#include "src/workers.h"
#include "src/progress.h"
#include "src/progress_mpi.h"
#include "src/mesh_filter.h"
#include "src/timeplot.h"
#include "src/coarse_bucket.h"
#include "src/worker_group_mpi.h"
#include "src/serialize.h"
#include "src/mlsgpu_core.h"

namespace po = boost::program_options;
using namespace std;

template<>
void sendItem(const FineBucketGroup::WorkItem &item, MPI_Comm comm, int dest)
{
    Serialize::send(item.chunkId, comm, dest);
    Serialize::send(item.grid, comm, dest);
    Serialize::send(item.recursionState, comm, dest);
    const Splat *splatPtr = (const Splat *) item.splats.get();
    Serialize::send(splatPtr, item.numSplats, comm, dest);
}

template<>
void recvItem(FineBucketGroup::WorkItem &item, MPI_Comm comm, int source)
{
    Serialize::recv(item.chunkId, comm, source);
    Serialize::recv(item.grid, comm, source);
    Serialize::recv(item.recursionState, comm, source);
    Splat *splatPtr = (Splat *) item.splats.get();
    Serialize::recv(splatPtr, item.numSplats, comm, source);
}

template<>
std::size_t sizeItem(const FineBucketGroup::WorkItem &item)
{
    return item.numSplats;
}

template<>
void sendItem(const MesherGroup::WorkItem &item, MPI_Comm comm, int dest)
{
    Serialize::send(item.work, comm, dest);
}

template<>
void recvItem(MesherGroup::WorkItem &item, MPI_Comm comm, int dest)
{
    Serialize::recv(item.work, item.alloc.get(), comm, dest);
}

template<>
std::size_t sizeItem(const MesherGroup::WorkItem &item)
{
    return item.work.mesh.getHostBytes();
}

namespace
{

/**
 * Function object for doing the GPU work. There is one slave launched
 * on each node that has GPUs.
 */
class Slave
{
private:
    const std::vector<std::pair<cl::Context, cl::Device> > &devices;
    const po::variables_map &vm;
    MPI_Comm controlComm;
    int controlRoot;
    MPI_Comm scatterComm;
    int scatterRoot;
    MPI_Comm gatherComm;
    int gatherRoot;
    MPI_Comm progressComm;
    int progressRoot;

public:
    Slave(const std::vector<std::pair<cl::Context, cl::Device> > &devices,
          const po::variables_map &vm,
          MPI_Comm controlComm, int controlRoot,
          MPI_Comm scatterComm, int scatterRoot,
          MPI_Comm gatherComm, int gatherRoot,
          MPI_Comm progressComm, int progressRoot)
        : devices(devices), vm(vm),
        controlComm(controlComm), controlRoot(controlRoot),
        scatterComm(scatterComm), scatterRoot(scatterRoot),
        gatherComm(gatherComm), gatherRoot(gatherRoot),
        progressComm(progressComm), progressRoot(progressRoot)
    {
    }

    void operator()() const;
};

class ScatterGroup : public WorkerGroupScatter<FineBucketGroup::WorkItem, ScatterGroup>
{
public:
    typedef FineBucketGroup::WorkItem WorkItem;

    ScatterGroup(
        std::size_t numWorkers, std::size_t requesters,
        MPI_Comm comm, std::size_t bufferSize)
        : WorkerGroupScatter<WorkItem, ScatterGroup>("scatter", numWorkers, requesters, comm),
        splatBuffer("mem.ScatterGroup.splats", bufferSize)
    {
    }

    boost::shared_ptr<WorkItem> get(Timeplot::Worker &tworker, std::size_t size)
    {
        boost::shared_ptr<WorkItem> item = WorkerGroupScatter<FineBucketGroup::WorkItem, ScatterGroup>::get(tworker, size);
        item->splats = splatBuffer.allocate(tworker, size * sizeof(Splat), &getStat);
        item->numSplats = size;
        return item;
    }

    void freeItem(boost::shared_ptr<WorkItem> item)
    {
        splatBuffer.free(item->splats);
    }

private:
    CircularBuffer splatBuffer;
};

class GatherGroup : public WorkerGroupGather<MesherGroup::WorkItem, GatherGroup>
{
public:
    typedef MesherGroup::WorkItem WorkItem;

    GatherGroup(MPI_Comm comm, int root, std::size_t bufferSize)
        : WorkerGroupGather<WorkItem, GatherGroup>("gather", comm, root),
        meshBuffer("mem.GatherGroup.mesh", bufferSize)
    {
    }

    boost::shared_ptr<WorkItem> get(Timeplot::Worker &tworker, std::size_t size)
    {
        boost::shared_ptr<WorkItem> item = WorkerGroupGather<WorkItem, GatherGroup>::get(tworker, size);
        std::size_t rounded = roundUp(size, sizeof(cl_ulong)); // to ensure alignment
        item->alloc = meshBuffer.allocate(tworker, rounded, &getStat);
        return item;
    }

    void freeItem(boost::shared_ptr<WorkItem> item)
    {
        meshBuffer.free(item->alloc);
    }

private:
    CircularBuffer meshBuffer;
};

class OutputFunctor
{
private:
    GatherGroup &outGroup;
    ChunkId chunkId;
    Timeplot::Worker &tworker;

public:
    typedef void result_type;

    OutputFunctor(GatherGroup &outGroup, const ChunkId &chunkId, Timeplot::Worker &tworker)
        : outGroup(outGroup), chunkId(chunkId), tworker(tworker)
    {
    }

    void operator()(
        const cl::CommandQueue &queue,
        const DeviceKeyMesh &mesh,
        const std::vector<cl::Event> *events,
        cl::Event *event)
    {
        std::size_t bytes = mesh.getHostBytes();

        boost::shared_ptr<GatherGroup::WorkItem> item = outGroup.get(tworker, bytes);
        item->work.mesh = HostKeyMesh(item->alloc.get(), mesh);
        std::vector<cl::Event> wait(3);
        enqueueReadMesh(queue, mesh, item->work.mesh, events, &wait[0], &wait[1], &wait[2]);
        CLH::enqueueMarkerWithWaitList(queue, &wait, event);

        item->work.chunkId = chunkId;
        item->work.hasEvents = true;
        item->work.verticesEvent = wait[0];
        item->work.vertexKeysEvent = wait[1];
        item->work.trianglesEvent = wait[2];
        outGroup.push(item);
    }
};

class GetOutputFunctor
{
private:
    GatherGroup &outGroup;
public:
    typedef Marching::OutputFunctor result_type;

    explicit GetOutputFunctor(GatherGroup &outGroup)
        : outGroup(outGroup)
    {
    }

    result_type operator()(const ChunkId &chunkId, Timeplot::Worker &tworker) const
    {
        return OutputFunctor(outGroup, chunkId, tworker);
    }
};

void Slave::operator()() const
{
    const int subsampling = vm[Option::subsampling].as<int>();
    const int levels = vm[Option::levels].as<int>();
    const std::size_t maxSplit = vm[Option::maxSplit].as<int>();
    const std::size_t maxDeviceSplats = getMaxDeviceSplats(vm);
    const unsigned int numDeviceThreads = vm[Option::deviceThreads].as<int>();
    const unsigned int numBucketThreads = vm[Option::bucketThreads].as<int>();
    const float boundaryLimit = vm[Option::fitBoundaryLimit].as<double>();
    const MlsShape shape = vm[Option::fitShape].as<Choice<MlsShapeWrapper> >();

    const std::size_t memHostSplats = vm[Option::memHostSplats].as<Capacity>();
    const std::size_t memDeviceSplats = vm[Option::memDeviceSplats].as<Capacity>();
    const std::size_t memGather = vm[Option::memGather].as<Capacity>();

    const unsigned int block = 1U << (levels + subsampling - 1);
    const unsigned int blockCells = block - 1;

    GatherGroup gatherGroup(gatherComm, gatherRoot, memGather);
    boost::ptr_vector<DeviceWorkerGroup> deviceWorkerGroups;
    std::vector<DeviceWorkerGroup *> deviceWorkerGroupPtrs;
    for (std::size_t i = 0; i < devices.size(); i++)
    {
        DeviceWorkerGroup *dwg = new DeviceWorkerGroup(
            numDeviceThreads, GetOutputFunctor(gatherGroup),
            devices[i].first, devices[i].second,
            maxDeviceSplats, blockCells,
            memDeviceSplats, getMeshMemory(vm),
            levels, subsampling,
            boundaryLimit, shape);
        deviceWorkerGroups.push_back(dwg);
        deviceWorkerGroupPtrs.push_back(dwg);
    }
    FineBucketGroup fineBucketGroup(
        numBucketThreads, deviceWorkerGroupPtrs,
        memHostSplats, maxDeviceSplats, blockCells, maxSplit);
    RequesterScatter<FineBucketGroup::WorkItem, FineBucketGroup> requester(
        "requester", fineBucketGroup, scatterComm, scatterRoot);

    Grid grid;
    /* If the slave shares a node with the master, then OpenMPI busy-waits
     * here which takes CPU cycles away from the bounding box pass. Rather
     * sleep until something happens.
     */
    {
        int flag;
        do
        {
            MPI_Iprobe(controlRoot, MPI_ANY_TAG, controlComm, &flag, MPI_STATUS_IGNORE);
            boost::this_thread::sleep(boost::posix_time::milliseconds(200));
        } while (!flag);
    }
    Serialize::recv(grid, controlComm, controlRoot);

    /* NB: this does not yet support multi-pass algorithms. Currently there
     * are none, however.
     */

    ProgressMPI progress(NULL, grid.numCells(), progressComm, progressRoot);
    for (std::size_t i = 0; i < deviceWorkerGroups.size(); i++)
        deviceWorkerGroups[i].setProgress(&progress);
    fineBucketGroup.setProgress(&progress);

    fineBucketGroup.start(grid);
    for (std::size_t i = 0; i < deviceWorkerGroups.size(); i++)
        deviceWorkerGroups[i].start(grid);
    gatherGroup.start();

    requester();

    fineBucketGroup.stop();
    for (std::size_t i = 0; i < deviceWorkerGroups.size(); i++)
        deviceWorkerGroups[i].stop();
    gatherGroup.stop();
    progress.sync();

    /* Gather up the statistics */
    Statistics::finalizeEventTimes();
    std::ostringstream statsStream;
    boost::archive::text_oarchive oa(statsStream);
    oa << Statistics::Registry::getInstance();
    std::string statsStr = statsStream.str();
    MPI_Send(const_cast<char *>(statsStr.data()), statsStr.length(), MPI_CHAR,
             controlRoot, MLSGPU_TAG_WORK, controlComm);
}

/**
 * Main execution.
 *
 * @param comm            Communicator indicating the group to run on
 * @param devices         List of OpenCL devices to use
 * @param out             Output filename or basename
 * @param vm              Command-line options
 */
static void run(
    MPI_Comm comm,
    const std::vector<std::pair<cl::Context, cl::Device> > &devices,
    const string &out,
    const po::variables_map &vm)
{
    const int root = 0;
    int rank, size;
    MPI_Comm scatterComm;
    MPI_Comm gatherComm;
    MPI_Comm progressComm;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    MPI_Comm_dup(comm, &scatterComm);
    MPI_Comm_dup(comm, &gatherComm);
    MPI_Comm_dup(comm, &progressComm);
    /* Work out how many slaves there will be */
    int isSlave = devices.empty() ? 0 : 1;
    vector<int> slaveMask(size);
    MPI_Gather(&isSlave, 1, MPI_INT, &slaveMask[0], 1, MPI_INT, root, comm);

    boost::scoped_ptr<boost::thread> slaveThread;
    if (!devices.empty())
    {
        slaveThread.reset(new boost::thread(Slave(
                    devices, vm,
                    comm, root, scatterComm, root, gatherComm, root,
                    progressComm, root)));
    }

    if (rank == root)
    {
        typedef SplatSet::FastBlobSet<SplatSet::FileSet, Statistics::Container::stxxl_vector<SplatSet::BlobData> > Splats;

        const int numSlaves = accumulate(slaveMask.begin(), slaveMask.end(), 0);
        const float spacing = vm[Option::fitGrid].as<double>();
        const float smooth = vm[Option::fitSmooth].as<double>();
        const float maxRadius = vm.count(Option::maxRadius)
            ? vm[Option::maxRadius].as<double>() : std::numeric_limits<float>::infinity();
        const FastPly::WriterType writerType = vm[Option::writer].as<Choice<FastPly::WriterTypeWrapper> >();
        const MesherType mesherType = OOC_MESHER;
        const std::size_t maxHostSplats = getMaxHostSplats(vm);
        const std::size_t maxSplit = vm[Option::maxSplit].as<int>();
        const double pruneThreshold = vm[Option::fitPrune].as<double>();
        const bool split = vm.count(Option::split);
        const unsigned int splitSize = vm[Option::splitSize].as<unsigned int>();
        const int subsampling = vm[Option::subsampling].as<int>();
        const int levels = vm[Option::levels].as<int>();
        const std::size_t memMesh = vm[Option::memMesh].as<Capacity>();
        const std::size_t memScatter = vm[Option::memScatter].as<Capacity>();
        const std::size_t memReorder = vm[Option::memReorder].as<Capacity>();

        Timeplot::Worker mainWorker("main");

        const unsigned int block = 1U << (levels + subsampling - 1);
        const unsigned int blockCells = block - 1;

        {
            Statistics::Timer grandTotalTimer("run.time");

            MesherBase::Namer namer;
            if (split)
                namer = ChunkNamer(out);
            else
                namer = TrivialNamer(out);

            boost::scoped_ptr<FastPly::WriterBase> writer(FastPly::createWriter(writerType));
            writer->addComment("mlsgpu version: " + provenanceVersion());
            writer->addComment("mlsgpu variant: " + provenanceVariant());
            writer->addComment("mlsgpu options:" + makeOptions(vm));
            boost::scoped_ptr<MesherBase> mesher(createMesher(mesherType, *writer, namer));
            mesher->setPruneThreshold(pruneThreshold);
            mesher->setReorderCapacity(memReorder);

            Log::log[Log::info] << "Initializing...\n";
            MesherGroup mesherGroup(memMesh);
            ReceiverGather<MesherGroup::WorkItem, MesherGroup> receiver("receiver", mesherGroup, gatherComm, numSlaves);
            // TODO: tune number of scatter senders
            ScatterGroup scatterGroup(1, numSlaves, scatterComm, memScatter);
            CoarseBucket<Splats, ScatterGroup> coarseBucket(scatterGroup, mainWorker);

            Splats splats("mem.blobData");
            prepareInputs(splats, vm, smooth, maxRadius);
            try
            {
                Timeplot::Action timer("bbox", mainWorker, "bbox.time");
                splats.computeBlobs(spacing, blockCells, &Log::log[Log::info]);
            }
            catch (std::length_error &e) // TODO: should be a subclass of runtime_error
            {
                cerr << "At least one input point is required.\n";
                MPI_Abort(comm, 1);
            }
            Grid grid = splats.getBoundingGrid();
            for (unsigned int i = 0; i < 3; i++)
                if (grid.numVertices(i) > Marching::MAX_GLOBAL_DIMENSION)
                {
                    cerr << "The bounding box is too big (" << grid.numVertices(i) << " grid units).\n"
                        << "Perhaps you have used the wrong units for --fit-grid?\n";
                    MPI_Abort(comm, 1);
                    double size = grid.numCells(i) * grid.getSpacing();
                    Statistics::getStatistic<Statistics::Variable>(std::string("bbox") + "XYZ"[i]).add(size);
                }

            unsigned int chunkCells = 0;
            if (split)
            {
                /* Determine a chunk size from splitSize. We assume that a chunk will be
                 * sliced by an axis-aligned plane. This plane will cut each vertical and
                 * each diagonal edge ones, thus generating 2x^2 vertices. We then
                 * apply a fudge factor of 10 to account for the fact that the real
                 * world is not a simple plane, and will have walls, noise, etc, giving
                 * 20x^2 vertices.
                 *
                 * A manifold with genus 0 has two triangles per vertex; vertices take
                 * 12 bytes (3 floats) and triangles take 13 (count plus 3 uints in
                 * PLY), giving 38 bytes per vertex. So there are 760x^2 bytes.
                 */
                chunkCells = (unsigned int) ceil(sqrt((1024.0 * 1024.0 / 760.0) * splitSize));
                if (chunkCells == 0) chunkCells = 1;
            }

            for (int i = 0; i < size; i++)
            {
                if (slaveMask[i])
                    Serialize::send(grid, comm, i);
            }

            for (unsigned int pass = 0; pass < mesher->numPasses(); pass++)
            {
                Log::log[Log::info] << "\nPass " << pass + 1 << "/" << mesher->numPasses() << endl;
                ostringstream passName;
                passName << "pass" << pass + 1 << ".time";
                Statistics::Timer timer(passName.str());

                ProgressDisplay progress(grid.numCells(), Log::log[Log::info]);
                ProgressMPI progressMPI(&progress, grid.numCells(), progressComm, 0);

                mesherGroup.setInputFunctor(mesher->functor(pass));

                // Start threads
                coarseBucket.start(grid);
                scatterGroup.start();
                boost::thread receiverThread(boost::ref(receiver));
                mesherGroup.start();
                boost::thread progressThread(boost::ref(progressMPI));

                try
                {
                    Timeplot::Action bucketTimer("compute", mainWorker, "bucket.coarse.compute");
                    Bucket::bucket(splats, grid, maxHostSplats, blockCells, chunkCells, true, maxSplit,
                                   boost::ref(coarseBucket), &progressMPI);
                }
                catch (...)
                {
                    // This can't be handled using unwinding, because that would operate in
                    // the wrong order
                    coarseBucket.stop();
                    scatterGroup.stop();
                    receiverThread.join();
                    mesherGroup.stop();
                    progressMPI.sync();
                    progressThread.interrupt();
                    progressThread.join();
                    throw;
                }

                /* Shut down threads. Note that it has to be done in forward order to
                 * satisfy the requirement that stop() is only called after producers
                 * are terminated.
                 */
                coarseBucket.stop();
                scatterGroup.stop();
                receiverThread.join();
                mesherGroup.stop();
                progressMPI.sync();
                progressThread.join();
            }

            mesher->write(mainWorker, &Log::log[Log::info]);
        } // ends scope for grandTotalTimer

        for (int slave = 0; slave < size; slave++)
            if (slaveMask[slave])
            {
                MPI_Status status;
                MPI_Probe(MPI_ANY_SOURCE, MLSGPU_TAG_WORK, comm, &status);
                int length;
                MPI_Get_count(&status, MPI_CHAR, &length);
                boost::scoped_array<char> data(new char[length]);
                MPI_Recv(data.get(), length, MPI_CHAR, status.MPI_SOURCE, MLSGPU_TAG_WORK, comm, MPI_STATUS_IGNORE);

                if (slave != root) // root will already share our registry
                {
                    std::string statsStr(data.get(), length);
                    std::istringstream statsStream(statsStr);
                    boost::archive::text_iarchive ia(statsStream);
                    Statistics::Registry slaveRegistry;
                    ia >> slaveRegistry;
                    Statistics::Registry::getInstance().merge(slaveRegistry);
                }
            }

        writeStatistics(vm);
    }

    if (slaveThread)
        slaveThread->join();
}

} // anonymous namespace

int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE)
    {
        std::cerr << "MPI implementation does not provide the required level of thread support\n";
        MPI_Finalize();
        return 1;
    }

    Serialize::init();

    Log::log.setLevel(Log::info);
    po::variables_map vm = processOptions(argc, argv, true);
    setLogLevel(vm);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<cl::Device> devices = CLH::findDevices(vm);
    int numDevices = devices.size();
    int totalDevices;
    MPI_Reduce(&numDevices, &totalDevices, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        if (totalDevices == 0)
        {
            cerr << "No suitable OpenCL device found\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        try
        {
            validateOptions(vm, true);
        }
        catch (invalid_option &e)
        {
            cerr << e.what() << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    CLH::ResourceUsage totalUsage = resourceUsage(vm);

    if (rank == 0)
        Log::log[Log::info] << "About " << totalUsage.getTotalMemory() / (1024 * 1024) << "MiB of device memory will be used per device.\n";

    /* Give each node a turn to validate things. Doing it serially prevents
     * the output from becoming interleaved.
     */
    for (int node = 0; node < size; node++)
    {
        if (node == rank)
        {
            BOOST_FOREACH(const cl::Device &device, devices)
            {
                try
                {
                    validateDevice(device, totalUsage);
                }
                catch (CLH::invalid_device &e)
                {
                    cerr << e.what() << endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                Log::log[Log::info] << "Using device " << device.getInfo<CL_DEVICE_NAME>() << "\n";
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    std::vector<std::pair<cl::Context, cl::Device> > cd;
    cd.reserve(devices.size());
    for (std::size_t i = 0; i < devices.size(); i++)
    {
        cd.push_back(std::make_pair(CLH::makeContext(devices[i]), devices[i]));
    }

    try
    {
        if (vm.count("timeplot"))
        {
            ostringstream name;
            name << vm["timeplot"].as<string>() << "." << rank;
            Timeplot::init(name.str());
        }

        run(MPI_COMM_WORLD, cd, vm[Option::outputFile].as<string>(), vm);
        if (rank == 0)
        {
            unsigned long long filesWritten = Statistics::getStatistic<Statistics::Counter>("output.files").getTotal();
            if (filesWritten == 0)
                Log::log[Log::warn] << "Warning: no output files written!\n";
            else if (filesWritten == 1)
                Log::log[Log::info] << "1 output file written.\n";
            else
                Log::log[Log::info] << filesWritten << " output files written.\n";
        }
    }
    catch (cl::Error &e)
    {
        cerr << "\nOpenCL error in " << e.what() << " (" << e.err() << ")\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    catch (std::ios::failure &e)
    {
        reportException(e);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    catch (std::runtime_error &e)
    {
        reportException(e);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Finalize();
    return 0;
}
