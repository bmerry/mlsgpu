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
#include <boost/thread/thread.hpp>
#include <boost/progress.hpp>
#include <boost/ref.hpp>
#include "src/tr1_unordered_map.h"
#include <iostream>
#include <map>
#include <vector>
#include <algorithm>
#include <limits>
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
#include "src/workers.h"
#include "src/progress.h"
#include "src/mesh_filter.h"
#include "src/timeplot.h"
#include "src/bucket_collector.h"
#include "src/bucket_loader.h"
#include "src/mlsgpu_core.h"

namespace po = boost::program_options;
using namespace std;

/**
 * Main execution.
 *
 * @param devices         List of OpenCL devices to use
 * @param out             Output filename or basename
 * @param vm              Command-line options
 */
static void run(const std::vector<std::pair<cl::Context, cl::Device> > &devices,
                 const std::string &out,
                 const po::variables_map &vm)
{
    typedef SplatSet::FastBlobSet<SplatSet::FileSet> Splats;

    const std::size_t maxLoadSplats = getMaxLoadSplats(vm);
    const std::size_t memMesh = vm[Option::memMesh].as<Capacity>();

    Timeplot::Worker mainWorker("main");

    {
        Statistics::Timer grandTotalTimer("run.time");

        boost::scoped_ptr<FastPly::Writer> writer(doCreateWriter(vm));
        boost::scoped_ptr<MesherBase> mesher(doCreateMesher(vm, *writer, out));

        if (vm.count(Option::resume))
        {
            boost::filesystem::path path(vm[Option::resume].as<std::string>());
            mesher->resume(mainWorker, path, &Log::log[Log::info]);
        }
        else
        {
            {
                // Open a scope so that objects will be released before finalization

                boost::scoped_ptr<Timeplot::Action> initTimer(new Timeplot::Action("init", mainWorker, "init.time"));

                Log::log[Log::info] << "Initializing...\n";
                MesherGroup mesherGroup(memMesh);
                SlaveWorkers slaveWorkers(
                    mainWorker, vm, devices,
                    makeOutputGenerator(mesherGroup));
                BucketCollector collector(maxLoadSplats, boost::ref(*slaveWorkers.loader));

                Splats splats;
                doComputeBlobs(mainWorker, vm, splats,
                               boost::bind(&Splats::computeBlobs, &splats, _1, _2, &Log::log[Log::info], true));
                Grid grid = splats.getBoundingGrid();
                unsigned int chunkCells = postprocessGrid(vm, grid);

                initTimer.reset();

                for (unsigned int pass = 0; pass < mesher->numPasses(); pass++)
                {
                    Log::log[Log::info] << "\nPass " << pass + 1 << "/" << mesher->numPasses() << endl;
                    ostringstream passName;
                    passName << "pass" << pass + 1 << ".time";
                    Statistics::Timer timer(passName.str());

                    ProgressDisplay progress(splats.numSplats(), Log::log[Log::info]);

                    mesherGroup.setInputFunctor(mesher->functor(pass));

                    // Start threads
                    slaveWorkers.start(splats, grid, &progress);
                    mesherGroup.start();

                    try
                    {
                        doBucket(mainWorker, vm, splats, grid, chunkCells, collector);
                    }
                    catch (...)
                    {
                        // This can't be handled using unwinding, because that would operate in
                        // the wrong order
                        collector.flush();
                        slaveWorkers.stop();
                        mesherGroup.stop();
                        throw;
                    }

                    /* Shut down threads. Note that it has to be done in forward order to
                     * satisfy the requirement that stop() is only called after producers
                     * are terminated.
                     */
                    collector.flush();
                    slaveWorkers.stop();
                    mesherGroup.stop();
                }
            }

            if (vm.count(Option::checkpoint))
            {
                const boost::filesystem::path path(vm[Option::checkpoint].as<std::string>());
                mesher->checkpoint(mainWorker, path);
            }
            else
                mesher->write(mainWorker, &Log::log[Log::info]);
        }
    } // ends scope for grandTotalTimer

    Statistics::finalizeEventTimes();
    writeStatistics(vm);
}

int main(int argc, char **argv)
{
    Log::log.setLevel(Log::info);

    po::variables_map vm = processOptions(argc, argv, false);
    setLogLevel(vm);

    std::vector<cl::Device> devices = CLH::findDevices(vm);
    if (devices.empty())
    {
        cerr << "No suitable OpenCL device found\n";
        exit(1);
    }

    try
    {
        validateOptions(vm, false);
    }
    catch (invalid_option &e)
    {
        cerr << e.what() << endl;
        exit(1);
    }

    CLH::ResourceUsage totalUsage = resourceUsage(vm);
    Log::log[Log::info] << "About " << totalUsage.getTotalMemory() / (1024 * 1024) << "MiB of device memory will be used per device.\n";
    BOOST_FOREACH(const cl::Device &device, devices)
    {
        try
        {
            validateDevice(device, totalUsage);
        }
        catch (CLH::invalid_device &e)
        {
            cerr << e.what() << endl;
            exit(1);
        }
        Log::log[Log::info] << "Using device " << device.getInfo<CL_DEVICE_NAME>() << "\n";
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
            Timeplot::init(vm["timeplot"].as<string>());

        run(cd, vm[Option::outputFile].as<string>(), vm);
        unsigned long long filesWritten = Statistics::getStatistic<Statistics::Counter>("output.files").getTotal();
        if (filesWritten == 0)
            Log::log[Log::warn] << "Warning: no output files written!\n";
        else if (filesWritten == 1)
            Log::log[Log::info] << "1 output file written.\n";
        else
            Log::log[Log::info] << filesWritten << " output files written.\n";
    }
    catch (cl::Error &e)
    {
        cerr << "\nOpenCL error in " << e.what() << " (" << e.err() << ")\n";
        return 1;
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

    return 0;
}
