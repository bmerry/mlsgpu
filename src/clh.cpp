/**
 * @file
 */

#ifndef __CL_ENABLE_EXCEPTIONS
#define __CL_ENABLE_EXCEPTIONS
#endif

#include <boost/program_options.hpp>
#include <boost/foreach.hpp>
#include <CL/cl.hpp>
#include <vector>
#include <string>
#include "clh.h"

namespace po = boost::program_options;
using namespace std;

namespace CLH
{

po::options_description getOptions()
{
    po::options_description ans("OpenCL options");
    ans.add_options()
        (Option::device,     po::value<string>(),    "OpenCL device name")
        (Option::cpu,                                "Use a CPU device")
        (Option::gpu,                                "Use a GPU device");
    return ans;
}

cl::Device findDevice(const po::variables_map &vm)
{
    /* Scores are used to decide between multiple matching devices */
    const int scoreGPU = 1;
    const int scoreExactDevice = 2;

    cl::Device ans;
    int score = -1;

    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    BOOST_FOREACH(const cl::Platform &platform, platforms)
    {
        vector<cl::Device> devices;
        cl_device_type type = CL_DEVICE_TYPE_ALL;

        platform.getDevices(type, &devices);
        BOOST_FOREACH(const cl::Device &device, devices)
        {
            bool good = true;
            int s = 0;
            /* Match name if given */
            if (vm.count(Option::device))
            {
                const std::string expected = vm[Option::device].as<string>();
                const std::string actual = device.getInfo<CL_DEVICE_NAME>();
                if (actual.substr(0, expected.size()) != expected)
                    good = false;
                else if (actual.size() == expected.size())
                    s += scoreExactDevice;
            }
            /* Match type if given */
            if (vm.count("cl-gpu"))
            {
                if (!(device.getInfo<CL_DEVICE_TYPE>() & CL_DEVICE_TYPE_GPU))
                    good = false;
            }
            if (vm.count("cl-cpu"))
            {
                if (!(device.getInfo<CL_DEVICE_TYPE>() & CL_DEVICE_TYPE_CPU))
                    good = false;
            }
            /* Give more weight to GPUs */
            if (device.getInfo<CL_DEVICE_TYPE>() & CL_DEVICE_TYPE_GPU)
                s += scoreGPU;
            /* Require OpenCL 1.1 */
            if (device.getInfo<CL_DEVICE_VERSION>() < string("OpenCL 1.1"))
                good = false;

            if (good && s > score)
            {
                ans = device;
                score = s;
            }
        }
    }
    return ans;
}

} // namespace CLH
