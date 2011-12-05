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
#include <sstream>
#include <utility>
#include <stdexcept>
#include <map>
#include "clh.h"
#include "logging.h"

namespace po = boost::program_options;

namespace CLH
{

namespace detail
{
// Implementation in generated code
const std::map<std::string, std::string> getSourceMap();
}

boost::program_options::options_description getOptions()
{
    boost::program_options::options_description ans("OpenCL options");
    ans.add_options()
        (Option::device,  boost::program_options::value<std::string>(),    "OpenCL device name")
        (Option::cpu,                                                 "Use a CPU device")
        (Option::gpu,                                                 "Use a GPU device");
    return ans;
}

cl::Device findDevice(const boost::program_options::variables_map &vm)
{
    /* Scores are used to decide between multiple matching devices */
    const int scoreGPU = 1;
    const int scoreExactDevice = 2;

    cl::Device ans;
    int score = -1;

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    BOOST_FOREACH(const cl::Platform &platform, platforms)
    {
        std::vector<cl::Device> devices;
        cl_device_type type = CL_DEVICE_TYPE_ALL;

        platform.getDevices(type, &devices);
        BOOST_FOREACH(const cl::Device &device, devices)
        {
            bool good = true;
            int s = 0;
            /* Match name if given */
            if (vm.count(Option::device))
            {
                const std::string expected = vm[Option::device].as<std::string>();
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
            if (device.getInfo<CL_DEVICE_VERSION>() < std::string("OpenCL 1.1"))
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

cl::Program build(const cl::Context &context, const std::vector<cl::Device> &devices,
                  const std::string &filename, const std::map<std::string, std::string> &defines,
                  const std::string &options)
{
    const std::map<std::string, std::string> &sourceMap = detail::getSourceMap();
    if (!sourceMap.count(filename))
        throw std::invalid_argument("No such program " + filename);
    const std::string &source = sourceMap.find(filename)->second;

    std::ostringstream s;
    for (std::map<std::string, std::string>::const_iterator i = defines.begin(); i != defines.end(); i++)
    {
        s << "#define " << i->first << " " << i->second;
    }
    s << "#line 1 \"" << filename << "\"\n";
    const std::string header = s.str();
    cl::Program::Sources sources(2);
    sources[0] = std::make_pair(header.data(), header.length());
    sources[1] = std::make_pair(source.data(), source.length());
    cl::Program program(context, sources);

    try
    {
        program.build(devices, options.c_str());
    }
    catch (cl::Error &e)
    {
        std::ostream &msg = Log::log[Log::error];
        BOOST_FOREACH(const cl::Device &device, devices)
        {
            const std::string log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
            if (log != "" && log != "\n")
            {
                msg << "Log for device " << device.getInfo<CL_DEVICE_NAME>() << '\n';
                msg << log << '\n';
            }
        }
        throw;
    }

    return program;
}

cl::Program build(const cl::Context &context,
                  const std::string &filename, const std::map<std::string, std::string> &defines,
                  const std::string &options)
{
    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    return build(context, devices, filename, defines, options);
}

} // namespace CLH
