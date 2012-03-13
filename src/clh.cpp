/**
 * @file
 */

#ifndef __CL_ENABLE_EXCEPTIONS
# define __CL_ENABLE_EXCEPTIONS
#endif

#include <boost/program_options.hpp>
#include <boost/foreach.hpp>
#include <CL/cl.hpp>
#include <vector>
#include <string>
#include <sstream>
#include <utility>
#include <stdexcept>
#include <algorithm>
#include <map>
#include <cstdlib>
#include "clh.h"
#include "logging.h"

namespace po = boost::program_options;

namespace CLH
{

namespace detail
{

MemoryMapping::MemoryMapping(const cl::Memory &memory, const cl::Device &device)
    : memory(memory)
{
    const cl::Context &context = memory.getInfo<CL_MEM_CONTEXT>();
    queue = cl::CommandQueue(context, device, 0);
    ptr = NULL;
}

MemoryMapping::~MemoryMapping()
{
    if (ptr)
    {
        queue.enqueueUnmapMemObject(memory, ptr);
        queue.finish();
    }
}

} // namespace detail

BufferMapping::BufferMapping(const cl::Buffer &buffer, const cl::Device &device, cl_map_flags flags, ::size_t offset, ::size_t size)
    : detail::MemoryMapping(buffer, device)
{
    setPointer(getQueue().enqueueMapBuffer(buffer, CL_TRUE, flags, offset, size));
}

ImageMapping::ImageMapping(
    const cl::Image &image, const cl::Device &device, cl_map_flags flags,
    const cl::size_t<3> &origin, const cl::size_t<3> &region,
    ::size_t *rowPitch, ::size_t *slicePitch)
    : detail::MemoryMapping(image, device)
{
    setPointer(getQueue().enqueueMapImage(image, CL_TRUE, flags, origin, region, rowPitch, slicePitch));
}

ResourceUsage ResourceUsage::operator+(const ResourceUsage &b) const
{
    ResourceUsage out;
    out.maxMemory = std::max(maxMemory, b.maxMemory);
    out.totalMemory = totalMemory + b.totalMemory;
    out.imageWidth = std::max(imageWidth, b.imageWidth);
    out.imageHeight = std::max(imageHeight, b.imageHeight);
    return out;
}

ResourceUsage &ResourceUsage::operator+=(const ResourceUsage &b)
{
    maxMemory = std::max(maxMemory, b.maxMemory);
    totalMemory += b.totalMemory;
    imageWidth = std::max(imageWidth, b.imageWidth);
    imageHeight = std::max(imageHeight, b.imageHeight);
    return *this;
}

ResourceUsage ResourceUsage::operator*(unsigned int n) const
{
    if (n == 0)
        return ResourceUsage();
    else
    {
        ResourceUsage out = *this;
        out.totalMemory *= n;
        return out;
    }
}

void ResourceUsage::addBuffer(std::tr1::uint64_t bytes)
{
    maxMemory = std::max(maxMemory, bytes);
    totalMemory += bytes;
}

void ResourceUsage::addImage(std::size_t width, std::size_t height, std::size_t bytesPerPixel)
{
    std::tr1::uint64_t size = width;
    size *= height;
    size *= bytesPerPixel;
    addBuffer(size);
    imageWidth = std::max(imageWidth, width);
    imageHeight = std::max(imageHeight, height);
}

namespace detail
{
// Implementation in generated code
const std::map<std::string, std::string> &getSourceMap();
}

void addOptions(boost::program_options::options_description &desc)
{
    desc.add_options()
        (Option::device, boost::program_options::value<std::string>(), "OpenCL device name")
        (Option::cpu,                                                  "Use a CPU device")
        (Option::gpu,                                                  "Use a GPU device");
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

static void CL_CALLBACK contextCallback(const char *msg, const void *ptr, ::size_t cb, void *user)
{
    (void) ptr;
    (void) cb;
    (void) user;
    Log::log[Log::warn] << msg << "\n";
}

cl::Context makeContext(const cl::Device &device)
{
    const cl::Platform &platform = device.getInfo<CL_DEVICE_PLATFORM>();
    cl_context_properties props[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties) platform(), 0};
    return cl::Context(device.getInfo<CL_DEVICE_TYPE>(), props, contextCallback);
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
        s << "#define " << i->first << " " << i->second << "\n";
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

/**
 * Return an event that is already signaled as @c CL_COMPLETE.
 * This is equivalent to the other form but uses the queue to determine the
 * context.
 */
void doneEvent(const cl::CommandQueue &queue, cl::Event *event)
{
    if (event != NULL)
    {
        cl::UserEvent signaled(queue.getInfo<CL_QUEUE_CONTEXT>());
        signaled.setStatus(CL_COMPLETE);
        *event = signaled;
    }
}

cl_int enqueueMarkerWithWaitList(const cl::CommandQueue &queue,
                                 const std::vector<cl::Event> *events,
                                 cl::Event *event)
{
    if (events != NULL && events->empty())
        events = NULL; // to avoid having to check for both conditions later

    if (events == NULL && event == NULL)
        return CL_SUCCESS;
    else if (event == NULL)
        return queue.enqueueWaitForEvents(*events);
    else if (!(queue.getInfo<CL_QUEUE_PROPERTIES>() & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)
             || (events != NULL && events->size() > 1))
    {
        /* For the events->size() > 1, out-of-order case this is inefficient
         * but correct.  Alternatives would be to enqueue a dummy task (which
         * would have potentially large overhead to allocate a dummy buffer or
         * something), or to create a separate thread to wait for completion of
         * the events and signal a user event when done (which would force
         * scheduling to round trip via multiple CPU threads).
         */
        return queue.enqueueMarker(event);
    }
    else if (events == NULL)
    {
        doneEvent(queue, event);
    }
    else
    {
        // Exactly one input event, so just copy it to the output
        if (event != NULL)
            *event = (*events)[0];
    }
    return CL_SUCCESS;
}

cl_int enqueueReadBuffer(
    const cl::CommandQueue &queue,
    const cl::Buffer &buffer,
    cl_bool blocking,
    std::size_t offset,
    std::size_t size,
    void *ptr,
    const std::vector<cl::Event> *events,
    cl::Event *event)
{
    if (size == 0)
        return enqueueMarkerWithWaitList(queue, events, event);
    else
        return queue.enqueueReadBuffer(buffer, blocking, offset, size, ptr, events, event);
}

cl_int enqueueWriteBuffer(
    const cl::CommandQueue &queue,
    const cl::Buffer &buffer,
    cl_bool blocking,
    std::size_t offset,
    std::size_t size,
    const void *ptr,
    const std::vector<cl::Event> *events,
    cl::Event *event)
{
    if (size == 0)
        return enqueueMarkerWithWaitList(queue, events, event);
    else
        return queue.enqueueWriteBuffer(buffer, blocking, offset, size, ptr, events, event);
}

cl_int enqueueCopyBuffer(
    const cl::CommandQueue &queue,
    const cl::Buffer &src,
    const cl::Buffer &dst,
    std::size_t srcOffset,
    std::size_t dstOffset,
    std::size_t size,
    const std::vector<cl::Event> *events,
    cl::Event *event)
{
    if (size == 0)
        return enqueueMarkerWithWaitList(queue, events, event);
    else
        return queue.enqueueCopyBuffer(src, dst, srcOffset, dstOffset, size, events, event);
}

cl_int enqueueNDRangeKernel(
    const cl::CommandQueue &queue,
    const cl::Kernel &kernel,
    const cl::NDRange &offset,
    const cl::NDRange &global,
    const cl::NDRange &local,
    const std::vector<cl::Event> *events,
    cl::Event *event)
{
    for (std::size_t i = 0; i < global.dimensions(); i++)
        if (static_cast<const std::size_t *>(global)[i] == 0)
        {
            return enqueueMarkerWithWaitList(queue, events, event);
        }
    return queue.enqueueNDRangeKernel(kernel, offset, global, local, events, event);
}

static cl::NDRange makeNDRange(cl_uint dimensions, const std::size_t *sizes)
{
    switch (dimensions)
    {
    case 0: return cl::NDRange();
    case 1: return cl::NDRange(sizes[0]);
    case 2: return cl::NDRange(sizes[0], sizes[1]);
    case 3: return cl::NDRange(sizes[0], sizes[1], sizes[2]);
    default: abort(); // should never be reached
    }
}

cl_int enqueueNDRangeKernelSplit(
    const cl::CommandQueue &queue,
    const cl::Kernel &kernel,
    const cl::NDRange &offset,
    const cl::NDRange &global,
    const cl::NDRange &local,
    const std::vector<cl::Event> *events,
    cl::Event *event)
{
    /* If no size given, pick a default.
     * TODO: get performance hint from CL_KERNEL_PREFERRED_KERNEL_WORK_GROUP_SIZE_MULTIPLE?
     */
    if (local.dimensions() == 0)
    {
        switch (global.dimensions())
        {
        case 1:
            return enqueueNDRangeKernelSplit(queue, kernel, offset, global,
                                             cl::NDRange(256), events, event);
        case 2:
            return enqueueNDRangeKernelSplit(queue, kernel, offset, global,
                                             cl::NDRange(16, 16), events, event);
        case 3:
            return enqueueNDRangeKernelSplit(queue, kernel, offset, global,
                                             cl::NDRange(8, 8, 8), events, event);
        default:
            return enqueueNDRangeKernel(queue, kernel, offset, global, local, events, event);
        }
    }

    const std::size_t *origGlobal = static_cast<const std::size_t *>(global);
    const std::size_t *origLocal = static_cast<const std::size_t *>(local);
    const std::size_t *origOffset = static_cast<const std::size_t *>(offset);
    const std::size_t dims = global.dimensions();

    std::size_t main[3], extra[3], extraOffset[3];

    for (std::size_t i = 0; i < dims; i++)
    {
        if (origLocal[i] == 0)
            throw cl::Error(CL_INVALID_WORK_GROUP_SIZE, "Local work group size is zero");
        main[i] = origGlobal[i] / origLocal[i] * origLocal[i];
        extra[i] = origGlobal[i] - main[i];
        extraOffset[i] = origOffset[i] + main[i];
    }

    std::vector<cl::Event> wait;
    for (std::size_t mask = 0; mask < (1U << dims); mask++)
    {
        std::size_t curOffset[3];
        std::size_t curGlobal[3];
        std::size_t curLocal[3];
        bool use = false;
        for (std::size_t i = 0; i < dims; i++)
        {
            if (mask & (1U << i))
            {
                curGlobal[i] = extra[i];
                curOffset[i] = extraOffset[i];
                curLocal[i] = extra[i];
            }
            else
            {
                curGlobal[i] = main[i];
                curOffset[i] = offset[i];
                curLocal[i] = origLocal[i];
            }
            use |= curGlobal[i] > 0;
        }
        if (use)
        {
            wait.push_back(cl::Event());
            queue.enqueueNDRangeKernel(kernel,
                                       makeNDRange(dims, curOffset),
                                       makeNDRange(dims, curGlobal),
                                       makeNDRange(dims, curLocal),
                                       events, &wait.back());
        }
    }
    return enqueueMarkerWithWaitList(queue, &wait, event);
}

} // namespace CLH
