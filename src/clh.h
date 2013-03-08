/**
 * @file
 */

#ifndef CLH_H
#define CLH_H

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include "tr1_cstdint.h"
#include "tr1_unordered_map.h"
#include <boost/program_options.hpp>
#include <boost/noncopyable.hpp>
#include <vector>
#include <string>
#include <map>
#include <CL/cl.hpp>
#include "allocator.h"

namespace Statistics
{
    class Variable;
    class Registry;
}

/// OpenCL helper functions
namespace CLH
{

#if HAVE_CL_LOCAL
#define CLH_LOCAL(arg) (cl::Local(arg))
#else
#define CLH_LOCAL(arg) (cl::__local(arg))
#endif

/**
 * Exception thrown when an OpenCL device cannot be used.
 */
class invalid_device : public std::runtime_error
{
private:
    cl::Device device;
public:
    invalid_device(const cl::Device &device, const std::string &msg)
        : std::runtime_error(device.getInfo<CL_DEVICE_NAME>() + ": " + msg) {}

    cl::Device getDevice() const
    {
        return device;
    }

    virtual ~invalid_device() throw()
    {
    }
};

namespace detail
{

/**
 * RAII encapsulation of a memory-mapped OpenCL buffer or image.
 */
class MemoryMapping : public boost::noncopyable
{
private:
    cl::Memory memory;      ///< Memory object to unmap on destruction
    cl::CommandQueue queue; ///< Privately allocated command queue
    void *ptr;              ///< Mapped pointer

protected:
    /**
     * Construct without a queue. The map and unmap operations will be
     * performed on an internally-allocated queue. The subclass constructor
     * must perform the mapping and provide the pointer to @ref setPointer.
     *
     * @param memory       The object that is mapped.
     * @param device       The device used to construct the command queue.
     */
    MemoryMapping(const cl::Memory &memory, const cl::Device &device);

    /**
     * Construct using an existing queue. The subclass constructor must
     * perform the mapping using this queue, and provide the pointer to
     * @ref setPointer.
     */
    MemoryMapping(const cl::Memory &memory, const cl::CommandQueue &queue);

    /**
     * Destructor that releases the mapping. Note that it will @em suppress any
     * error that occurs during unmapping, due to the exception safety
     * requirements of constructors. In most cases you should use @ref reset
     * and only rely on the destructor for cleaning up after an exception.
     */
    ~MemoryMapping();

    /**
     * Subclasses must call this inside the constructor to set the mapped pointer.
     *
     * @param ptr         Pointer to the mapping memory.
     * @pre @a ptr is not @c NULL.
     */
    void setPointer(void *ptr) { this->ptr = ptr; }

    /**
     * Retrieve the queue that will be used for unmapping. This will be a @c
     * NULL pointer if @ref reset has been called.
     */
    const cl::CommandQueue &getQueue() const { return queue; }

    /**
     * Retrieve the mapped memory object. This will be a @c NULL pointer if
     * @ref reset has been called.
     */
    const cl::Memory &getMemory() const { return memory; }

public:
    /**
     * Retrieve the pointer to the mapped memory.
     *
     * This will be @c NULL if @ref reset has been called.
     */
    void *get() const { return ptr; }

    /**
     * Release the mapping and optionally return an event. It is safe to call
     * this multiple times, but if the object has already been released then
     * @a event will be set to a NULL pointer.
     *
     * @param      events  If non-NULL, events to wait for before the unmapping occurs.
     * @param[out] event   If non-NULL, contains an event for the unmapping.
     *
     * @post The mapping will be unmapped and the references to the queue and
     * the memory object are released.
     */
    void reset(const std::vector<cl::Event> *events = NULL, cl::Event *event = NULL);
};

template<typename T>
class TypedMemoryMapping : public MemoryMapping
{
protected:
    TypedMemoryMapping(const cl::Memory &memory, const cl::Device &device)
        : MemoryMapping(memory, device)
    {
    }

    TypedMemoryMapping(const cl::Memory &memory, const cl::CommandQueue &queue)
        : MemoryMapping(memory, queue)
    {
    }

public:
    T *get() const { return static_cast<T *>(MemoryMapping::get()); }
    T &operator *() const { return *get(); }
    T *operator->() const { return get(); }
    T &operator[](std::size_t index) const { return get()[index]; }
};

} // namespace detail

/**
 * RAII wrapper around mapping and unmapping a buffer.
 * If a non-NULL event pointer is provided, the mapping is asynchronous and
 * the event is used to signal completion. Otherwise, the mapping is
 * synchronous.
 */
template<typename T>
class BufferMapping : public detail::TypedMemoryMapping<T>
{
public:
    BufferMapping(const cl::Buffer &buffer, const cl::Device &device, cl_map_flags flags, ::size_t offset, ::size_t size, cl::Event *event = NULL)
        : detail::TypedMemoryMapping<T>(buffer, device)
    {
        this->setPointer(this->getQueue.enqueueMapBuffer(buffer, event == NULL, flags, offset, size, NULL, event));
    }

    BufferMapping(const cl::Buffer &buffer, const cl::CommandQueue &queue, cl_map_flags flags, ::size_t offset, ::size_t size, cl::Event *event = NULL)
        : detail::TypedMemoryMapping<T>(buffer, queue)
    {
        this->setPointer(queue.enqueueMapBuffer(buffer, event == NULL, flags, offset, size, NULL, event));
    }

    const cl::Buffer &getBuffer() const { return static_cast<const cl::Buffer &>(this->getMemory()); }
};

/**
 * RAII wrapper to both allocate and map a buffer with the @c
 * CL_MEM_ALLOC_HOST_PTR flag. It can be treated as a memory allocator for
 * memory that can be efficiently transferred to and from device memory (at
 * least for some particular CL implementations) but which cannot be directly
 * used on the device.
 */
template<typename T>
class PinnedMemory : public detail::TypedMemoryMapping<T>
{
private:
    Statistics::Allocator<std::allocator<T> > allocator;
    std::size_t nElements;
public:
    PinnedMemory(const std::string &name, const cl::Context &context, const cl::Device &device, std::size_t nElements = 1)
        : detail::TypedMemoryMapping<T>(cl::Buffer(context, CL_MEM_ALLOC_HOST_PTR, nElements * sizeof(T)), device),
        allocator(Statistics::makeAllocator<Statistics::Allocator<std::allocator<T> > >(name)),
        nElements(nElements)
    {
        allocator.recordAllocate(nElements * sizeof(T));
        T *ptr = static_cast<T *>(this->getQueue().enqueueMapBuffer(
                static_cast<const cl::Buffer &>(this->getMemory()),
                CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, nElements * sizeof(T)));
        this->setPointer(ptr);
        for (std::size_t i = 0; i < nElements; i++)
            new(ptr + i) T(); // run constructor on the allocated memory
    }

    ~PinnedMemory()
    {
        if (this->get() != NULL)
            for (std::size_t i = 0; i < nElements; i++)
                this->get()[i].~T();
        allocator.recordDeallocate(nElements * sizeof(T));
    }
};

/**
 * Represents the resources required or consumed by an algorithm class.
 */
class ResourceUsage
{
private:
    std::tr1::uint64_t maxMemory;     ///< Largest single memory allocation
    std::tr1::uint64_t totalMemory;   ///< Sum of all allocations
    std::size_t imageWidth;           ///< Maximum image width used (0 if no images)
    std::size_t imageHeight;          ///< Maximum image height used (0 if no images)

    typedef std::tr1::unordered_map<std::string, std::size_t> map_type;
    /**
     * Allocation amount associated with each name.
     */
    map_type allocations;

public:
    ResourceUsage() : maxMemory(0), totalMemory(0), imageWidth(0), imageHeight(0) {}

    /// Indicate that memory for a buffer is required.
    void addBuffer(const std::string &name, std::tr1::uint64_t bytes);

    /// Indicate that memory for a 2D image is required.
    void addImage(const std::string &name, std::size_t width, std::size_t height, std::size_t bytesPerPixel);

    /**
     * Computes the combined requirements given the individual requirements for two
     * steps. This assumes that the steps are active simultaneously, and hence that
     * totals must be added.
     */
    ResourceUsage operator+(const ResourceUsage &r) const;

    /**
     * Computes the combined requirements given the individual requirements for two
     * steps. This assumes that the steps are active simultaneously, and hence that
     * totals must be added.
     */
    ResourceUsage &operator+=(const ResourceUsage &r);

    /**
     * Adds @a n copies of the resource.
     */
    ResourceUsage operator*(unsigned int n) const;

    /// Retrieve the maximum single allocation size required.
    std::tr1::uint64_t getMaxMemory() const { return maxMemory; }
    /// Retrieve the maximum total memory required.
    std::tr1::uint64_t getTotalMemory() const { return totalMemory; }
    /// Retrieve the largest image width required (0 if no images).
    std::size_t getImageWidth() const { return imageWidth; }
    /// Retrieve the largest image height required (0 if no images).
    std::size_t getImageHeight() const { return imageHeight; }

    /**
     * Increment peak statistics based on the internal table of allocations.
     * @param registry      The registry to update.
     * @param prefix        A prefix that is prepended to the internal names to make statistic names
     */
    void addStatistics(Statistics::Registry &registry, const std::string &prefix);
};

/// Option names for OpenCL options
namespace Option
{
const char * const device = "cl-device";
const char * const gpu = "cl-gpu";
const char * const cpu = "cl-cpu";
} // namespace Option

/**
 * Append program options for selecting an OpenCL device.
 *
 * The resulting variables map can be passed to @ref findDevices.
 */
void addOptions(boost::program_options::options_description &desc);

/**
 * Pick OpenCL devices based on command-line options. Each device is matched
 * against a number of criteria and used if any of them match.
 * - <tt>--cl-device=name:n</tt> matches for the nth device with a prefix of @a
 *   name.
 * - <tt>--cl-device=name</tt> matches for all devices with a prefix of @a name.
 * - <tt>--cl-gpu</tt> causes all GPU devices to match.
 * - <tt>--cl-cpu</tt> causes all CPU devices to match.
 *
 * If none of --cl-device, --cl-cpu and --cl-gpu is given then --cl-gpu is implied.
 *
 * @return A (possibly empty) list of devices matching the command-line options.
 */
std::vector<cl::Device> findDevices(const boost::program_options::variables_map &vm);

/**
 * Create an OpenCL context suitable for use with a device.
 */
cl::Context makeContext(const cl::Device &device);

/**
 * Build a program for potentially multiple devices.
 *
 * If compilation fails, the build log will be emitted to the error log.
 *
 * @param context         Context to use for building.
 * @param devices         Devices to build for.
 * @param filename        File to load (relative to current directory).
 * @param defines         Defines that will be set before the source is preprocessed.
 * @param options         Extra compilation options.
 *
 * @throw std::invalid_argument if the file could not be opened.
 * @throw cl::Error if the program could not be compiled.
 */
cl::Program build(const cl::Context &context, const std::vector<cl::Device> &devices,
                  const std::string &filename, const std::map<std::string, std::string> &defines = std::map<std::string, std::string>(),
                  const std::string &options = "");

/**
 * Build a program for all devices associated with a context.
 *
 * This is a convenience wrapper for the form that takes an explicit device
 * list.
 */
cl::Program build(const cl::Context &context,
                  const std::string &filename, const std::map<std::string, std::string> &defines = std::map<std::string, std::string>(),
                  const std::string &options = "");

/**
 * Implementation of clEnqueueMarkerWithWaitList which can be used in OpenCL
 * 1.1. It differs from the OpenCL 1.2 function in several ways:
 *  - If no input events are passed, it does not wait for anything (other than as
 *    constrained by an in-order queue), rather than waiting for all previous
 *    work.
 *  - If exactly one input event is passed, it may return this event (with
 *    an extra reference) instead of creating a new one.
 *  - It may choose to wait on all previous events in the command queue.
 *    Thus, if your algorithm depends on it not doing so (e.g. you've used
 *    user events to create dependencies backwards in time) it may cause
 *    a deadlock.
 *  - It is legal for @a event to be @c NULL (in which case the marker is
 *    still enqueued, so that if the wait list contained events from other
 *    queues then a barrier on this queue would happen-after those events).
 */
cl_int enqueueMarkerWithWaitList(const cl::CommandQueue &queue,
                                 const std::vector<cl::Event> *events,
                                 cl::Event *event);

/**
 * Extension of @c cl::CommandQueue::enqueueReadBuffer that allows the
 * size to be zero.
 */
cl_int enqueueReadBuffer(
    const cl::CommandQueue &queue,
    const cl::Buffer &buffer,
    cl_bool blocking,
    std::size_t offset,
    std::size_t size,
    void *ptr,
    const std::vector<cl::Event> *events = NULL,
    cl::Event *event = NULL);

/**
 * Extension of @c cl::CommandQueue::enqueueWriteBuffer that allows the
 * size to be zero.
 */
cl_int enqueueWriteBuffer(
    const cl::CommandQueue &queue,
    const cl::Buffer &buffer,
    cl_bool blocking,
    std::size_t offset,
    std::size_t size,
    const void *ptr,
    const std::vector<cl::Event> *events = NULL,
    cl::Event *event = NULL);

/**
 * Extension of @c cl::CommandQueue::enqueueCopyBuffer that allows the size
 * to be zero.
 */
cl_int enqueueCopyBuffer(
    const cl::CommandQueue &queue,
    const cl::Buffer &src,
    const cl::Buffer &dst,
    std::size_t srcOffset,
    std::size_t dstOffset,
    std::size_t size,
    const std::vector<cl::Event> *events = NULL,
    cl::Event *event = NULL);

/**
 * Extension of @c cl::CommandQueue::enqueueNDRangeKernel that allows the
 * number of work-items to be zero.
 *
 * @param queue      Queue to enqueue on
 * @param kernel,offset,global,local,events,event As for @c cl::CommandQueue::enqueueNDRangeKernel
 * @param stat       If non-NULL, the event time (if any) will be recorded in this statistic.
 */
cl_int enqueueNDRangeKernel(
    const cl::CommandQueue &queue,
    const cl::Kernel &kernel,
    const cl::NDRange &offset,
    const cl::NDRange &global,
    const cl::NDRange &local = cl::NullRange,
    const std::vector<cl::Event> *events = NULL,
    cl::Event *event = NULL,
    Statistics::Variable *stat = NULL);

/**
 * Extends kernel enqueuing by allowing the global size to not be a multiple of
 * the local size. Where necessary, multiple launches are used to handle the
 * left-over bits at the edges, adjusting the global offset to compensate.
 *
 * This does have some side-effects:
 *  - Different work-items will participate in workgroups of different sizes.
 *    Thus, the workgroup size cannot be baked into the kernel.
 *  - @c get_global_id will work as expected, but @c get_group_id and
 *    @c get_global_offset may not behave as expected.
 * In general this function is best suited to cases where the workitems
 * operate complete independently.
 *
 * The provided @a local is used both as the preferred work group size for the
 * bulk of the work, and as an upper bound on work group size.
 *
 * @param queue      Queue to enqueue on
 * @param kernel,offset,global,local,events,event As for @c cl::CommandQueue::enqueueNDRangeKernel
 * @param stat       If non-NULL, the event time will be recorded in this statistic.
 */
cl_int enqueueNDRangeKernelSplit(
    const cl::CommandQueue &queue,
    const cl::Kernel &kernel,
    const cl::NDRange &offset,
    const cl::NDRange &global,
    const cl::NDRange &local = cl::NullRange,
    const std::vector<cl::Event> *events = NULL,
    cl::Event *event = NULL,
    Statistics::Variable *stat = NULL);

} // namespace CLH

#endif /* !CLH_H */
