/**
 * @file
 */

#ifndef CLH_H
#define CLH_H

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <boost/program_options.hpp>
#include <boost/noncopyable.hpp>
#include <CL/cl.hpp>

/// OpenCL helper functions
namespace CLH
{

namespace detail
{

class MemoryMapping : public boost::noncopyable
{
private:
    cl::Memory memory;      ///< Memory object to unmap on destruction
    cl::CommandQueue queue; ///< Privately allocated command queue
    void *ptr;              ///< Mapped pointer

protected:
    MemoryMapping(const cl::Memory &memory, const cl::Device &device);
    ~MemoryMapping();

    void setPointer(void *ptr) { this->ptr = ptr; }
    const cl::CommandQueue &getQueue() const { return queue; }
    const cl::Memory &getMemory() const { return memory; }

public:
    void *get() const { return ptr; }
};

} // namespace detail

/**
 * RAII wrapper around mapping and unmapping a buffer.
 * It only handles synchronous mapping and unmapping.
 */
class BufferMapping : public detail::MemoryMapping
{
public:
    BufferMapping(const cl::Buffer &buffer, const cl::Device &device, cl_map_flags flags, ::size_t offset, ::size_t size);

    const cl::Buffer &getBuffer() const { return static_cast<const cl::Buffer &>(getMemory()); }
};

/**
 * RAII wrapper around mapping and unmapping an image.
 * It only handles synchronous mapping and unmapped.
 */
class ImageMapping : public detail::MemoryMapping
{
public:
    ImageMapping(const cl::Image &image, const cl::Device &device, cl_map_flags flags,
                 const cl::size_t<3> &origin, const cl::size_t<3> &region,
                 ::size_t *rowPitch, ::size_t *slicePitch);

    const cl::Image &getImage() const { return static_cast<const cl::Image &>(getMemory()); }
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
 * The resulting variables map can be passed to @ref findDevice.
 */
void addOptions(boost::program_options::options_description &desc);

/**
 * Pick an OpenCL device based on command-line options.
 *
 * If more than one device matches the criteria, GPU devices are preferred.
 * If there is no exact match for the device name, a prefix will be accepted.
 *
 * @return A device matching the command-line options, or @c NULL if none matches.
 */
cl::Device findDevice(const boost::program_options::variables_map &vm);

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

} // namespace CLH

#endif /* !CLH_H */
