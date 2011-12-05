/**
 * @file
 */

#ifndef CLH_H
#define CLH_H

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <boost/program_options.hpp>
#include <CL/cl.hpp>

namespace CLH
{

namespace Option
{
const char * const device = "cl-device";
const char * const gpu = "cl-gpu";
const char * const cpu = "cl-cpu";
} // namespace Option

/// Program options for selecting an OpenCL device
boost::program_options::options_description getOptions();

/**
 * Pick an OpenCL device based on command-line options.
 *
 * If more than one device matches the criteria, GPU devices are preferred.
 * If there is no exact match for the device name, a prefix will be accepted.
 *
 * @return A device matching the command-line options, or @c NULL if none matches.
 */
cl::Device findDevice(const boost::program_options::variables_map &vm);

} // namespace CLH

#endif /* !CLH_H */
