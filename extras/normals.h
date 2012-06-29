/**
 * @file
 *
 * Shared definitions for the normal-computation tool.
 */

#ifndef EXTRAS_NORMALS_H
#define EXTRAS_NORMALS_H

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <Eigen/Core>
#include <vector>

Eigen::Vector3f computeNormal(
    const Splat &s,
    const std::vector<Eigen::Vector3f> &neighbors,
    float &angle,
    float &quality);

namespace Option
{
    static inline const char *help()    { return "help"; }
    static inline const char *quiet()   { return "quiet"; }
    static inline const char *debug()   { return "debug"; }

    static inline const char *inputFile() { return "input-file"; }

    static inline const char *radius()  { return "radius"; }
    static inline const char *neighbors() { return "neighbors"; }
    static inline const char *mode()    { return "mode"; }

    static inline const char *statistics() { return "statistics"; }
    static inline const char *statisticsFile() { return "statistics-file"; }

    static inline const char *reader()  { return "reader"; }
};

#endif /* EXTRAS_NORMALS_H */
