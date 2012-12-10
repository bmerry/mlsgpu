/**
 * @file
 */

#ifndef SPLAT_H
#define SPLAT_H

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <string>
#include <limits>
#include <cmath>
#include <boost/tr1/cmath.hpp>
#include <boost/numeric/conversion/cast.hpp>

/**
 * A point together with a normal and a radius of influence.
 */
struct Splat
{
    float position[3];
    float radius;
    float normal[3];
    float quality;

    /**
     * Checks whether all attributes are finite.
     */
    inline bool isFinite() const
    {
        return (std::tr1::isfinite)(position[0])
            && (std::tr1::isfinite)(position[1])
            && (std::tr1::isfinite)(position[2])
            && (std::tr1::isfinite)(radius)
            && (std::tr1::isfinite)(normal[0])
            && (std::tr1::isfinite)(normal[1])
            && (std::tr1::isfinite)(normal[2])
            && (std::tr1::isfinite)(quality);
    }
};

/**
 * Comparator that orders splats by position to give good spatial
 * locality.
 */
class CompareSplatsMorton
{
public:
    bool operator()(const Splat &a, const Splat &b) const;
    Splat min_value() const;
    Splat max_value() const;
};

#endif /* !SPLAT_H */
