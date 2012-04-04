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
#include "ply.h"

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

/**
 * Implementation of the @ref PLY builder concept that loads splats
 * from a PLY file.
 */
class SplatBuilder
{
private:
    Splat current;
    float smooth;

public:
    typedef Splat Element;

    SplatBuilder(float smooth) : smooth(smooth)
    {
        current.quality = std::numeric_limits<float>::quiet_NaN();
    }

    template<typename T> void setProperty(const std::string &name, const T &value)
    {
        if (name == "x") current.position[0] = boost::numeric_cast<float>(value);
        else if (name == "y") current.position[1] = boost::numeric_cast<float>(value);
        else if (name == "z") current.position[2] = boost::numeric_cast<float>(value);
        else if (name == "nx") current.normal[0] = boost::numeric_cast<float>(value);
        else if (name == "ny") current.normal[1] = boost::numeric_cast<float>(value);
        else if (name == "nz") current.normal[2] = boost::numeric_cast<float>(value);
        else if (name == "radius") current.radius = smooth * boost::numeric_cast<float>(value);
        else if (name == "quality") current.quality = boost::numeric_cast<float>(value);
    }

    template<typename Iterator> void setProperty(const std::string &, Iterator, Iterator)
    {
    }

    Element create()
    {
        if ((std::tr1::isnan)(current.quality))
            current.quality = 1.0 / (current.radius * current.radius);
        return current;
    }

    static void validateProperties(const PLY::PropertyTypeSet &properties);
};

#endif /* !SPLAT_H */
