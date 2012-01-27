/**
 * @file
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <string>
#include "ply.h"
#include "splat.h"
#include "misc.h"

bool CompareSplatsMorton::operator()(const Splat &a, const Splat &b) const
{
    // bit-casts of the positions of a and b
    std::tr1::uint32_t ap[3], bp[3];
    for (int i = 0; i < 3; i++)
    {
        std::memcpy(&ap[i], &a.position[i], sizeof(ap[i]));
        std::memcpy(&bp[i], &b.position[i], sizeof(bp[i]));
    }
    for (int i = 31; i >= 0; i--)
        for (int j = 0; j < 3; j++)
        {
            std::tr1::uint32_t as = floatToBits(a.position[j]) >> i;
            std::tr1::uint32_t bs = floatToBits(b.position[j]) >> i;
            if (as != bs)
                return as < bs;
        }
    return a.radius < b.radius;
}

Splat CompareSplatsMorton::min_value() const
{
    Splat ans;
    std::memset(&ans, 0, sizeof(ans));
    ans.radius = -1.0f; // this makes it strictly smaller than any real splat
    return ans;
}

Splat CompareSplatsMorton::max_value() const
{
    Splat ans;
    std::memset(&ans, 255, sizeof(ans));
    // This is a NaN encoding of positions, so won't match any real splat
    return ans;
}

void SplatBuilder::validateProperties(const PLY::PropertyTypeSet &properties)
{
    static const char * const names[] = {"radius", "x", "y", "z", "nx", "ny", "nz"};
    for (unsigned int i = 0; i < sizeof(names) / sizeof(names[0]); i++)
    {
        PLY::PropertyTypeSet::index<PLY::Name>::type::const_iterator p;
        p = properties.get<PLY::Name>().find(names[i]);
        if (p == properties.get<PLY::Name>().end())
        {
            throw PLY::FormatError(std::string("Missing property ") + names[i]);
        }
        else if (p->isList)
            throw PLY::FormatError(std::string("Property ") + names[i] + " should not be a list");
    }
}
