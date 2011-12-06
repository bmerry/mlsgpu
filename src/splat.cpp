/**
 * @file
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <string>
#include "ply.h"
#include "splat.h"

void SplatBuilder::validateProperties(const PLY::PropertyTypeSet &properties)
{
    static const char * const names[] = {"radius", "x", "y", "z", "nx", "ny", "nz"};
    for (unsigned int i = 0; i < sizeof(names) / sizeof(names[0]); i++)
    {
        typename PLY::PropertyTypeSet::index<PLY::Name>::type::const_iterator p;
        p = properties.get<PLY::Name>().find(names[i]);
        if (p == properties.get<PLY::Name>().end())
        {
            throw PLY::FormatError(std::string("Missing property ") + names[i]);
        }
        else if (p->isList)
            throw PLY::FormatError(std::string("Property ") + names[i] + " should not be a list");
    }
}
