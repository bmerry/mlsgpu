/**
 * @file
 *
 * Implementation of @ref PLY::VertexFetcher and @ref PLY::TriangleFetcher.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif

// Not actually used, but suppresses a warning in cl.hpp.
#ifndef __CL_ENABLE_EXCEPTIONS
# define __CL_ENABLE_EXCEPTIONS
#endif

#include <tr1/cstdint>
#include "ply.h"
#include "ply_mesh.h"
#include "splat.h"

namespace PLY
{

PropertyTypeSet SplatFetcher::getProperties() const
{
    const PropertyType props[] =
    {
        PropertyType("x", FLOAT32),
        PropertyType("y", FLOAT32),
        PropertyType("z", FLOAT32),
        PropertyType("nx", FLOAT32),
        PropertyType("ny", FLOAT32),
        PropertyType("nz", FLOAT32),
        PropertyType("radius", FLOAT32)
    };
    return PropertyTypeSet(props, props + 7);
}

void SplatFetcher::writeElement(const Element &e, Writer &writer) const
{
    writer.writeField<float>(e.position[0]);
    writer.writeField<float>(e.position[1]);
    writer.writeField<float>(e.position[2]);
    writer.writeField<float>(e.normal[0]);
    writer.writeField<float>(e.normal[1]);
    writer.writeField<float>(e.normal[2]);
    writer.writeField<float>(e.radius);
}

PropertyTypeSet VertexFetcher::getProperties() const
{
    const PropertyType props[] =
    {
        PropertyType("x", FLOAT32),
        PropertyType("y", FLOAT32),
        PropertyType("z", FLOAT32)
    };
    return PropertyTypeSet(props, props + 3);
}

void VertexFetcher::writeElement(const Element &e, Writer &writer) const
{
    writer.writeField<float>(e.x);
    writer.writeField<float>(e.y);
    writer.writeField<float>(e.z);
}

PropertyTypeSet TriangleFetcher::getProperties() const
{
    const PropertyType props[] =
    {
        PropertyType("vertex_indices", UINT8, UINT32)
    };
    return PropertyTypeSet(props, props + 1);
}

void TriangleFetcher::writeElement(const Element &e, Writer &writer) const
{
    writer.writeField<std::tr1::uint8_t>(3);
    writer.writeField<std::tr1::uint32_t>(e[0]);
    writer.writeField<std::tr1::uint32_t>(e[1]);
    writer.writeField<std::tr1::uint32_t>(e[2]);
}

} // namespace PLY
