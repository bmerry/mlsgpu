/**
 * @file
 *
 * Utility code for validating that a mesh is manifold and extracting metadata.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif

#include "manifold.h"

namespace Manifold
{

Metadata::Metadata() : numVertices(0), numTriangles(0), numComponents(0), numBoundaries(0)
{
}

} // namespace Manifold
