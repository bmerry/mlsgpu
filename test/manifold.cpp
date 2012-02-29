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

Metadata::Metadata() : nVertices(0), nTriangles(0), nComponents(0), nBoundaries(0)
{
}

} // namespace Manifold
