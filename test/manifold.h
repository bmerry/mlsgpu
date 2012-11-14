/**
 * @file
 *
 * Utility code for validating that a mesh is manifold and extracting metadata.
 */

#ifndef MANIFOLD_H
#define MANIFOLD_H

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <cstddef>
#include <vector>
#include <map>
#include <set>
#include <utility>
#include <string>
#include <algorithm>
#include <sstream>
#include <cassert>
#include "../src/tr1_cstdint.h"
#include <boost/type_traits/make_signed.hpp>
#include "../src/union_find.h"

/// Utilities for validating that a mesh is manifold and extracting metadata.
namespace Manifold
{

/**
 * Returned data from @ref isManifold.
 */
struct Metadata
{
    std::size_t numVertices;   ///< Number of vertices in the mesh.
    std::size_t numTriangles;  ///< Number of triangles in the mesh.
    /**
     * Number of connected components in the mesh.
     *
     * This is equal to the number of connected components in the graph
     * consisting of the vertices and edges. Note that this may be different
     * to a topologically-inspired definition in which two triangles sharing
     * only a vertex are not considered to be adjacent.
     */
    std::size_t numComponents;

    /**
     * Number of boundary loops in the mesh. See @ref isManifold
     * for a clarification about what constitutes a boundary loop.
     */
    std::size_t numBoundaries;

    /// Constructor that sets all fields to zero.
    Metadata();
};

/**
 * Determine whether a triangle mesh is an oriented manifold with boundary.
 *
 * A mesh is considered non-manifold if it has out-of-range indices or
 * isolated vertices (not part of any triangle). Otherwise, the edges
 * opposite each vertex must form either a ring or a disjoint collection of
 * linear runs. Note that this is more general than the topological
 * definition of a manifold, since it allows a single vertex to sit on multiple
 * boundary loops. This situation does occur when removing arbitrary triangles
 * from a manifold. In the returned metadata, this case is considered to be a
 * single boundary.
 *
 * @param numVertices  The number of vertices referenced by the triangles.
 * @param first, last  An input range where each element is a random access container
 *                     of three elements (normally <code>boost::array</code>) each of
 *                     which is an unsigned integral vertex index.
 * @param[out] data    If non-NULL, it is populated with metadata about the mesh on
 *                     output. If the mesh is not manifold, it is set to all zeros.
 * @return The empty string if the mesh is manifold, or a human-readable
 * explanation if it is non-manifold.
 */
template<typename InputIterator>
std::string isManifold(std::size_t numVertices, InputIterator first, InputIterator last, Metadata *data = NULL)
{
    typedef typename std::iterator_traits<InputIterator>::value_type triangle_type;
    typedef typename triangle_type::value_type index_type;
    std::ostringstream reason;
    std::vector<UnionFind::Node<std::tr1::int64_t> > components(numVertices);

    if (data != NULL)
        *data = Metadata(); // prepare this so we can bail out on error
    Metadata out; // where we stage changes before copying them
    out.numVertices = numVertices;

    // List of edges opposite each vertex
    std::vector<std::vector<std::pair<index_type, index_type> > > edges(numVertices);
    for (InputIterator i = first; i != last; ++i)
    {
        const triangle_type &triangle = *i;
        index_type indices[3] = {triangle[0], triangle[1], triangle[2]};
        for (unsigned int j = 0; j < 3; j++)
        {
            if (indices[0] >= numVertices)
            {
                reason << "Triangle " << out.numTriangles << " contains out-of-range index " << indices[0] << "\n";
                return reason.str();
            }
            assert(indices[0] < numVertices);
            if (indices[0] == indices[1])
            {
                reason << "Triangle " << out.numTriangles << " contains vertex " << indices[0] << " twice\n";
                return reason.str();
            }
            edges[indices[0]].push_back(std::make_pair(indices[1], indices[2]));
            std::rotate(indices, indices + 1, indices + 3);
        }
        UnionFind::merge(components, indices[0], indices[1]);
        UnionFind::merge(components, indices[1], indices[2]);
        out.numTriangles++;
    }

    // Count components
    for (std::size_t i = 0; i < numVertices; i++)
    {
        if (components[i].isRoot())
            out.numComponents++;
        components[i] = UnionFind::Node<std::tr1::int64_t>(); // reset to use for counting boundaries
    }

    // Now check that the neighborhood of each vertex is a line or ring
    for (std::size_t i = 0; i < numVertices; i++)
    {
        const std::vector<std::pair<index_type, index_type> > &neigh = edges[i];
        if (neigh.empty())
        {
            // disallow isolated vertices
            reason << "Vertex " << i << " is isolated\n";
            return reason.str();
        }
        std::map<index_type, index_type> arrow; // maps .first to .second
        std::set<index_type> seen; // .second that have been observed
        for (std::size_t j = 0; j < neigh.size(); j++)
        {
            index_type x = neigh[j].first;
            index_type y = neigh[j].second;
            if (arrow.count(x))
            {
                reason << "Edge " << i << " - " << x << " occurs twice with same winding\n";
                return reason.str();
            }
            arrow[x] = y;
            if (seen.count(y))
            {
                reason << "Edge " << y << " - " << i << " occurs twice with same winding\n";
                return reason.str();
            }
            seen.insert(y);
        }

        /* At this point, we have in-degree and out-degree of at most 1 for
         * each vertex, so we have a collection of lines and rings.
         */

        // Find lines
        std::size_t len = 0;
        for (std::size_t j = 0; j < neigh.size(); j++)
        {
            if (!seen.count(neigh[j].first))
            {
                index_type first = neigh[j].first;
                index_type cur = first;
                while (arrow.count(cur))
                {
                    cur = arrow[cur];
                    len++;
                }
                // track boundary loops
                UnionFind::merge(components, first, i);
                UnionFind::merge(components, cur, i);
            }
        }
        if (len != 0 && len != neigh.size())
        {
            // There were lines but they didn't cover everything.
            reason << "Vertex " << i << " is both in the interior and on the boundary\n";
            return reason.str();
        }
        else if (len == 0)
        {
            // There are only rings. Check that there is exactly one.
            index_type start = neigh[0].first;
            index_type cur = start;
            do
            {
                cur = arrow[cur];
                len++;
            } while (cur != start);
            if (len != neigh.size())
            {
                reason << "Vertex " << i << " tunnels between interior regions\n";
                return reason.str();
            }
        }
    }

    // Count boundaries
    for (std::size_t i = 0; i < numVertices; i++)
    {
        if (components[i].isRoot() && components[i].size() >= 3)
            out.numBoundaries++;
    }

    if (data != NULL)
        *data = out;
    return "";
}

} // namespace Manifold

#endif /* !MANIFOLD_H */
