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

// Utilities for validating that a mesh is manifold and extracting metadata.
namespace Manifold
{

class Metadata
{
private:
    std::size_t nVertices;
    std::size_t nTriangles;
    std::size_t nComponents;
    std::size_t nBoundaries;

    template<typename ForwardIterator>
    friend std::string isManifold(std::size_t numVertices, ForwardIterator first, ForwardIterator last, Metadata *data = NULL);
public:
    std::size_t vertices() const { return nVertices; }
    std::size_t triangles() const { return nTriangles; }
    std::size_t components() const { return nComponents; }
    std::size_t boundaries() const { return nBoundaries; }

    Metadata();
};

/**
 * Determine whether a triangle mesh is manifold (possibly with boundary).
 *
 * A mesh is considered non-manifold if it has out-of-range indices or
 * isolated vertices (not part of any triangle).
 *
 * @pre @ref finalize() has already been called.
 */
template<typename ForwardIterator>
std::string isManifold(std::size_t numVertices, ForwardIterator first, ForwardIterator last, Metadata *data = NULL)
{
    typedef typename std::iterator_traits<ForwardIterator>::value_type triangle_type;
    typedef typename triangle_type::value_type index_type;
    std::ostringstream reason;

    if (data != NULL)
        *data = Metadata(); // prepare this so we can bail out on error
    Metadata out; // where we stage changes before copying them
    out.nVertices = numVertices;

    // List of edges opposite each vertex
    std::vector<std::vector<std::pair<index_type, index_type> > > edges(numVertices);
    for (ForwardIterator i = first; i != last; i++)
    {
        const triangle_type &triangle = *i;
        index_type indices[3] = {triangle[0], triangle[1], triangle[2]};
        for (unsigned int j = 0; j < 3; j++)
        {
            if (indices[0] >= numVertices)
            {
                reason << "Triangle " << out.nTriangles << " contains out-of-range index " << indices[0] << "\n";
                return reason.str();
            }
            assert(indices[0] < numVertices);
            if (indices[0] == indices[1])
            {
                reason << "Triangle " << out.nTriangles << " contains vertex " << indices[0] << " twice\n";
                return reason.str();
            }
            edges[indices[0]].push_back(std::make_pair(indices[1], indices[2]));
            std::rotate(indices, indices + 1, indices + 3);
        }
        out.nTriangles++;
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

        // Look for a starting point for a line
        index_type start = neigh[0].first;
        for (std::size_t j = 0; j < neigh.size(); j++)
        {
            if (!seen.count(neigh[j].first))
            {
                start = neigh[j].first;
                break;
            }
        }
        std::size_t len = 0;
        index_type cur = start;
        do
        {
            cur = arrow[cur];
            len++;
        } while (arrow.count(cur) && cur != start);
        if (len != neigh.size())
        {
            reason << "Vertex " << i << " contains multiple boundaries\n";
            return reason.str();
        }
    }

    if (data != NULL)
        *data = out;
    return "";
}

} // namespace Manifold

#endif /* !MANIFOLD_H */
