/**
 * @file
 *
 * Marching tetrahedra algorithms.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif

#ifndef __CL_ENABLE_EXCEPTIONS
# define __CL_ENABLE_EXCEPTIONS
#endif

#include <vector>
#include <utility>
#include <algorithm>
#include <cassert>
#include "marching.h"

const unsigned char Marching::edgeIndices[NUM_EDGES][2] =
{
    {0, 1},
    {0, 2},
    {0, 3},
    {1, 3},
    {2, 3},
    {0, 4},
    {0, 5},
    {1, 5},
    {4, 5},
    {0, 6},
    {2, 6},
    {4, 6},
    {0, 7},
    {1, 7},
    {2, 7},
    {3, 7},
    {4, 7},
    {5, 7},
    {6, 7}
};

const unsigned char Marching::tetrahedronIndices[NUM_TETRAHEDRA][4] =
{
    { 0, 7, 1, 3 },
    { 0, 7, 3, 2 },
    { 0, 7, 2, 6 },
    { 0, 7, 6, 4 },
    { 0, 7, 4, 5 },
    { 0, 7, 5, 1 }
};

unsigned int Marching::findEdgeByVertexIds(unsigned int v0, unsigned int v1)
{
    if (v0 > v1) std::swap(v0, v1);
    for (unsigned int i = 0; i < NUM_EDGES; i++)
        if (edgeIndices[i][0] == v0 && edgeIndices[i][1] == v1)
            return i;
    assert(false);
    return -1;
}

template<typename Iterator>
unsigned int Marching::permutationParity(Iterator first, Iterator last)
{
    unsigned int parity = 0;
    // Simple implementation, since we will only ever call it with 4
    // items.
    for (Iterator i = first; i != last; ++i)
    {
        Iterator j = i;
        for (++j; j != last; ++j)
            if (*i > *j)
                parity ^= 1;
    }
    return parity;
}

void Marching::makeTables(const cl::Context &context)
{
    std::vector<cl_uchar> hVertexTable, hIndexTable;
    std::vector<cl_uchar2> hCountTable(NUM_CUBES);
    std::vector<cl_ushort2> hStartTable(NUM_CUBES + 1);
    for (unsigned int i = 0; i < NUM_CUBES; i++)
    {
        hStartTable[i].s0 = hVertexTable.size();
        hStartTable[i].s1 = hIndexTable.size();

        /* Find triangles. For now we record triangle indices
         * as edge numbers, which we will compact later.
         */
        std::vector<cl_uchar> triangles;
        for (unsigned int j = 0; j < NUM_TETRAHEDRA; j++)
        {
            // Holds a vertex index together with the inside/outside flag
            typedef std::pair<unsigned char, bool> tvtx;
            tvtx tvtxs[4];
            unsigned int outside = 0;
            // Copy the vertices to tvtxs, and count vertices that are external
            for (unsigned int k = 0; k < 4; k++)
            {
                unsigned int v = tetrahedronIndices[j][k];
                bool o = (i & (1 << v));
                outside += o;
                tvtxs[k] = tvtx(v, o);
            }
            unsigned int baseParity = permutationParity(tvtxs, tvtxs + 4);

            // Reduce number of cases to handle by flipping inside/outside to
            // ensure that outside <= 2.
            if (outside > 2)
            {
                // Causes triangle winding to flip as well - otherwise
                // the triangle will be inside out.
                baseParity ^= 1;
                for (unsigned int k = 0; k < 4; k++)
                    tvtxs[k].second = !tvtxs[k].second;
            }

            /* To reduce the number of cases to handle, the tetrahedron is
             * rotated to match one of the canonical configurations (all
             * inside, v0 outside, (v0, v1) outside). There are 24 permutations
             * of the vertices, half of which are rotations and half of which are
             * reflections. Not all of them need to be tried, but this code isn't
             * performance critical.
             */
            sort(tvtxs, tvtxs + 4);
            do
            {
                // Check that it is a rotation rather than a reflection
                if (permutationParity(tvtxs, tvtxs + 4) == baseParity)
                {
                    const unsigned int t0 = tvtxs[0].first;
                    const unsigned int t1 = tvtxs[1].first;
                    const unsigned int t2 = tvtxs[2].first;
                    const unsigned int t3 = tvtxs[3].first;
                    unsigned int mask = 0;
                    for (unsigned int k = 0; k < 4; k++)
                        mask |= tvtxs[k].second << k;
                    if (mask == 0)
                    {
                        break; // no outside vertices, so no triangles needed
                    }
                    else if (mask == 1)
                    {
                        // One outside vertex, one triangle needed
                        triangles.push_back(findEdgeByVertexIds(t0, t1));
                        triangles.push_back(findEdgeByVertexIds(t0, t3));
                        triangles.push_back(findEdgeByVertexIds(t0, t2));
                        break;
                    }
                    else if (mask == 3)
                    {
                        // Two outside vertices, two triangles needed to tile a quad
                        triangles.push_back(findEdgeByVertexIds(t0, t2));
                        triangles.push_back(findEdgeByVertexIds(t1, t2));
                        triangles.push_back(findEdgeByVertexIds(t1, t3));

                        triangles.push_back(findEdgeByVertexIds(t1, t3));
                        triangles.push_back(findEdgeByVertexIds(t0, t3));
                        triangles.push_back(findEdgeByVertexIds(t0, t2));
                        break;
                    }
                }
            } while (next_permutation(tvtxs, tvtxs + 4));
        }

        // Determine which edges are in use, and assign indices
        int edgeCompact[NUM_EDGES];
        int pool = 0;
        for (unsigned int j = 0; j < NUM_EDGES; j++)
        {
            if (std::count(triangles.begin(), triangles.end(), j))
            {
                edgeCompact[j] = pool++;
                hVertexTable.push_back(j);
            }
        }
        for (unsigned int j = 0; j < triangles.size(); j++)
        {
            hIndexTable.push_back(edgeCompact[triangles[j]]);
        }

        hCountTable[i].s0 = hVertexTable.size() - hStartTable[i].s0;
        hCountTable[i].s1 = hIndexTable.size() - hStartTable[i].s1;
    }

    hStartTable[NUM_CUBES].s0 = hVertexTable.size();
    hStartTable[NUM_CUBES].s1 = hIndexTable.size();

    /* We're going to concatenate hVertexTable and hIndexTable, so the start values
     * need to be offset to point to where hIndexTable sits afterwards.
     */
    for (unsigned int i = 0; i <= NUM_CUBES; i++)
    {
        hStartTable[i].s1 += hVertexTable.size();
    }
    // Concatenate the two tables into one
    hVertexTable.insert(hVertexTable.end(), hIndexTable.begin(), hIndexTable.end());

    countTable = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            hCountTable.size() * sizeof(hCountTable[0]), &hCountTable[0]);
    startTable = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            hStartTable.size() * sizeof(hStartTable[0]), &hStartTable[0]);
    dataTable =  cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            hVertexTable.size() * sizeof(hVertexTable), &hVertexTable[0]);
}
