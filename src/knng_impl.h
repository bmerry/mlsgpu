/**
 * Implementation of @ref src/knng.h.
 */

#ifndef KNNG_IMPL_H
#define KNNG_IMPL_H

#if HAVE_CONFIG_H
# include <config.h>
#endif

#include <vector>
#include <algorithm>
#include <memory>
#include <cassert>
#include "knng.h"

/**
 * Transient data structures needed while computing nearest neighbors.
 */
template<typename Coord, typename SizeType>
struct KNNGData
{
    /// Per-node: maximum distance to a kth-nearest neighbor
    std::vector<Coord> worstSquared;

    /// Pointer to the output
    KNNG<Coord, SizeType> *out;

    /// Cutoff distance for finding neighbors
    Coord maxDistanceSquared;
};

template<typename Coord, int Dim, typename SizeType>
void KDTree<Coord, Dim, SizeType>::buildTree(
    size_type N, KDPoint points[], KDPoint pointsTmp[],
    size_type *permute[DIM], size_type *permuteTmp[DIM],
    size_type *remap)
{
    size_type nodeId = nodes.size();
    nodes.push_back(KDNode());
    KDNode &n = nodes.back();

    for (int j = 0; j < DIM; j++)
    {
        n.bbox[j][0] = points[permute[j][0]].pos[j];
        n.bbox[j][1] = points[permute[j][N - 1]].pos[j];
    }
    if (N >= 2 * MIN_LEAF) // TODO: make tunable
    {
        coord_type maxSpread = n.bbox[0][1] - n.bbox[0][0];
        int axis = 0;
        for (int j = 1; j < DIM; j++)
        {
            coord_type diff = n.bbox[j][1] - n.bbox[j][0];
            if (diff > maxSpread)
            {
                maxSpread = diff;
                axis = j;
            }
        }

        const size_type H = N / 2;
        coord_type split = points[permute[axis][H]].pos[axis];

        for (size_type i = 0; i < N; i++)
        {
            remap[permute[axis][i]] = i;
            pointsTmp[i] = points[permute[axis][i]];
        }

        for (int a = 0; a < DIM; a++)
        {
            size_type L = 0;
            size_type R = H;
            for (size_type i = 0; i < N; i++)
            {
                size_type newidx = remap[permute[a][i]];
                if (newidx < H)
                {
                    permuteTmp[a][L] = newidx;
                    L++;
                }
                else
                {
                    permuteTmp[a][R] = newidx - H;
                    R++;
                }
            }
        }

        n.axis = axis;
        n.u.internal.split = split;
        n.u.internal.left = nodes.size();

        buildTree(H, pointsTmp, points, permuteTmp, permute, remap);
        // Can no longer use n: recursive call can invalidate the reference
        size_type *subPermute[DIM], *subPermuteTmp[DIM];
        for (int j = 0; j < DIM; j++)
        {
            subPermute[j] = permute[j] + H;
            subPermuteTmp[j] = permuteTmp[j] + H;
        }
        nodes[nodeId].u.internal.right = nodes.size();
        buildTree(N - H, pointsTmp + H, points + H, subPermuteTmp, subPermute, remap);
    }
    else
    {
        n.u.leaf.first = this->points.size();
        n.u.leaf.last = n.u.leaf.first + N;
        n.axis = -1;
        for (size_type i = 0; i < N; i++)
        {
            this->points.push_back(points[i]);
        }
    }
}

template<typename Coord, int Dim, typename SizeType>
void KDTree<Coord, Dim, SizeType>::updatePairOneWay(
    size_type p, size_type q, Coord distSquared,
    KNNG<Coord, SizeType> &out) const
{
    bool full = out.numNeighbors[p] == out.K;
    std::pair<coord_type, size_type> *nn = &out.neighbors[p * out.K];
    if (!full || distSquared < nn[out.K - 1].first)
    {
        int pos = out.numNeighbors[p];
        while (pos > 0 && distSquared < nn[pos - 1].first)
            pos--;
        if (!full)
            out.numNeighbors[p]++;
        for (int i = out.numNeighbors[p] - 1; i > pos; i--)
            nn[i] = nn[i - 1];
        nn[pos].first = distSquared;
        nn[pos].second = points[q].id;
    }
}

template<typename Coord, int Dim, typename SizeType>
void KDTree<Coord, Dim, SizeType>::updatePair(
    size_type p, size_type q,
    KNNGData<Coord, SizeType> &data) const
{
    Coord distSquared = (points[p].pos - points[q].pos).squaredNorm();
    if (distSquared <= data.maxDistanceSquared)
    {
        updatePairOneWay(p, q, distSquared, *data.out);
        updatePairOneWay(q, p, distSquared, *data.out);
    }
}

template<typename Coord, int Dim, typename SizeType>
void KDTree<Coord, Dim, SizeType>::updateWorstSquared(size_type nodeIdx, KNNGData<Coord, SizeType> &data) const
{
    coord_type wmax = 0;
    const KDNode &node = nodes[nodeIdx];
    size_type K = data.out->K;
    for (size_type i = node.first(); i < node.last(); i++)
    {
        const KDPoint &pi = points[i];
        Coord w;
        if (data.out->numNeighbors[i] == K)
            w = data.out->neighbors[pi.id * K + K - 1].first;
        else
            w = data.maxDistanceSquared;
        if (w > wmax)
            wmax = w;
    }
    data.worstSquared[nodeIdx] = wmax;
}

template<typename Coord, int Dim, typename SizeType>
void KDTree<Coord, Dim, SizeType>::knngRecurse(
    size_type root1, size_type root2,
    KNNGData<Coord, SizeType> &data) const
{
    const KDNode &node1 = nodes[root1];
    const KDNode &node2 = nodes[root2];

    coord_type distSquared = coord_type();
    for (int i = 0; i < DIM; i++)
    {
        if (node1.bbox[i][0] > node2.bbox[i][1])
        {
            coord_type d = node1.bbox[i][0] - node2.bbox[i][1];
            distSquared += d * d;
        }
        else if (node2.bbox[i][0] > node1.bbox[i][1])
        {
            coord_type d = node2.bbox[i][0] - node1.bbox[i][1];
            distSquared += d * d;
        }
    }
    if (distSquared > std::max(data.worstSquared[root1], data.worstSquared[root2]))
        return;

    if (node1.isLeaf() && node2.isLeaf())
    {
        if (root1 == root2)
        {
            for (size_t i = node1.first(); i < node1.last(); i++)
                for (size_t j = i + 1; j < node1.last(); j++)
                    updatePair(i, j, data);
            updateWorstSquared(root1, data);
        }
        else
        {
            for (size_t i = node1.first(); i != node1.last(); ++i)
                for (size_t j = node2.first(); j != node2.last(); ++j)
                    updatePair(i, j, data);
            updateWorstSquared(root1, data);
            updateWorstSquared(root2, data);
        }
    }
    else if (root1 == root2)
    {
        // Split both
        knngRecurse(node1.left(), node2.left(), data);
        knngRecurse(node1.right(), node2.right(), data);
        knngRecurse(node1.right(), node2.left(), data);
    }
    else
    {
        bool split2;
        if (node1.isLeaf())
            split2 = true;
        else if (node2.isLeaf())
            split2 = false;
        else
        {
            coord_type vol1 = 1;
            coord_type vol2 = 1;
            for (int i = 0; i < DIM; i++)
            {
                vol1 *= node1.bbox[i][1] - node1.bbox[i][0];
                vol2 *= node2.bbox[i][1] - node2.bbox[i][0];
            }
            split2 = vol2 > vol1;
        }

        if (split2)
        {
            int axis = node2.axis;
            coord_type split = node2.split();
            coord_type mid = coord_type(0.5) * (node1.bbox[axis][0] + node1.bbox[axis][1]);
            if (mid < split)
            {
                knngRecurse(root1, node2.left(), data);
                knngRecurse(root1, node2.right(), data);
            }
            else
            {
                knngRecurse(root1, node2.right(), data);
                knngRecurse(root1, node2.left(), data);
            }

            // TODO: need to udpate node2.worstSquared!
        }
        else
        {
            int axis = node1.axis;
            coord_type split = node1.split();
            coord_type mid = coord_type(0.5) * (node2.bbox[axis][0] + node2.bbox[axis][1]);
            if (mid < split)
            {
                knngRecurse(node1.left(), root2, data);
                knngRecurse(node1.right(), root2, data);
            }
            else
            {
                knngRecurse(node1.right(), root2, data);
                knngRecurse(node1.left(), root2, data);
            }

            data.worstSquared[root1] =
                std::max(data.worstSquared[node1.left()],
                         data.worstSquared[node1.right()]);
        }
    }
}

template<typename CoordType, typename SizeType>
KNNG<CoordType, SizeType>::KNNG(size_type N, size_type K)
    : N(N), K(K), reorder(N), neighbors(std::size_t(N) * K), numNeighbors(N)
{
    // Friend class populates the values
}

template<typename CoordType, int Dim, typename SizeType>
template<typename Iterator>
KDTree<CoordType, Dim, SizeType>::KDTree(Iterator first, Iterator last)
{
    std::vector<KDPoint> pointData[2];
    std::vector<size_type> permuteData[2][DIM];
    size_type *permute[2][DIM];

    // Reserve storage
    size_type N = last - first;
    points.reserve(N);
    nodes.reserve(2 * (N / MIN_LEAF) + 1);

    // Create transient data structures
    pointData[0].reserve(N);
    size_type idx = 0;
    for (Iterator i = first; i != last; ++i, ++idx)
    {
        pointData[0].push_back(KDPoint(*i, idx));
    }
    pointData[1].resize(N); // arbitrary data

    for (int axis = 0; axis < DIM; axis++)
    {
        permuteData[1][axis].resize(N); // arbitrary data
        permuteData[0][axis].reserve(N);
        for (size_type i = 0; i < N; i++)
            permuteData[0][axis].push_back(i);
        std::sort(permuteData[0][axis].begin(), permuteData[0][axis].end(), CompareAxis(axis, &pointData[0][0]));

        permute[0][axis] = &permuteData[0][axis][0];
        permute[1][axis] = &permuteData[1][axis][0];
    }

    std::vector<size_type> remap(N); // arbitrary data

    buildTree(N, &pointData[0][0], &pointData[1][0], permute[0], permute[1], &remap[0]);
}

template<typename CoordType, int Dim, typename SizeType>
KNNG<CoordType, SizeType> *KDTree<CoordType, Dim, SizeType>::knn(
    size_type K, coord_type maxDistanceSquared) const
{
    size_type N = size();
    std::auto_ptr<KNNG<CoordType, SizeType> > ans(new KNNG<CoordType, SizeType>(N, K));
    KNNGData<CoordType, SizeType> data;
    data.out = ans.get();
    data.maxDistanceSquared = maxDistanceSquared;

    for (size_type i = 0; i < N; i++)
        ans->reorder[points[i].id] = i;

    data.worstSquared.resize(nodes.size());
    for (size_type i = 0; i < nodes.size(); i++)
        data.worstSquared[i] = maxDistanceSquared;

    knngRecurse(0, 0, data);
    return ans.release();
}

#endif /* !KNNG_IMPL_H */
