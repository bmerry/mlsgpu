#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <vector>
#include <algorithm>
#include <cstddef>
#include <cassert>
#include <limits>
#include <Eigen/Core>
#include <boost/smart_ptr/scoped_ptr.hpp>
#include "knng.h"
#include "../src/tr1_cstdint.h"

namespace
{

struct KDTree
{
public:
    typedef std::tr1::uint32_t size_type;

    size_type size() const { return points.size(); }

    size_type numNeighbors(size_type idx) const;
    std::pair<float, size_type> getNeighbor(size_type idx, size_type nidx) const;

    template<typename Iterator>
    KDTree(Iterator first, Iterator last, size_type K, float maxDistanceSquared);

private:
    enum
    {
        MIN_LEAF = 4
    };

    struct KDPointBase
    {
        Eigen::Vector3f pos;
        size_type id;

        KDPointBase() {}
        KDPointBase(float x, float y, float z, size_type id) : pos(x, y, z), id(id) {}
    };

    struct KDPoint : public KDPointBase
    {
        size_type numNeighbors;

        KDPoint() : numNeighbors(0) {}
        KDPoint(const KDPointBase &base) : KDPointBase(base), numNeighbors(0) {}
    };

    struct KDNode
    {
        int axis;       ///< Split axis, or -1 for a leaf
        union
        {
            struct
            {
                float split;            ///< Split plane value
                size_type left, right;  ///< Indices of left and right children
            } internal;
            struct
            {
                size_type first, last;  ///< Indices of first and past-the-end points
            } leaf;
        } u;
        float worstSquared;

        KDNode() : axis(-1) {}
        bool isLeaf() const { return axis < 0; }
        size_type left() const { return u.internal.left; }
        size_type right() const { return u.internal.right; }
        size_type first() const { return u.leaf.first; }
        size_type last() const { return u.leaf.last; }
        float split() const { return u.internal.split; }
    };

    class CompareAxis
    {
    private:
        int axis;
        const KDPointBase *points;

    public:
        CompareAxis(int axis, const KDPointBase *points) : axis(axis), points(points) {}

        bool operator()(size_type a, size_type b) const
        {
            return points[a].pos[axis] < points[b].pos[axis];
        }
    };

    size_type K;
    float maxDistanceSquared;
    std::vector<KDPoint> points;
    std::vector<KDNode> nodes;
    std::vector<size_type> reorder; ///< Inverse of the permutation encoded in the points
    std::vector<std::pair<float, size_type> > neighbors;

    void buildTree(size_type N, KDPointBase points[], KDPointBase pointsTmp[],
                   size_type *permute[3], size_type *permuteTmp[3],
                   size_type *remap);

    void updatePair(size_type p, size_type q);

    void knngRecurse(size_type root1, float bbox1[6],
                     size_type root2, float bbox2[6]);
};

KDTree::size_type KDTree::numNeighbors(size_type idx) const
{
    assert(idx < points.size());
    idx = reorder[idx];
    return points[idx].numNeighbors;
}

std::pair<float, KDTree::size_type> KDTree::getNeighbor(size_type idx, size_type nidx) const
{
    assert(idx < points.size());
    idx = reorder[idx];
    assert(nidx < points[idx].numNeighbors);
    return neighbors[idx * K + nidx];
}

void KDTree::buildTree(size_type N, KDPointBase points[], KDPointBase pointsTmp[],
                       size_type *permute[3], size_type *permuteTmp[3],
                       size_type *remap)
{
    size_type nodeId = nodes.size();
    nodes.push_back(KDNode());
    KDNode &n = nodes.back();

    n.worstSquared = maxDistanceSquared;
    if (N >= 2 * MIN_LEAF) // TODO: make tunable
    {
        float maxSpread = -1.0f;
        int axis = -1;
        for (int j = 0; j < 3; j++)
        {
            float lo = points[permute[j][0]].pos[j];
            float hi = points[permute[j][N - 1]].pos[j];
            if (hi - lo > maxSpread)
            {
                maxSpread = hi - lo;
                axis = j;
            }
        }
        assert(axis != -1);

        const size_type H = N / 2;
        float split = points[permute[axis][H]].pos[axis];

        for (size_type i = 0; i < N; i++)
        {
            remap[permute[axis][i]] = i;
            pointsTmp[i] = points[permute[axis][i]];
        }

        for (int a = 0; a < 3; a++)
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
        size_type *subPermute[3], *subPermuteTmp[3];
        for (int j = 0; j < 3; j++)
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

void KDTree::updatePair(size_type p, size_type q)
{
    KDPoint &pp = points[p];
    const KDPoint &pq = points[q];
    float dist2 = (pp.pos - pq.pos).squaredNorm();

    if (dist2 <= maxDistanceSquared)
    {
        bool full = pp.numNeighbors == K;
        size_type first = p * K;
        if (!full || dist2 < neighbors[first].first)
        {
            std::pair<float, size_type> cur(dist2, pq.id);
            if (full)
                std::pop_heap(neighbors.begin() + first, neighbors.begin() + (first + K));
            else
                pp.numNeighbors++;
            neighbors[first + pp.numNeighbors - 1] = cur;
            std::push_heap(neighbors.begin() + first, neighbors.begin() + (first + pp.numNeighbors));
        }
    }
}

static inline float sqr(float x)
{
    return x * x;
}

void KDTree::knngRecurse(size_type root1, float bbox1[6],
                         size_type root2, float bbox2[6])
{
    KDNode &node1 = nodes[root1];
    const KDNode &node2 = nodes[root2];

    float distSquared = 0.0f;
    // TODO: figure out some form of incremental distance computation
    for (int i = 0; i < 3; i++)
    {
        if (bbox1[i * 2] > bbox2[i * 2 + 1])
            distSquared += sqr(bbox1[i * 2] - bbox2[i * 2 + 1]);
        else if (bbox2[i * 2] > bbox1[i * 2 + 1])
            distSquared += sqr(bbox2[i * 2] - bbox1[i * 2 + 1]);
    }
    if (distSquared > node1.worstSquared)
        return;

    if (node1.isLeaf() && node2.isLeaf())
    {
        float wmax = 0.0f;
        for (size_t i = node1.first(); i != node1.last(); ++i)
        {
            for (size_t j = node2.first(); j != node2.last(); ++j)
                if (j != i)
                    updatePair(i, j);
            const KDPoint &pi = points[i];
            float w;
            if (pi.numNeighbors == K)
                w = neighbors[pi.id * K].first;
            else
                w = maxDistanceSquared;
            if (w > wmax)
                wmax = w;
        }
        node1.worstSquared = wmax;
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
            float vol1 = (bbox1[1] - bbox1[0]) * (bbox1[3] - bbox1[2]) * (bbox1[5] - bbox1[4]);
            float vol2 = (bbox2[1] - bbox2[0]) * (bbox2[3] - bbox2[2]) * (bbox2[5] - bbox2[4]);
            split2 = vol2 > vol1;
        }

        if (split2)
        {
            int axis = node2.axis;
            int bl = 2 * axis;
            int br = bl + 1;
            float split = node2.split();
            float mid = 0.5 * (bbox1[bl] + bbox1[br]);
            float L = bbox2[bl];
            float R = bbox2[br];
            if (mid < split)
            {
                bbox2[br] = split;
                knngRecurse(root1, bbox1, node2.left(), bbox2);
                bbox2[br] = R;
                bbox2[bl] = split;
                knngRecurse(root1, bbox1, node2.right(), bbox2);
                bbox2[bl] = L;
            }
            else
            {
                bbox2[bl] = split;
                knngRecurse(root1, bbox1, node2.right(), bbox2);
                bbox2[bl] = L;
                bbox2[br] = split;
                knngRecurse(root1, bbox1, node2.left(), bbox2);
                bbox2[br] = R;
            }
        }
        else
        {
            int axis = node1.axis;
            int bl = 2 * axis;
            int br = 2 * axis + 1;
            float split = node1.split();
            float mid = 0.5 * (bbox2[bl] + bbox2[br]);
            float L = bbox1[bl];
            float R = bbox1[br];
            if (mid < split)
            {
                bbox1[br] = split;
                knngRecurse(node1.left(), bbox1, root2, bbox2);
                bbox1[br] = R;
                bbox1[bl] = split;
                knngRecurse(node1.right(), bbox1, root2, bbox2);
                bbox1[bl] = L;
            }
            else
            {
                bbox1[bl] = split;
                knngRecurse(node1.right(), bbox1, root2, bbox2);
                bbox1[bl] = L;
                bbox1[br] = split;
                knngRecurse(node1.left(), bbox1, root2, bbox2);
                bbox1[br] = R;
            }

            node1.worstSquared = std::max(nodes[node1.left()].worstSquared,
                                          nodes[node1.right()].worstSquared);
        }
    }
}

template<typename Iterator>
KDTree::KDTree(Iterator first, Iterator last, size_type K, float maxDistanceSquared)
    : K(K), maxDistanceSquared(maxDistanceSquared)
{
    std::vector<KDPointBase> pointData[2];
    std::vector<size_type> permuteData[2][3];
    size_type *permute[2][3];

    // Reserve storage
    size_type N = last - first;
    points.reserve(N);
    neighbors.resize(N * K);
    nodes.reserve(2 * (N / MIN_LEAF) + 1);

    float bbox[2][6];
    for (int j = 0; j < 3; j++)
    {
        bbox[0][2 * j] = std::numeric_limits<float>::infinity();
        bbox[0][2 * j + 1] = -std::numeric_limits<float>::infinity();
    }

    // Create transient data structures and compute bbox
    pointData[0].reserve(N);
    size_type idx = 0;
    for (Iterator i = first; i != last; ++i, ++idx)
    {
        pointData[0].push_back(KDPointBase(i->position[0], i->position[1], i->position[2], idx));
        for (int j = 0; j < 3; j++)
        {
            bbox[0][2 * j] = std::min(bbox[0][2 * j], i->position[j]);
            bbox[0][2 * j + 1] = std::max(bbox[0][2 * j + 1], i->position[j]);
        }
    }
    pointData[1].resize(N); // arbitrary data
    std::copy(bbox[0], bbox[0] + 6, bbox[1]);

    for (int axis = 0; axis < 3; axis++)
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
    reorder.resize(N);
    for (size_type i = 0; i < N; i++)
        reorder[points[i].id] = i;

    knngRecurse(0, bbox[0], 0, bbox[1]);
}

} // anonymous namespace

std::vector<std::vector<std::pair<float, int> > > knng(const Statistics::Container::vector<Splat> &splats, int K, float maxDistanceSquared)
{
    KDTree tree(splats.begin(), splats.end(), K, maxDistanceSquared);

    std::vector<std::vector<std::pair<float, int> > > ans;
    ans.reserve(tree.size());
    for (KDTree::size_type i = 0; i < tree.size(); i++)
    {
        int k = tree.numNeighbors(i);
        ans.push_back(std::vector<std::pair<float, int> >(k));
        for (int j = 0; j < k; j++)
        {
            std::pair<float, KDTree::size_type> n = tree.getNeighbor(i, j);
            ans[i][j] = std::make_pair(n.first, int(n.second));
        }
    }
    return ans;
}
