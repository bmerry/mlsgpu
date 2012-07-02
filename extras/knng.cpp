#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <vector>
#include <algorithm>
#include <cstddef>
#include <cassert>
#include <cstring>
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

    std::vector<std::pair<float, size_type> > getNeighbors(size_type idx) const;

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
        float bbox[3][2];
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
    void updatePairOneWay(size_type p, size_type q, float distSquared);
    void updateWorstSquared(KDNode &node);

    void knngRecurse(size_type root1, size_type root2);
};

std::vector<std::pair<float, KDTree::size_type> > KDTree::getNeighbors(size_type idx) const
{
    assert(idx < points.size());
    idx = reorder[idx];
    size_type first = idx * K;
    return std::vector<std::pair<float, KDTree::size_type> >(
        neighbors.begin() + first,
        neighbors.begin() + (first + points[idx].numNeighbors));
}

void KDTree::buildTree(size_type N, KDPointBase points[], KDPointBase pointsTmp[],
                       size_type *permute[3], size_type *permuteTmp[3],
                       size_type *remap)
{
    size_type nodeId = nodes.size();
    nodes.push_back(KDNode());
    KDNode &n = nodes.back();

    n.worstSquared = maxDistanceSquared;
    for (int j = 0; j < 3; j++)
    {
        n.bbox[j][0] = points[permute[j][0]].pos[j];
        n.bbox[j][1] = points[permute[j][N - 1]].pos[j];
    }
    if (N >= 2 * MIN_LEAF) // TODO: make tunable
    {
        float maxSpread = -1.0f;
        int axis = -1;
        for (int j = 0; j < 3; j++)
        {
            float lo = n.bbox[j][0];
            float hi = n.bbox[j][1];
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

void KDTree::updatePairOneWay(size_type p, size_type q, float distSquared)
{
    KDPoint &pp = points[p];
    const KDPoint &pq = points[q];

    bool full = pp.numNeighbors == K;
    std::pair<float, size_type> *nn = &neighbors[p * K];
    if (!full || distSquared < nn[K - 1].first)
    {
        int p = pp.numNeighbors;
        while (p > 0 && distSquared < nn[p - 1].first)
            p--;
        if (!full)
            pp.numNeighbors++;
        for (int i = pp.numNeighbors - 1; i > p; i--)
            nn[i] = nn[i - 1];
        nn[p].first = distSquared;
        nn[p].second = pq.id;
    }
}

void KDTree::updatePair(size_type p, size_type q)
{
    float distSquared = (points[p].pos - points[q].pos).squaredNorm();
    if (distSquared <= maxDistanceSquared)
    {
        updatePairOneWay(p, q, distSquared);
        updatePairOneWay(q, p, distSquared);
    }
}

static inline float sqr(float x)
{
    return x * x;
}

void KDTree::updateWorstSquared(KDNode &node)
{
    float wmax = 0.0f;
    for (size_type i = node.first(); i < node.last(); i++)
    {
        const KDPoint &pi = points[i];
        float w;
        if (pi.numNeighbors == K)
            w = neighbors[pi.id * K + K - 1].first;
        else
            w = maxDistanceSquared;
        if (w > wmax)
            wmax = w;
    }
    node.worstSquared = wmax;
}

void KDTree::knngRecurse(size_type root1, size_type root2)
{
    KDNode &node1 = nodes[root1];
    KDNode &node2 = nodes[root2];

    float distSquared = 0.0f;
    for (int i = 0; i < 3; i++)
    {
        if (node1.bbox[i][0] > node2.bbox[i][1])
            distSquared += sqr(node1.bbox[i][0] - node2.bbox[i][1]);
        else if (node2.bbox[i][0] > node1.bbox[i][1])
            distSquared += sqr(node2.bbox[i][0] - node1.bbox[i][1]);
    }
    if (distSquared > std::max(node1.worstSquared, node2.worstSquared))
        return;

    if (node1.isLeaf() && node2.isLeaf())
    {
        if (root1 == root2)
        {
            for (size_t i = node1.first(); i < node1.last(); i++)
                for (size_t j = i + 1; j < node1.last(); j++)
                    updatePair(i, j);
            updateWorstSquared(node1);
        }
        else
        {
            for (size_t i = node1.first(); i != node1.last(); ++i)
                for (size_t j = node2.first(); j != node2.last(); ++j)
                    updatePair(i, j);
            updateWorstSquared(node1);
            updateWorstSquared(node2);
        }
    }
    else if (root1 == root2)
    {
        // Split both
        knngRecurse(node1.left(), node2.left());
        knngRecurse(node1.right(), node2.right());
        knngRecurse(node1.right(), node2.left());
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
            float vol1 = 1.0f;
            float vol2 = 1.0f;
            for (int i = 0; i < 3; i++)
            {
                vol1 *= node1.bbox[i][1] - node1.bbox[i][0];
                vol2 *= node2.bbox[i][1] - node2.bbox[i][0];
            }
            split2 = vol2 > vol1;
        }

        if (split2)
        {
            int axis = node2.axis;
            float split = node2.split();
            float mid = 0.5 * (node1.bbox[axis][0] + node1.bbox[axis][1]);
            if (mid < split)
            {
                knngRecurse(root1, node2.left());
                knngRecurse(root1, node2.right());
            }
            else
            {
                knngRecurse(root1, node2.right());
                knngRecurse(root1, node2.left());
            }
        }
        else
        {
            int axis = node1.axis;
            float split = node1.split();
            float mid = 0.5 * (node2.bbox[axis][0] + node2.bbox[axis][1]);
            if (mid < split)
            {
                knngRecurse(node1.left(), root2);
                knngRecurse(node1.right(), root2);
            }
            else
            {
                knngRecurse(node1.right(), root2);
                knngRecurse(node1.left(), root2);
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

    // Create transient data structures
    pointData[0].reserve(N);
    size_type idx = 0;
    for (Iterator i = first; i != last; ++i, ++idx)
    {
        pointData[0].push_back(KDPointBase(i->position[0], i->position[1], i->position[2], idx));
    }
    pointData[1].resize(N); // arbitrary data

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

    knngRecurse(0, 0);
}

} // anonymous namespace

std::vector<std::vector<std::pair<float, std::tr1::uint32_t> > > knng(const Statistics::Container::vector<Splat> &splats, int K, float maxDistanceSquared)
{
    std::vector<std::vector<std::pair<float, std::tr1::uint32_t> > > ans;
    if (splats.empty())
        return ans;

    KDTree tree(splats.begin(), splats.end(), K, maxDistanceSquared);
    ans.reserve(tree.size());
    for (KDTree::size_type i = 0; i < tree.size(); i++)
        ans.push_back(tree.getNeighbors(i));
    return ans;
}
