#if HAVE_CONFIG_H
# include <config.h>
#endif
#include "knng.h"
#include <vector>
#include <algorithm>
#include <cstddef>
#include <cassert>
#include <limits>
#include <Eigen/Core>
#include <boost/smart_ptr/scoped_ptr.hpp>

namespace
{

struct KDPoint
{
    Eigen::Vector3f pos;
    int id;

    KDPoint(float x, float y, float z, int id) : pos(x, y, z), id(id) {}
};

struct KDNode
{
    int axis;
    float split;
    boost::scoped_ptr<KDNode> left, right;
    KDPoint *first, *last;
    float worstSquared;

    KDNode() : split(0.0f), left(NULL), right(NULL), first(NULL), last(NULL) {}

    bool isLeaf() const { return !left; }
};

class CompareAxis
{
private:
    int axis;
    const KDPoint *points;

public:
    CompareAxis(int axis, const KDPoint *points) : axis(axis), points(points) {}

    bool operator()(int a, int b) const
    {
        return points[a].pos[axis] < points[b].pos[axis];
    }
};

KDNode *buildTree(int N, KDPoint points[], KDPoint pointsTmp[],
                  int *permute[3], int *permuteTmp[3],
                  int *remap, float worstSquared)
{
    std::auto_ptr<KDNode> n(new KDNode);

    n->first = points;
    n->last = points + N;
    n->worstSquared = worstSquared;
    if (N > 8)
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

        const int H = N / 2;
        float split = points[permute[axis][H]].pos[axis];

        for (int i = 0; i < N; i++)
        {
            remap[permute[axis][i]] = i;
            pointsTmp[i] = points[permute[axis][i]];
        }
        std::copy(pointsTmp, pointsTmp + N, points);

        for (int a = 0; a < 3; a++)
        {
            int L = 0;
            int R = H;
            for (int i = 0; i < N; i++)
            {
                int newidx = remap[permute[a][i]];
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

        n->split = split;
        n->axis = axis;
        n->left.reset(buildTree(H, points, pointsTmp, permuteTmp, permute, remap, worstSquared));
        int *subPermute[3], *subPermuteTmp[3];
        for (int j = 0; j < 3; j++)
        {
            subPermute[j] = permute[j] + H;
            subPermuteTmp[j] = permuteTmp[j] + H;
        }
        // TODO: can we reuse temp data from part 1 to improve cache hits?
        n->right.reset(buildTree(N - H, points + H, pointsTmp + H, subPermuteTmp, subPermute, remap + H, worstSquared));

#if DEBUGGING
        assert(n->left->first == n->first);
        assert(n->left->last == n->first + H);
        assert(n->right->first == n->left->last);
        assert(n->right->last == n->last);
        for (int i = 0; i < H; i++)
            assert(n->left->first[i].pos[axis] <= split);
        for (int i = 0; i < N - H; i++)
            assert(n->right->first[i].pos[axis] >= split);
#endif
    }
    return n.release();
}

void updatePair(std::vector<std::vector<std::pair<float, int> > > &ans,
                 KDPoint *p, const KDPoint *q, float maxDistanceSquared)
{
    float dist2 = (p->pos - q->pos).squaredNorm();
    std::vector<std::pair<float, int> > &a = ans[p->id];
    if (dist2 <= maxDistanceSquared)
    {
        bool full = a.size() == a.capacity();
        if (!full || dist2 < a[0].first)
        {
            std::pair<float, int> cur(dist2, q->id);
            if (full)
            {
                std::pop_heap(a.begin(), a.end());
                a.back() = cur;
            }
            else
                a.push_back(cur);
            std::push_heap(a.begin(), a.end());
        }
    }
}

float sqr(float x)
{
    return x * x;
}

void knngRecurse(std::vector<std::vector<std::pair<float, int> > > &ans,
                 KDNode *root1, float bbox1[6], KDNode *root2, float bbox2[6],
                 float maxDistanceSquared)
{
#if DEBUGGING
    float wtest = 0.0f;
    for (KDPoint *i = root1->first; i != root1->last; ++i)
    {
        std::vector<std::pair<float, int> > &a = ans[i->id];
        float w;
        assert(a.capacity() == 16);
        if (a.size() == a.capacity())
            w = a[0].first;
        else
            w = maxDistanceSquared;
        wtest = std::max(wtest, w);
    }
    assert(root1->worstSquared == wtest);
#endif

    float dist2 = 0.0f;
    for (int i = 0; i < 3; i++)
    {
        if (bbox1[i * 2] > bbox2[i * 2 + 1])
            dist2 += sqr(bbox1[i * 2] - bbox2[i * 2 + 1]);
        else if (bbox1[i * 2 + 1] < bbox2[i * 2])
            dist2 += sqr(bbox2[i * 2] - bbox1[i * 2 + 1]);
    }
    if (dist2 > root1->worstSquared)
        return;

    if (root1->isLeaf() && root2->isLeaf())
    {
        float wmax = 0.0f;
        for (KDPoint *i = root1->first; i != root1->last; ++i)
        {
#if DEBUGGING
            assert(i->pos[0] >= bbox1[0] && i->pos[0] <= bbox1[1]);
            assert(i->pos[1] >= bbox1[2] && i->pos[1] <= bbox1[3]);
            assert(i->pos[2] >= bbox1[4] && i->pos[2] <= bbox1[5]);
#endif
            for (const KDPoint *j = root2->first; j != root2->last; ++j)
                if (j != i)
                    updatePair(ans, i, j, maxDistanceSquared);
            std::vector<std::pair<float, int> > &a = ans[i->id];
            float w;
            if (a.size() == a.capacity())
                w = a[0].first;
            else
                w = maxDistanceSquared;
            if (w > wmax)
                wmax = w;
        }
        root1->worstSquared = wmax;
    }
    else
    {
        bool split2;
        if (root1->isLeaf())
            split2 = true;
        else if (root2->isLeaf())
            split2 = false;
        else
        {
            float vol1 = (bbox1[1] - bbox1[0]) * (bbox1[3] - bbox1[2]) * (bbox1[5] - bbox1[4]);
            float vol2 = (bbox2[1] - bbox2[0]) * (bbox2[3] - bbox2[2]) * (bbox2[5] - bbox2[4]);
            split2 = vol2 > vol1;
        }

        if (split2)
        {
            int axis = root2->axis;
            int bl = 2 * axis;
            int br = bl + 1;
            float split = root2->split;
            float mid = 0.5 * (bbox1[bl] + bbox1[br]);
            float L = bbox2[bl];
            float R = bbox2[br];
            if (mid < split)
            {
                bbox2[br] = split;
                knngRecurse(ans, root1, bbox1, root2->left.get(), bbox2, maxDistanceSquared);
                bbox2[br] = R;
                bbox2[bl] = split;
                knngRecurse(ans, root1, bbox1, root2->right.get(), bbox2, maxDistanceSquared);
                bbox2[bl] = L;
            }
            else
            {
                bbox2[bl] = split;
                knngRecurse(ans, root1, bbox1, root2->right.get(), bbox2, maxDistanceSquared);
                bbox2[bl] = L;
                bbox2[br] = split;
                knngRecurse(ans, root1, bbox1, root2->left.get(), bbox2, maxDistanceSquared);
                bbox2[br] = R;
            }
        }
        else
        {
            int axis = root1->axis;
            int bl = 2 * axis;
            int br = 2 * axis + 1;
            float split = root1->split;
            float mid = 0.5 * (bbox2[bl] + bbox2[br]);
            float L = bbox1[bl];
            float R = bbox1[br];
            if (mid < split)
            {
                bbox1[br] = split;
                knngRecurse(ans, root1->left.get(), bbox1, root2, bbox2, maxDistanceSquared);
                bbox1[br] = R;
                bbox1[bl] = split;
                knngRecurse(ans, root1->right.get(), bbox1, root2, bbox2, maxDistanceSquared);
                bbox1[bl] = L;
            }
            else
            {
                bbox1[bl] = split;
                knngRecurse(ans, root1->right.get(), bbox1, root2, bbox2, maxDistanceSquared);
                bbox1[bl] = L;
                bbox1[br] = split;
                knngRecurse(ans, root1->left.get(), bbox1, root2, bbox2, maxDistanceSquared);
                bbox1[br] = R;
            }

            root1->worstSquared = std::max(root1->left->worstSquared, root1->right->worstSquared);
        }
    }
}

} // anonymous namespace

std::vector<std::vector<std::pair<float, int> > > knng(const Statistics::Container::vector<Splat> &splats, int K, float maxDistanceSquared)
{
    int N = splats.size();
    std::vector<KDPoint> points, pointsTmp;
    std::vector<int> permuteData[6];
    points.reserve(N);
    pointsTmp.reserve(N); // TODO: needs to be populated
    for (int j = 0; j < 3; j++)
    {
        permuteData[j].reserve(N);
        permuteData[j + 3].resize(N); // tmp array
    }
    for (int i = 0; i < N; i++)
    {
        const Splat &s = splats[i];
        points.push_back(KDPoint(s.position[0], s.position[1], s.position[2], i));
        for (int j = 0; j < 3; j++)
            permuteData[j].push_back(i);
    }

    int *permute[3];
    int *permuteTmp[3];
    for (int j = 0; j < 3; j++)
    {
        std::sort(permuteData[j].begin(), permuteData[j].end(), CompareAxis(j, &points[0]));
        permute[j] = &permuteData[j][0];
        permuteTmp[j] = &permuteData[j + 3][0];
    }
    std::vector<int> remap(N);
    boost::scoped_ptr<KDNode> root(buildTree(N, &points[0], &pointsTmp[0],
                                             permute, permuteTmp,
                                             &remap[0], maxDistanceSquared));

    // TODO: clear the temporary arrays

    std::vector<std::vector<std::pair<float, int> > > ans(N);
    for (int i = 0; i < N; i++)
        ans[i].reserve(K);

    float bbox1[6];
    float bbox2[6];
    for (int j = 0; j < 3; j++)
    {
        bbox1[2 * j] = -std::numeric_limits<float>::infinity();
        bbox1[2 * j + 1] = std::numeric_limits<float>::infinity();
    }
    std::copy(bbox1, bbox1 + 6, bbox2);
    knngRecurse(ans, root.get(), bbox1, root.get(), bbox2, maxDistanceSquared);
    return ans;
}
