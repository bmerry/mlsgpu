/**
 * @file
 *
 * CPU code to find k nearest neighboring points for every point in a point
 * set.
 */

#ifndef KNNG_H
#define KNNG_H

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <memory>
#include <vector>
#include <utility>
#include <cstddef>
#include "../src/errors.h"
#include "../src/tr1_cstdint.h"

template<typename Coord, int Dim = 3, typename SizeType = std::tr1::uint32_t>
class KDTree;

template<typename Coord, typename SizeType>
struct KNNGData;

class TestKDTree;

/**
 * Representation of a k-nearest neighbor graph. For each point, one can
 * query a list of neighbors, expressed as a pair of the squared distance
 * to the neighbor and the index of the neighbor. The neighbors are
 * @em not necessarily ordered by increasing distance.
 */
template<typename Coord, typename SizeType>
class KNNG
{
    template<typename Coord2, int Dim2, typename SizeType2> friend class KDTree;
public:
    typedef Coord coord_type;
    typedef SizeType size_type;
    typedef std::vector<std::pair<Coord, SizeType> > value_type;

    /// Return the number of neighbors requested.
    size_type getK() const { return K; }

    /// Return the number of points in the graph.
    size_type size() const { return N; }

    /**
     * Return the neighbors of a point.
     * @param idx    Original point index
     * @pre idx < @ref size()
     */
    value_type operator[](SizeType idx) const
    {
        MLSGPU_ASSERT(idx < N, std::out_of_range);
        idx = reorder[idx];
        std::size_t first = std::size_t(K) * idx;
        return value_type(neighbors.begin() + first,
                          neighbors.begin() + (first + numNeighbors[idx]));
    }

private:
    size_type N;  ///< Number of points
    size_type K;  ///< Maximum neighbors

    std::vector<size_type> reorder; ///< Inverse of the permutation encoded in the points

    /**
     * Storage for the neighbors. The ith point has neighbors starting at
     * <code>reorder[i] * K</code>, consecutively for <code>numNeighbors[reorder[i]]</code>
     * elements.
     */
    std::vector<std::pair<coord_type, size_type> > neighbors;
    /// Number of neighbors of each point
    std::vector<size_type> numNeighbors;

    KNNG() {} // prevents default construction
    KNNG(size_type N, size_type K); // used by the friend class
};

/**
 * A static in-memory KD tree. The @a SizeType has a significant effect on total memory
 * requirements, so should not be too large.
 *
 * @param Coord     A scalar type of the coordinates (must be floating point).
 * @param Dim       The number of dimensions.
 * @param SizeType  A type used to store point and node indices.
 */
template<typename Coord, int Dim, typename SizeType>
class KDTree
{
    friend class TestKDTree;
public:
    typedef SizeType size_type;  ///< Type used to store point and node indices
    typedef Coord coord_type;    ///< Scalar type for the coordinates
    enum
    {
        DIM = Dim                ///< Number of dimensions
    };

    /// Return the number of points in the KD Tree
    size_type size() const { return points.size(); }

    /**
     * Constructor. The iterator must model random access iterator
     * (or random access traversal iterator), and the value type must
     * be <code>Eigen::Matrix<coord_type, DIM, 1></code>.
     * 
     * If the underlying data is of a different type, iterator adaptors
     * should be used.
     *
     * @param first, last  A range of points
     */
    template<typename Iterator>
    KDTree(Iterator first, Iterator last);

    /**
     * Compute the k-nearest neighbor graph for all points.
     *
     * @param K     Maximum number of neighbors to find.
     * @param maxDistanceSquared Upper bound on distance to search neighbors
     * @return A newly allocated @ref KNNG structure, which should be
     * destroyed with @c delete.
     */
    KNNG<coord_type, size_type> *knn(size_type K, coord_type maxDistanceSquared) const;

private:
    enum
    {
        MIN_LEAF = 4             ///< Target minimum leaf bucket size
    };

    /// Internal type to hold point position data
    typedef Eigen::Matrix<Coord, DIM, 1> Point;

    /// An entry for a single point in a leaf
    struct KDPoint
    {
        Point pos;          ///< Spatial position
        size_type id;       ///< Index in the user-provided list

        KDPoint() {}
        KDPoint(const Point &pos, size_type id) : pos(pos), id(id) {}
    };

    /**
     * Node in the tree (either internal or leaf).
     */
    struct KDNode
    {
        int axis;                       ///< Split axis, or -1 for a leaf
        union
        {
            struct
            {
                coord_type split;       ///< Split plane value
                size_type left, right;  ///< Indices of left and right children
            } internal;                 ///< State for non-leaf nodes
            struct
            {
                size_type first, last;  ///< Indices of first and past-the-end points
            } leaf;                     ///< State for leaf nodes
        } u;
        coord_type bbox[DIM][2];        ///< Bounding box

        /// Default constructor
        KDNode() : axis(-1) {}
        /// Whether this point is a leaf
        bool isLeaf() const { return axis < 0; }
        /**
         * Left child index.
         * @pre @ref isLeaf() is false
         */
        size_type left() const { return u.internal.left; }
        /**
         * Right child index.
         * @pre @ref isLeaf() is false
         */
        size_type right() const { return u.internal.right; }
        /**
         * Index of first point in bucket.
         * @pre @ref isLeaf()
         */
        size_type first() const { return u.leaf.first; }
        /**
         * Index one past the last point in bucket.
         * @pre @ref isLeaf()
         */
        size_type last() const { return u.leaf.last; }
        /**
         * Split coordinate.
         * @pre @ref isLeaf() is false
         */
        coord_type split() const { return u.internal.split; }
    };

    /// Sorts point indices on one axis
    class CompareAxis
    {
    private:
        int axis;
        const KDPoint *points;

    public:
        /**
         * Constructor.
         * @param axis    Axis to sort on (0 to @ref DIM - 1)
         * @param points  Backing array of points.
         */
        CompareAxis(int axis, const KDPoint *points) : axis(axis), points(points) {}

        bool operator()(size_type a, size_type b) const
        {
            return points[a].pos[axis] < points[b].pos[axis];
        }
    };

    /**
     * Flat list of points, ordered such that each node contains a contiguous sequence.
     */
    std::vector<KDPoint> points;
    /**
     * Flat list of nodes. The node indices all reference this list. Node 0 is
     * the root. It is also possible for this list to be empty if the tree is
     * empty.
     */
    std::vector<KDNode> nodes;

    /**
     * Recursively build a node in the tree, appending it and its descendants
     * to @ref nodes. The node itself will be the first one to be appended.
     *
     * @param N          Number of points to consider.
     * @param points     Array of @a N points to add.
     * @param pointsTmp  Scratch space for at least @a N points
     * @param permute    Each element is a pointer to an array of @a N indices into @a points,
     *                   giving the order of the points sorted along the corresponding axis.
     * @param permuteTmp Scratch space with the same storage space as @a permute.
     * @param remap      Scratch space for @a N indices.
     */
    void buildTree(size_type N, KDPoint points[], KDPoint pointsTmp[],
                   size_type *permute[DIM], size_type *permuteTmp[DIM],
                   size_type *remap);

    /**
     * Update @a p's list of nearest neighbors with @a q.
     * @param p, q            Indices into @ref points (not equal).
     * @param distSquared     Squared distance from point @a p to point @a q.
     * @param out             Nearest neighbor structure to update.
     */
    void updatePairOneWay(size_type p, size_type q, coord_type distSquared, KNNG<coord_type, size_type> &out) const;

    /**
     * Update pairwise distances between @a p and @a q.
     */
    void updatePair(size_type p, size_type q, KNNGData<coord_type, size_type> &out) const;

    /**
     * Update the maximum squared distance bound for leaf @a nodeIdx based on its points.
     * @param nodeIdx         A node index to update.
     * @param data            Data structure holding value to update.
     * @pre @a nodeIdx indexes a leaf node.
     */
    void updateWorstSquared(size_type nodeIdx, KNNGData<coord_type, size_type> &data) const;

    /**
     * Recursively consider all pairs of nodes in two subtrees and update neighborhoods.
     *
     * The strategy is recursive on both subtrees i.e. it will split one or the other
     * (or both) before recursing again.
     *
     * @param root1, root2    Indices of two nodes (can be equal).
     * @param data            Data structure to update with neighborhoods.
     * @pre  The worst-case distances for @a root1 and @a root2 are up to date.
     * @post The worst-case distances for @a root1 and @a root2 are again up to date.
     */
    void knngRecurse(size_type root1, size_type root2, KNNGData<coord_type, size_type> &data) const;
};

#include "knng_impl.h"

#endif /* !KNNG_H */
