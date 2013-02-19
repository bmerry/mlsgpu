/**
 * @file
 *
 * Union-find data structure for efficient identification of components.
 */

#ifndef UNION_FIND_H
#define UNION_FIND_H

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <cassert>
#include <algorithm>
#include <boost/static_assert.hpp>
#include <boost/type_traits/is_signed.hpp>
#include <boost/serialization/serialization.hpp>

#if UNIT_TESTS
class TestUnionFind;
#endif

/**
 * Union-find data structure for efficient identification of components. It uses a tree
 * with path compression and balancing by size, giving it O(alpha(N)) amortized time per
 * operation.
 *
 * Each component has a canonical member, or @em root. Per-vertex storage is provided by
 * the caller in a vector (or other random-access container).
 */
namespace UnionFind
{

/**
 * A per-vertex element in the union-find data structure. If additional
 * per-component information is desired, this class can be subclassed and @ref
 * merge overloaded.
 *
 * @param Size A @b signed type with enough range to represent the number of
 * elements in the graph.
 */
template<typename Size>
class Node
{
    friend class boost::serialization::access;
public:
    typedef Size size_type;     ///< Type used to store either parent index or node size

    /**
     * Default constructor. Creates a root node of size 1.
     */
    Node() : parentSize(-1) {}

    /// Determines whether this node is the canonical root node for its component.
    bool isRoot() const
    {
        return parentSize < 0;
    }

    /**
     * The size of this component.
     *
     * @pre @ref isRoot() is true.
     */
    size_type size() const
    {
        assert(parentSize < 0);
        return -parentSize;
    }

private:
    template<typename NodeVector>
        friend typename NodeVector::iterator::value_type::size_type
        findRoot(const NodeVector &nodes, typename NodeVector::iterator::value_type::size_type id);
    template<typename NodeVector>
        friend bool merge(NodeVector &nodes,
                          typename NodeVector::iterator::value_type::size_type a,
                          typename NodeVector::iterator::value_type::size_type b);
#if UNIT_TESTS
    friend class ::TestUnionFind;
#endif

    /**
     * Size if negative, parent if non-negative. This is mutable because path
     * compression will alter it without making semantic changes to the
     * structure.
     */
    mutable size_type parentSize;

    /**
     * Get the parent index in the tree.
     *
     * @pre @ref isRoot() is false.
     */
    size_type parent() const
    {
        assert(parentSize >= 0);
        return parentSize;
    }

    /// Sets the parent pointer in the tree, marking this node as non-root.
    void setParent(size_type p) const
    {
        assert(p >= 0);
        parentSize = p;
    }

    /// Sets the size of this component, marking this node as a root.
    void setSize(size_type s)
    {
        BOOST_STATIC_ASSERT(boost::is_signed<size_type>::value);
        assert(s > 0);
        parentSize = -s;
    }

    template<typename Archive>
    void serialize(Archive &ar, const unsigned int)
    {
        ar & parentSize;
    }

protected:
    /**
     * Update metadata about a component when another component is merged in.
     * This method can be overloaded in subclasses to track additional
     * component data.
     *
     * @param b     Root of a component that will become a child of this node.
     */
    void merge(const Node &b)
    {
        assert(isRoot() && b.isRoot());
        parentSize += b.parentSize;
    }
};

/**
 * Determines the root node of the component of a given node.
 *
 * @param nodes        Random access container of nodes giving a union-find structure.
 * @param id           Index of the query node.
 * @return Index of the unique root node that is in the same component as @a id.
 * @note Although this function is semantically read-only, it performs path
 * compression and so does modify the internals of @a nodes.
 */
template<typename NodeVector>
typename NodeVector::iterator::value_type::size_type
findRoot(const NodeVector &nodes, typename NodeVector::iterator::value_type::size_type id)
{
    typename NodeVector::iterator::value_type::size_type root, next;
    root = id;
    while (!nodes[root].isRoot())
        root = nodes[root].parent();
    while (id != root)
    {
        next = nodes[id].parent();
        nodes[id].setParent(root);
        id = next;
    }
    return root;
}

/**
 * Combine two components. It is legal to call this function when the given
 * nodes are already in the same component.
 *
 * @param nodes        Random access container of nodes giving a union-find structure.
 * @param a, b         Two nodes (not necessarily roots) to combine.
 * @retval @c true if two separate components were merged.
 * @retval @c false if @a a and @a b were already in the same component.
 */
template<typename NodeVector>
bool merge(NodeVector &nodes,
           typename NodeVector::iterator::value_type::size_type a,
           typename NodeVector::iterator::value_type::size_type b)
{
    a = findRoot(nodes, a);
    b = findRoot(nodes, b);
    bool merged = (a != b);
    if (merged)
    {
        if (nodes[a].size() > nodes[b].size())
            std::swap(a, b);
        nodes[b].merge(nodes[a]);
        nodes[a].setParent(b);
    }
    return merged;
}

} // namespace UnionFind

#endif /* !UNION_FIND_H */
