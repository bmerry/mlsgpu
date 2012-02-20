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

namespace UnionFind
{

template<typename Size>
class Node
{
public:
    typedef Size size_type;

    Node() : parentSize(-1) {}
    Node(size_type size) { setSize(size); }

    bool isRoot() const
    {
        return parentSize < 0;
    }

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

    mutable size_type parentSize; ///< size if negative, parent if non-negative

    size_type parent() const
    {
        assert(parentSize >= 0);
        return parentSize;
    }

    void setParent(size_type p) const
    {
        assert(p >= 0);
        parentSize = p;
    }

    void setSize(size_type s)
    {
        BOOST_STATIC_ASSERT(boost::is_signed<size_type>::value);
        assert(s > 0);
        parentSize = -s;
    }

protected:
    void merge(const Node &b)
    {
        assert(isRoot() && b.isRoot());
        parentSize += b.parentSize;
    }
};

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
