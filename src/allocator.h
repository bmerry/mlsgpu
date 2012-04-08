/**
 * @file
 *
 * Custom allocator for tracking memory usage
 */

#ifndef ALLOCATOR_H
#define ALLOCATOR_H

#if HAVE_CONFIG_H
# include <config.h>
#endif

#include <new>
#include <vector>
#include <string>
#include <tr1/unordered_map>
#include <tr1/unordered_set>
#include "statistics.h"

namespace Statistics
{

template<typename BaseAllocator>
class Allocator : public BaseAllocator
{
private:
    Statistics::Peak<typename BaseAllocator::size_type> *usage;
    template<typename B> friend class Allocator;

public:
    explicit Allocator(Statistics::Peak<typename BaseAllocator::size_type> *usage = NULL,
                       const BaseAllocator &base = BaseAllocator()) throw()
        : BaseAllocator(base), usage(usage) {}

    /// Copy and conversion constructors
    template<typename B>
    Allocator(const Allocator<B> &b) : BaseAllocator(b), usage(b.usage) {}

    template<typename U> struct rebind
    {
        typedef Allocator<typename BaseAllocator::template rebind<U>::other> other;
    };

    typename BaseAllocator::pointer allocate(typename BaseAllocator::size_type n)
    {
        typename BaseAllocator::pointer ans = BaseAllocator::allocate(n);
        if (usage != NULL)
            *usage += n * sizeof(typename BaseAllocator::value_type);
        return ans;
    }

    template<typename U>
    typename BaseAllocator::pointer allocate(
        typename BaseAllocator::size_type n,
        typename BaseAllocator::template rebind<U>::other::pointer hint)
    {
        typename BaseAllocator::pointer ans = BaseAllocator::allocate(n, hint);
        if (usage != NULL)
            *usage += n * sizeof(typename BaseAllocator::value_type);
        return ans;
    }

    void deallocate(typename BaseAllocator::pointer p, typename BaseAllocator::size_type n)
    {
        BaseAllocator::deallocate(p, n);
        if (usage != NULL)
            *usage -= n * sizeof(typename BaseAllocator::value_type);
    }

    template<typename A, typename B>
    friend bool operator==(const Allocator<A> &a, const Allocator<B> &b);
};

template<typename A, typename B>
bool operator==(const Allocator<A> &a, const Allocator<B> &b)
{
    return a.usage == b.usage && static_cast<const A &>(a) == static_cast<const B &>(b);
}

template<typename A, typename B>
bool operator!=(const Allocator<A> &a, const Allocator<B> &b)
{
    return !(a == b);
}

template<typename Alloc>
Alloc makeAllocator(const std::string &name)
{
    typedef typename Alloc::size_type size_type;
    return Alloc(&Statistics::getStatistic<Statistics::Peak<size_type> >(name));
}

namespace Container
{

template<
    typename T,
    typename Alloc = Allocator<std::allocator<T> > >
class vector : public std::vector<T, Alloc>
{
private:
    typedef std::vector<T, Alloc> BaseType;
public:
    explicit vector(const std::string &allocName)
        : BaseType(makeAllocator<Alloc>(allocName)) {}

    explicit vector(const std::string &allocName, typename BaseType::size_type n, const T &value = T())
        : BaseType(n, value, makeAllocator<Alloc>(allocName)) {}

    template<typename InputIterator>
    vector(const std::string &allocName, InputIterator first, InputIterator last)
        : BaseType(first, last, makeAllocator<Alloc>(allocName)) {}
};

template<
    typename Key,
    typename T,
    typename Hash = std::tr1::hash<Key>,
    typename Pred = std::equal_to<Key>,
    typename Alloc = Allocator<std::allocator<std::pair<const Key, T> > > >
class unordered_map : public std::tr1::unordered_map<Key, T, Hash, Pred, Alloc>
{
private:
    typedef std::tr1::unordered_map<Key, T, Hash, Pred, Alloc> BaseType;
public:
    explicit unordered_map(
        const std::string &allocName,
        typename BaseType::size_type n = 10,   // implementation-defined in base class, so perfect forwarding impossible
        const typename BaseType::hasher &hf = typename BaseType::hasher(),
        const typename BaseType::key_equal &eql = typename BaseType::key_equal())
        : BaseType(n, hf, eql, makeAllocator<Alloc>(allocName)) {}

    template<typename InputIterator>
    unordered_map(
        const std::string &allocName,
        InputIterator f, InputIterator l,
        typename BaseType::size_type n = 10,   // implementation-defined in base class, so perfect forwarding impossible
        const typename BaseType::hasher &hf = typename BaseType::hasher(),
        const typename BaseType::key_equal &eql = typename BaseType::key_equal())
        : BaseType(f, l, n, hf, eql, makeAllocator<Alloc>(allocName)) {}
};

} // namespace Container
} // namespace Statistics

#endif /* ALLOCATOR_H */
