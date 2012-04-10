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

/**
 * An STL-compatible allocator that wraps another allocator to measure peak
 * statistics. All the allocator functionality is passed to the wrapped version.
 *
 * Because the C++ standard allows the standard containers to make extra
 * assumptions about allocators (i.e. that they're all interchangeable) this is
 * not guaranteed to work on all implementations.
 */
template<typename BaseAllocator>
class Allocator : public BaseAllocator
{
private:
    /**
     * The statistic in which we store total allocated memory. It may be @c NULL
     * (and will be if this allocator is default-constructed).
     */
    Statistics::Peak<typename BaseAllocator::size_type> *usage;

    template<typename B> friend class Allocator;

public:
    /// Underlying allocator type
    typedef BaseAllocator base_type;

    /**
     * Constructor.
     *
     * @param usage   The statistic that will track usage from this allocator.
     *                It may already have a non-zero current value. If it is @c NULL, no
     *                tracking is done.
     * @param base    The underlying allocator providing the real functionality.
     */
    explicit Allocator(Statistics::Peak<typename BaseAllocator::size_type> *usage = NULL,
                       const BaseAllocator &base = BaseAllocator()) throw()
        : BaseAllocator(base), usage(usage) {}

    /// Copy and conversion constructors
    template<typename B>
    Allocator(const Allocator<B> &b) : BaseAllocator(b), usage(b.usage) {}

    /// Interface requirement
    template<typename U> struct rebind
    {
        typedef Allocator<typename BaseAllocator::template rebind<U>::other> other;
    };

    /// Allocate raw space for @a n items of the value type
    typename BaseAllocator::pointer allocate(typename BaseAllocator::size_type n)
    {
        // Note: we do the allocation before the tracking, in case it throws.
        // The tracking cannot throw.
        typename BaseAllocator::pointer ans = BaseAllocator::allocate(n);
        if (usage != NULL)
            *usage += n * sizeof(typename BaseAllocator::value_type);
        return ans;
    }

    /// Allocate raw space for @a n items of the value type, with a location hint
    template<typename U>
    typename BaseAllocator::pointer allocate(
        typename BaseAllocator::size_type n,
        typename BaseAllocator::template rebind<U>::other::pointer hint)
    {
        // See comments in the other overload
        typename BaseAllocator::pointer ans = BaseAllocator::allocate(n, hint);
        if (usage != NULL)
            *usage += n * sizeof(typename BaseAllocator::value_type);
        return ans;
    }

    /// Release previously allocated memory
    void deallocate(typename BaseAllocator::pointer p, typename BaseAllocator::size_type n)
    {
        BaseAllocator::deallocate(p, n);
        if (usage != NULL)
            *usage -= n * sizeof(typename BaseAllocator::value_type);
    }

    template<typename A, typename B>
    friend bool operator==(const Allocator<A> &a, const Allocator<B> &b);
};

/// Returns true if storage allocated from one can be released by the other.
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

/**
 * Takes a statistic name and generates an allocator that uses a statistic with
 * that name from the default registry, as well as a statistic called @c mem.all
 * from the default registry.
 */
template<typename Alloc>
Alloc makeAllocator(const std::string &name)
{
    typedef typename Alloc::size_type size_type;
    static Statistics::Peak<size_type> &allStat = Statistics::getStatistic<Statistics::Peak<size_type> >("mem.all");
    static typename Alloc::base_type base(&allStat);

    Statistics::Peak<size_type> &myStat = Statistics::getStatistic<Statistics::Peak<size_type> >(name);
    return Alloc(&myStat, base);
}

/**
 * Wrappers around standard container types which use @ref Statistics::Allocator instead
 * of @c std::allocator. Each wrapper provides forwarding constructors that take an extra
 * initial argument, @a allocName. The corresponding statistic is generated or found in
 * the default registry.
 */
namespace Container
{

/**
 * Wrapper around @c std::vector that uses @ref Statistics::Allocator.
 * @see @ref Statistics::Container
 */
template<
    typename T,
    typename Alloc = Allocator<Allocator<std::allocator<T> > > >
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

/**
 * Wrapper around @c std::tr1::unordered_set that uses @ref Statistics::Allocator.
 * @see @ref Statistics::Container
 */
template<
    typename Value,
    typename Hash = std::tr1::hash<Value>,
    typename Pred = std::equal_to<Value>,
    typename Alloc = Allocator<Allocator<std::allocator<Value> > > >
class unordered_set : public std::tr1::unordered_set<Value, Hash, Pred, Alloc>
{
private:
    typedef std::tr1::unordered_set<Value, Hash, Pred, Alloc> BaseType;
public:
    explicit unordered_set(
        const std::string &allocName,
        typename BaseType::size_type n = 10,   // implementation-defined in base class, so perfect forwarding impossible
        const typename BaseType::hasher &hf = typename BaseType::hasher(),
        const typename BaseType::key_equal &eql = typename BaseType::key_equal())
        : BaseType(n, hf, eql, makeAllocator<Alloc>(allocName)) {}

    template<typename InputIterator>
    unordered_set(
        const std::string &allocName,
        InputIterator f, InputIterator l,
        typename BaseType::size_type n = 10,   // implementation-defined in base class, so perfect forwarding impossible
        const typename BaseType::hasher &hf = typename BaseType::hasher(),
        const typename BaseType::key_equal &eql = typename BaseType::key_equal())
        : BaseType(f, l, n, hf, eql, makeAllocator<Alloc>(allocName)) {}
};

/**
 * Wrapper around @c std::tr1::unordered_map that uses @ref Statistics::Allocator.
 * @see @ref Statistics::Container
 */
template<
    typename Key,
    typename T,
    typename Hash = std::tr1::hash<Key>,
    typename Pred = std::equal_to<Key>,
    typename Alloc = Allocator<Allocator<std::allocator<std::pair<const Key, T> > > > >
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
