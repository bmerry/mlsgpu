/*
 * mlsgpu: surface reconstruction from point clouds
 * Copyright (C) 2013  University of Cape Town
 *
 * This file is part of mlsgpu.
 *
 * mlsgpu is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

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
#include <list>
#include <boost/multi_array.hpp>
#include "tr1_unordered_map.h"
#include "tr1_unordered_set.h"
#include "pod_buffer.h"
#include "statistics.h"

class TestAllocator;

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
    friend class ::TestAllocator;
    template<typename B> friend class Allocator;
private:
    /**
     * The statistic in which we store total allocated memory. It may be @c NULL
     * (and will be if this allocator is default-constructed).
     */
    Statistics::Peak *usage;

    /**
     * The statistic for storing the global total memory allocation. It may be
     * @c NULL (and will be if this allocator is default-constructed).
     */
    Statistics::Peak *allUsage;

public:
    /// Underlying allocator type
    typedef BaseAllocator base_type;

    /**
     * Constructor.
     *
     * @param usage   The statistic that will track usage from this allocator.
     *                It may already have a non-zero current value. If it is @c NULL, no
     *                tracking is done.
     * @param allUsage The statistic that will track total memory usage. It is
     *                functionally equivalent to @a usage.
     * @param base    The underlying allocator providing the real functionality.
     */
    explicit Allocator(Statistics::Peak *usage = NULL,
                       Statistics::Peak *allUsage = NULL,
                       const BaseAllocator &base = BaseAllocator()) throw()
        : BaseAllocator(base), usage(usage), allUsage(allUsage) {}

    /// Copy and conversion constructors
    template<typename B>
    Allocator(const Allocator<B> &b) :
        BaseAllocator(static_cast<const B &>(b)),
        usage(b.usage),
        allUsage(b.allUsage) {}

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
        if (allUsage != NULL)
            *allUsage += n * sizeof(typename BaseAllocator::value_type);
        return ans;
    }

    /// Allocate raw space for @a n items of the value type, with a location hint
    typename BaseAllocator::pointer allocate(
        typename BaseAllocator::size_type n,
        std::allocator<void>::const_pointer hint)
    {
        // See comments in the other overload
        typename BaseAllocator::pointer ans = BaseAllocator::allocate(n, hint);
        if (usage != NULL)
            *usage += n * sizeof(typename BaseAllocator::value_type);
        if (allUsage != NULL)
            *allUsage += n * sizeof(typename BaseAllocator::value_type);
        return ans;
    }

    /// Release previously allocated memory
    void deallocate(typename BaseAllocator::pointer p, typename BaseAllocator::size_type n)
    {
        BaseAllocator::deallocate(p, n);
        if (usage != NULL)
            *usage -= n * sizeof(typename BaseAllocator::value_type);
        if (allUsage != NULL)
            *allUsage -= n * sizeof(typename BaseAllocator::value_type);
    }

    /// Manually record an allocation that bypassed this allocator
    void recordAllocate(typename BaseAllocator::size_type bytes)
    {
        if (usage != NULL)
            *usage += bytes;
        if (allUsage != NULL)
            *allUsage += bytes;
    }

    /// Manually record a deallocation that matches @ref recordAllocate
    void recordDeallocate(typename BaseAllocator::size_type bytes)
    {
        if (usage != NULL)
            *usage -= bytes;
        if (allUsage != NULL)
            *allUsage -= bytes;
    }

    template<typename A, typename B>
    friend bool operator==(const Allocator<A> &a, const Allocator<B> &b);
};

/// Returns true if storage allocated from one can be released by the other.
template<typename A, typename B>
bool operator==(const Allocator<A> &a, const Allocator<B> &b)
{
    return a.usage == b.usage
        && a.allUsage == b.allUsage
        && static_cast<const A &>(a) == static_cast<const B &>(b);
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
    Statistics::Registry &registry = Statistics::Registry::getInstance();
    Statistics::Peak &allStat = registry.getStatistic<Statistics::Peak>("mem.all");
    Statistics::Peak &myStat = registry.getStatistic<Statistics::Peak>(name);

    return Alloc(&myStat, &allStat);
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

/**
 * Wrapper around @c std::list that uses @ref Statistics::Allocator.
 * @see @ref Statistics::Container
 */
template<
    typename T,
    typename Alloc = Allocator<std::allocator<T> > >
class list : public std::list<T, Alloc>
{
private:
    typedef std::list<T, Alloc> BaseType;
public:
    explicit list(const std::string &allocName)
        : BaseType(makeAllocator<Alloc>(allocName)) {}

    template<typename InputIterator>
    list(const std::string &allocName, InputIterator first, InputIterator last)
        : BaseType(first, last, makeAllocator<Alloc>(allocName)) {}
};

template<typename T, typename Alloc = Allocator<std::allocator<T> > >
class PODBuffer : public ::PODBuffer<T, Alloc>
{
private:
    typedef ::PODBuffer<T, Alloc> BaseType;
public:
    explicit PODBuffer(const std::string &allocName)
        : BaseType(makeAllocator<Alloc>(allocName)) {}

    explicit PODBuffer(const std::string &allocName, typename BaseType::size_type capacity)
        : BaseType(capacity, makeAllocator<Alloc>(allocName)) {}
};

/**
 * Wrapper around @c std::tr1::unordered_set that uses @ref Statistics::Allocator.
 * @see @ref Statistics::Container
 */
template<
    typename Value,
    typename Hash = std::tr1::hash<Value>,
    typename Pred = std::equal_to<Value>,
    typename Alloc = Allocator<std::allocator<Value> > >
class unordered_set : public std::tr1::unordered_set<Value, Hash, Pred, Alloc>
{
private:
    typedef std::tr1::unordered_set<Value, Hash, Pred, Alloc> BaseType;
    typedef typename BaseType::size_type size_type;
    typedef typename BaseType::hasher hasher;
    typedef typename BaseType::key_equal key_equal;
public:
    explicit unordered_set(
        const std::string &allocName,
        size_type n = 10,   // implementation-defined in base class, so perfect forwarding impossible
        const hasher &hf = hasher(),
        const key_equal &eql = key_equal())
        : BaseType(n, hf, eql, makeAllocator<Alloc>(allocName)) {}

    template<typename InputIterator>
    unordered_set(
        const std::string &allocName,
        InputIterator f, InputIterator l,
        size_type n = 10,   // implementation-defined in base class, so perfect forwarding impossible
        const hasher &hf = hasher(),
        const key_equal &eql = key_equal())
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
    typename Alloc = Allocator<std::allocator<std::pair<const Key, T> > > >
class unordered_map : public std::tr1::unordered_map<Key, T, Hash, Pred, Alloc>
{
private:
    typedef std::tr1::unordered_map<Key, T, Hash, Pred, Alloc> BaseType;
    typedef typename BaseType::size_type size_type;
    typedef typename BaseType::hasher hasher;
    typedef typename BaseType::key_equal key_equal;
public:
    explicit unordered_map(
        const std::string &allocName,
        size_type n = 10,   // implementation-defined in base class, so perfect forwarding impossible
        const hasher &hf = hasher(),
        const key_equal &eql = key_equal())
        : BaseType(n, hf, eql, makeAllocator<Alloc>(allocName)) {}

    template<typename InputIterator>
    unordered_map(
        const std::string &allocName,
        InputIterator f, InputIterator l,
        size_type n = 10,   // implementation-defined in base class, so perfect forwarding impossible
        const hasher &hf = hasher(),
        const key_equal &eql = key_equal())
        : BaseType(f, l, n, hf, eql, makeAllocator<Alloc>(allocName)) {}
};

template<
    typename ValueType,
    std::size_t NumDims,
    typename Alloc = Allocator<std::allocator<ValueType> > >
class multi_array : public boost::multi_array<ValueType, NumDims, Alloc>
{
private:
    typedef boost::multi_array<ValueType, NumDims, Alloc> BaseType;

public:
    explicit multi_array(
        const std::string &allocName)
        : BaseType(std::vector<std::size_t>(NumDims, 0), boost::c_storage_order(),
                   makeAllocator<Alloc>(allocName)) {}

    template<typename ExtentList>
    multi_array(const std::string &allocName,
                const ExtentList &sizes,
                const typename BaseType::storage_order_type &store = boost::c_storage_order())
        : BaseType(sizes, store, makeAllocator<Alloc>(allocName)) {}
};

} // namespace Container
} // namespace Statistics

#endif /* ALLOCATOR_H */
