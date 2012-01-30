/**
 * @file
 *
 * Concepts and implementations for disk-backed collections of items.
 */

#ifndef COLLECTION_H
#define COLLECTION_H

#if HAVE_CONFIG_H
# include <config.h>
#endif

#include <stxxl.h>
#include <algorithm>
#include <stdexcept>
#include <vector>
#include "fast_ply.h"
#include "errors.h"

class Splat;

#if DOXYGEN_FAKE_CODE
/**
 * A random-access collection of objects. This is similar to
 * the Random Access Container STL concept, but it only
 * supports access to ranges of items rather an individual ones,
 * and only supports read-only access. It is not required to
 * be copyable.
 *
 * The restrictions allow it to be efficiently used when
 * backed by an on-disk file or other external container.
 *
 * @note This class does not exist. It is a concept only.
 */
class Collection
{
public:
    /// Type stored in the collection.
    typedef int value_type;

    /// Type used to index elements to fetch.
    typedef std::size_t size_type;

    /**
     * Default constructor. Models are expected to provide
     * model-specific constructors to populate the collection.
     */
    Collection();

    /// Number of items in the container
    size_type size() const;

    /**
     * Copy out a contiguous range from a scan.
     *
     * @param first,last  Half-open interval to copy.
     * @param out         Output iterator which receives the values.
     * @return The new value of the output iterator.
     * @pre
     * - @a first &lt;= @a last &lt;= @ref size()
     */
    template<typename OutputIterator>
    OutputIterator read(size_type first, size_type last, OutputIterator out) const;

    /**
     * Call a function object for all splats in a range.
     *
     * The function object signature should be
     * <code>
     * void(size_type index, const value_type &item)
     * </code>
     *
     * @param first, last   Half-open range of indices to process
     * @param f             Function object to call.
     *
     * @pre
     * - @a first &lt;= @a last &lt;= @ref size()
     */
    template<typename Func>
    void forEach(size_type first, size_type last, const Func &f) const;
};
#endif // DOXYGEN_FAKE_CODE

/**
 * Wraps a reference to a type with a vector-like interface into the
 * @ref Collection concept.
 *
 * @param VectorType  The underlying vector type.
 */
template<typename VectorType>
class VectorCollection
#if DOXYGEN_FAKE_CODE
: public Collection
#endif
{
public:
    typedef VectorType vector_type;
    typedef typename VectorType::value_type value_type;
    typedef typename VectorType::size_type size_type;

    /**
     * Wraps a reference to an existing vector.
     *
     * @warning This object holds a reference to @a items, so it must not
     * be destroyed until this object is destroyed.
     */
    VectorCollection(const vector_type &items) : items(items) {}

    size_type size() const { return items.size(); }

    template<typename OutputIterator>
    OutputIterator read(size_type first, size_type last, OutputIterator out) const;

    template<typename Func>
    void forEach(size_type first, size_type last, const Func &f) const;

private:
    /// Underlying wrapped vector.
    const vector_type &items;
};

/**
 * Convenience wrapper around @ref VectorCollection that takes the element type
 * and computes a @c std::vector instantiation. It does not change the interface
 * at all, merely simplifies the name of the type.
 */
template<typename ValueType>
class StdVectorCollection : public VectorCollection<std::vector<ValueType> >
{
private:
    /// Superclass
    typedef VectorCollection<typename std::vector<ValueType> > base_type;
public:
    typedef typename base_type::vector_type vector_type;
    typedef typename base_type::size_type size_type;
    typedef typename base_type::value_type value_type;

    using base_type::size;
    using base_type::read;
    using base_type::forEach;

    /**
     * Constructor.
     * @see VectorCollection::VectorCollection.
     */
    StdVectorCollection(const vector_type &items) : base_type(items) {}
};

/**
 * Convenience wrapper around @ref VectorCollection that takes the element type
 * and computes an @c stxxl::vector instantiation.  Note that the chosen
 * instantiation is @em not the STXXL default. It is selected for the specific use
 * cases for which this class is being used.
 */
template<typename ValueType>
class StxxlVectorCollection : public VectorCollection<typename stxxl::VECTOR_GENERATOR<ValueType, 4, 27, 32768 * sizeof(ValueType)>::result>
{
private:
    /// Superclass
    typedef VectorCollection<typename stxxl::VECTOR_GENERATOR<ValueType, 4, 27, 32768 * sizeof(ValueType)>::result> base_type;
public:
    typedef typename base_type::vector_type vector_type;
    typedef typename base_type::size_type size_type;
    typedef typename base_type::value_type value_type;

    using base_type::size;
    using base_type::read;
    using base_type::forEach;

    StxxlVectorCollection(const vector_type &items) : base_type(items) {}
};

/**
 * Implementation of the STXXL stream concept that reads all items
 * from a collection of Collections.
 *
 * @param Iterator a forward iterator whose value type is a model of
 * @ref Collection.
 */
template<typename Iterator>
class CollectionStream
{
private:
    typedef typename Iterator::value_type Collection;

    Iterator first, last;
    Iterator current;
    typename Collection::size_type position;
public:
    typedef typename Collection::value_type value_type;

    /**
     * Constructs an empty stream.
     */
    CollectionStream();

    /**
     * Constructs a stream from an existing range of collections.
     */
    CollectionStream(Iterator first, Iterator last);

    /// Obtain the next item in the stream (concept requirement).
    value_type operator*() const;
    /// Advance the stream (concept requirement).
    CollectionStream<Iterator> &operator++();
    /// Determine whether there is a next item (concept requirement).
    bool empty() const;
};



template<typename VectorType>
template<typename OutputIterator>
OutputIterator VectorCollection<VectorType>::read(
    size_type first, size_type last, OutputIterator out) const
{
    MLSGPU_ASSERT(first <= last && last <= size(), std::out_of_range);
    return std::copy(items.begin() + first, items.begin() + last, out);
}

template<typename VectorType>
template<typename Func>
void VectorCollection<VectorType>::forEach(size_type first, size_type last, const Func &f) const
{
    // TODO: see if there is some way to leverage the prefetching in
    // stxxl::for_each. We can't use it directly because it doesn't
    // pass the iterator.
    MLSGPU_ASSERT(first <= last && last <= size(), std::out_of_range);
    for (size_type i = first; i < last; i++)
    {
        // In stxxl it is dangerous to hold a reference
        value_type item = items[i];
        f(i, item);
    }
}


template<typename Iterator>
CollectionStream<Iterator>::CollectionStream() : first(), last(), current(last), position(0) {}

template<typename Iterator>
CollectionStream<Iterator>::CollectionStream(Iterator first, Iterator last)
    : first(first), last(last), current(first), position(0)
{
    while (current != last && current->size() == 0)
        ++current;
}

template<typename Iterator>
typename CollectionStream<Iterator>::value_type CollectionStream<Iterator>::operator*() const
{
    assert(!empty());
    value_type ans;
    current->read(position, position + 1, &ans);
    return ans;
}

template<typename Iterator>
CollectionStream<Iterator> &CollectionStream<Iterator>::operator++()
{
    assert(!empty());
    ++position;
    while (current != last && position == current->size())
    {
        position = 0;
        ++current;
    }
    return *this;
}

template<typename Iterator>
bool CollectionStream<Iterator>::empty() const
{
    return current == last;
}

#endif /* !COLLECTION_H */
