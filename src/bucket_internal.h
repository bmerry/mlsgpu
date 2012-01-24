/**
 * @file
 *
 * Classes and functions in @ref bucket.cpp that are declared in a header so that
 * they can be tested. The declarations in this header should not be used
 * other than in @ref bucket.cpp and in test code.
 */

#ifndef BUCKET_INTERNAL_H
#define BUCKET_INTERNAL_H

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <tr1/cstdint>
#include "bucket.h"

namespace Bucket
{
namespace internal
{

/**
 * Tracks how many ranges are needed to encode a list of splats and
 * how many splats are in the list. It will match the actual number
 * written by @ref RangeCollector.
 */
class RangeCounter
{
private:
    std::tr1::uint64_t ranges;
    std::tr1::uint64_t splats;
    Range current;

public:
    /// Constructor
    RangeCounter();

    /// Adds a new splat to the virtual list.
    void append(Range::scan_type scan, Range::index_type splat);

    /// Returns the number of ranges that would be required to encode the provided splats.
    std::tr1::uint64_t countRanges() const;

    /// Returns the number of splats given to @ref append
    std::tr1::uint64_t countSplats() const;
};

/**
 * Accepts a list of splat IDs and merges them into ranges which are then
 * output.
 * @param OutputIterator an output iterator that accepts assignments of @ref Range.
 */
template<typename OutputIterator>
class RangeCollector
{
public:
    typedef OutputIterator iterator_type;

private:
    Range current;
    iterator_type out;

public:
    /**
     * Constructor.
     *
     * @param out     Output iterator to which completed ranges will be written.
     */
    RangeCollector(iterator_type out);

    /**
     * Destructor. It will flush any buffered ranges.
     */
    ~RangeCollector();

    /**
     * Adds a new splat to the list.
     *
     * @return The output iterator after the update.
     */
    iterator_type append(Range::scan_type scan, Range::index_type splat);

    /**
     * Force any buffered ranges to be emitted. This is done implicitly
     * by the destructor, so it is only necessary if there are more
     * ranges to be written later with the same object, or if lifetime
     * management makes it inconvenient to destroy the object.
     *
     * @return The output iterator after the update.
     */
    iterator_type flush();
};


template<typename OutputIterator>
RangeCollector<OutputIterator>::RangeCollector(iterator_type out)
    : current(), out(out)
{
}

template<typename OutputIterator>
RangeCollector<OutputIterator>::~RangeCollector()
{
    flush();
}

template<typename OutputIterator>
OutputIterator RangeCollector<OutputIterator>::append(Range::scan_type scan, Range::index_type splat)
{
    if (!current.append(scan, splat))
    {
        *out++ = current;
        current = Range(scan, splat);
    }
    return out;
}

template<typename OutputIterator>
OutputIterator RangeCollector<OutputIterator>::flush()
{
    if (current.size > 0)
    {
        *out++ = current;
        current = Range();
    }
    return out;
}

} // namespace internal
} // namespace Bucket

#endif /* !BUCKET_INTERNAL_H */
