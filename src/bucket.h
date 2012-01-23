/**
 * @file
 *
 * Bucketing of splats into sufficiently small buckets.
 */

#ifndef BUCKET_H
#define BUCKET_H

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <vector>
#include <tr1/cstdint>
#include <boost/function.hpp>
#include <boost/ptr_container/ptr_array.hpp>
#include "splat.h"
#include "grid.h"
#include "fast_ply.h"

/**
 * Indexes a sequential range of splats from an input file.
 *
 * This is intended to be POD that can be put in a @c stxxl::vector.
 *
 * @invariant @ref start + @ref size - 1 does not overflow @ref index_type.
 * (maintained by constructor and by @ref append).
 */
struct SplatRange
{
    typedef std::tr1::uint32_t scan_type;
    typedef std::tr1::uint32_t size_type;
    typedef std::tr1::uint64_t index_type;

    /* Note: the order of these is carefully chosen for alignment */
    scan_type scan;    ///< Index of the originating file
    size_type size;    ///< Size of the range
    index_type start;  ///< Splat index in the file

    /**
     * Constructs an empty scan range.
     */
    SplatRange();

    /**
     * Constructs a splat range with one splat.
     */
    SplatRange(scan_type scan, index_type splat);

    /**
     * Constructs a splat range with multiple splats.
     *
     * @pre @a start + @a size - 1 must fit within @ref index_type.
     */
    SplatRange(scan_type scan, index_type start, size_type size);

    /**
     * Attempts to extend this range with a new element.
     * @param scan, splat     The new element
     * @retval true if the element was successfully appended
     * @retval false otherwise.
     */
    bool append(scan_type scan, index_type splat);
};

/**
 * Tracks how many ranges are needed to encode a list of splats and
 * how many splats are in the list. It will match the actual number
 * written by @ref SplatRangeCollector.
 */
class SplatRangeCounter
{
private:
    std::tr1::uint64_t ranges;
    std::tr1::uint64_t splats;
    SplatRange current;

public:
    /// Constructor
    SplatRangeCounter();

    /// Adds a new splat to the virtual list.
    void append(SplatRange::scan_type scan, SplatRange::index_type splat);

    /// Returns the number of ranges that would be required to encode the provided splats.
    std::tr1::uint64_t countRanges() const;

    /// Returns the number of splats given to @ref append
    std::tr1::uint64_t countSplats() const;
};

/**
 * Accepts a list of splat IDs and merges them into ranges which are then
 * output.
 * @param OutputIterator an output iterator that accepts assignments of @ref SplatRange.
 */
template<typename OutputIterator>
class SplatRangeCollector
{
public:
    typedef OutputIterator iterator_type;

private:
    SplatRange current;
    iterator_type out;

public:
    /**
     * Constructor.
     *
     * @param out     Output iterator to which completed ranges will be written.
     */
    SplatRangeCollector(iterator_type out);

    /**
     * Destructor. It will flush any buffered ranges.
     */
    ~SplatRangeCollector();

    /**
     * Adds a new splat to the list.
     *
     * @return The output iterator after the update.
     */
    iterator_type append(SplatRange::scan_type scan, SplatRange::index_type splat);

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

typedef std::vector<SplatRange>::const_iterator SplatRangeConstIterator;
typedef boost::function<void(const boost::ptr_vector<FastPly::Reader> &, SplatRange::index_type, SplatRangeConstIterator, SplatRangeConstIterator, const Grid &)> BucketProcessor;

void bucket(const boost::ptr_vector<FastPly::Reader> &files,
            const Grid &bbox,
            SplatRange::index_type maxSplats,
            int maxCells,
            std::size_t maxSplit,
            const BucketProcessor &process);

/**
 * Grid that encloses the bounding spheres of all the input splats.
 *
 * The grid is constructed as follows:
 *  -# The bounding box of the sample points is found, ignoring influence regions.
 *  -# The lower bound is used as the grid reference point.
 *  -# The grid extends are set to cover the full bounding box.
 *
 * @param files         Files containing the splats.
 * @param spacing       The spacing between grid vertices.
 *
 * @pre There is at least one splat.
 */
Grid makeGrid(const boost::ptr_vector<FastPly::Reader> &files,
              float spacing);

template<typename OutputIterator>
SplatRangeCollector<OutputIterator>::SplatRangeCollector(iterator_type out)
    : current(), out(out)
{
}

template<typename OutputIterator>
SplatRangeCollector<OutputIterator>::~SplatRangeCollector()
{
    flush();
}

template<typename OutputIterator>
OutputIterator SplatRangeCollector<OutputIterator>::append(SplatRange::scan_type scan, SplatRange::index_type splat)
{
    if (!current.append(scan, splat))
    {
        *out++ = current;
        current = SplatRange(scan, splat);
    }
    return out;
}

template<typename OutputIterator>
OutputIterator SplatRangeCollector<OutputIterator>::flush()
{
    if (current.size > 0)
    {
        *out++ = current;
        current = SplatRange();
    }
    return out;
}

#endif /* BUCKET_H */
