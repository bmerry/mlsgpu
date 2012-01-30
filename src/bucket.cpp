/**
 * @file
 *
 * Bucketing of splats into sufficiently small buckets.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <vector>
#include <tr1/cstdint>
#include <limits>
#include <stdexcept>
#include <algorithm>
#include <utility>
#include <cassert>
#include <functional>
#include <boost/array.hpp>
#include <boost/multi_array.hpp>
#include <boost/foreach.hpp>
#include <boost/ref.hpp>
#include <boost/numeric/conversion/converter.hpp>
#include <stxxl.h>
#include "splat.h"
#include "bucket.h"
#include "bucket_internal.h"
#include "bucket_impl.h"
#include "errors.h"
#include "statistics.h"
#include "timer.h"
#include "misc.h"

namespace Bucket
{

Range::Range() :
    scan(std::numeric_limits<scan_type>::max()),
    size(0),
    start(std::numeric_limits<index_type>::max())
{
}

Range::Range(scan_type scan, index_type splat) :
    scan(scan),
    size(1),
    start(splat)
{
}

Range::Range(scan_type scan, index_type start, size_type size)
    : scan(scan), size(size), start(start)
{
    MLSGPU_ASSERT(size == 0 || start <= std::numeric_limits<index_type>::max() - size + 1, std::out_of_range);
}

bool Range::append(scan_type scan, index_type splat)
{
    if (size == 0)
    {
        /* An empty range can always be extended. */
        this->scan = scan;
        size = 1;
        start = splat;
    }
    else if (this->scan == scan && splat >= start && splat - start <= size)
    {
        if (splat - start == size)
        {
            if (size == std::numeric_limits<size_type>::max())
                return false; // would overflow
            size++;
        }
    }
    else
        return false;
    return true;
}

namespace internal
{

Cell::Cell(const size_type lower[3], const size_type upper[3], unsigned int level) : level(level)
{
    for (unsigned int i = 0; i < 3; i++)
    {
        MLSGPU_ASSERT(lower[i] < upper[i], std::invalid_argument);
        this->lower[i] = lower[i];
        this->upper[i] = upper[i];
    }
}

Cell::Cell(
    size_type lowerX, size_type lowerY, size_type lowerZ,
    size_type upperX, size_type upperY, size_type upperZ,
    unsigned int level) : level(level)
{
    lower[0] = lowerX;
    lower[1] = lowerY;
    lower[2] = lowerZ;
    upper[0] = upperX;
    upper[1] = upperY;
    upper[2] = upperZ;
    for (unsigned int i = 0; i < 3; i++)
        MLSGPU_ASSERT(lower[i] < upper[i], std::invalid_argument);
}

bool Cell::operator==(const Cell &c) const
{
    return std::equal(lower, lower + 3, c.lower)
        && std::equal(upper, upper + 3, c.upper)
        && level == c.level;
}

Cell Cell::child(unsigned int idx) const
{
    MLSGPU_ASSERT(level > 0, std::invalid_argument);
    MLSGPU_ASSERT(idx < 8, std::invalid_argument);
    Cell c = *this;
    c.level--;
    for (int i = 0; i < 3; i++)
    {
        size_type mid = (lower[i] + upper[i]) / 2;
        if ((idx >> i) & 1)
            c.lower[i] = mid;
        else
            c.upper[i] = mid;
    }
    return c;
}

RangeCounter::RangeCounter() : ranges(0), splats(0), current()
{
}

void RangeCounter::append(Range::scan_type scan, Range::index_type splat)
{
    splats++;
    /* On the first call, the append will succeed (empty range), but we still
     * need to set ranges to 1 since this is the first real range.
     */
    if (ranges == 0 || !current.append(scan, splat))
    {
        current = Range(scan, splat);
        ranges++;
    }
}

std::tr1::uint64_t RangeCounter::countRanges() const
{
    return ranges;
}

std::tr1::uint64_t RangeCounter::countSplats() const
{
    return splats;
}

const std::size_t BucketState::BAD_BLOCK;

BucketState::BucketState(
    const BucketParameters &params, const Grid &grid,
    Cell::size_type microSize, int macroLevels)
    : params(params), grid(grid), microSize(microSize), macroLevels(macroLevels),
    cellStates(macroLevels), nextOffset(0)
{
    for (int i = 0; i < 3; i++)
        this->dims[i] = grid.numCells(i);

    for (int level = 0; level < macroLevels; level++)
    {
        /* We don't need a full power-of-two allocation for a level of the octree,
         * just enough to completely cover the original dimensions.
         */
        boost::array<Cell::size_type, 3> s;
        for (int i = 0; i < 3; i++)
        {
            s[i] = divUp(dims[i], microSize << level);
            assert(level != macroLevels - 1 || s[i] == 1);
        }
        cellStates[level].resize(s);
    }
}

BucketState::CellState &BucketState::getCellState(const Cell &cell)
{
    boost::array<Cell::size_type, 3> coords;
    Cell::size_type factor = microSize << cell.getLevel();
    for (int i = 0; i < 3; i++)
    {
        assert(cell.getLower()[i] % factor == 0);
        coords[i] = cell.getLower()[i] / factor;
    }
    assert(cell.getLevel() < cellStates.size());
    return cellStates[cell.getLevel()](coords);
}

const BucketState::CellState &BucketState::getCellState(const Cell &cell) const
{
    return const_cast<BucketState *>(this)->getCellState(cell);
}

bool CountSplat::operator()(Range::scan_type scan, Range::index_type id,
                            const Splat &splat, const Cell &cell) const
{
    (void) splat;

    // Add to the counters
    state.getCellState(cell).counter.append(scan, id);

    // Recurse into children, unless we've reached microblock level
    return cell.getLevel() > 0;
}

bool PickCells::operator()(const Cell &cell) const
{
    BucketState::CellState &cs = state.getCellState(cell);

    // Skip completely empty regions
    if (cs.counter.countSplats() == 0)
        return false;

    if (cell.getLevel() == 0
        || (cell.getSize(0) <= state.params.maxCells
            && cell.getSize(1) <= state.params.maxCells
            && cell.getSize(2) <= state.params.maxCells
            && cs.counter.countSplats() <= state.params.maxSplats))
    {
        cs.blockId = state.picked.size();
        state.picked.push_back(cell);
        state.pickedOffset.push_back(state.nextOffset);
        state.nextOffset += cs.counter.countRanges();
        return false; // no more recursion required
    }
    else
        return true;
}

bool BucketSplats::operator()(Range::scan_type scan, Range::index_type id,
                              const Splat &splat, const Cell &cell) const
{
    (void) splat;

    BucketState::CellState &cs = state.getCellState(cell);
    if (cs.blockId == BucketState::BAD_BLOCK)
    {
        return true; // this cell is too coarse, so refine recursively
    }
    else
    {
        assert(cs.blockId < state.childCur.size());
        state.childCur[cs.blockId].append(scan, id);
        return false;
    }
}

/**
 * Pick the smallest possible power of 2 size for a microblock.
 * The limitation is thta there must be at most @a maxSplit microblocks.
 */
Cell::size_type chooseMicroSize(
    const Cell::size_type dims[3], std::size_t maxSplit)
{
    Cell::size_type microSize = 1;
    std::size_t microBlocks = 1;
    for (int i = 0; i < 3; i++)
        microBlocks = mulSat(microBlocks, std::size_t(divUp(dims[i], microSize)));
    while (microBlocks > maxSplit)
    {
        microSize *= 2;
        microBlocks = 1;
        for (int i = 0; i < 3; i++)
            microBlocks = mulSat(microBlocks, std::size_t(divUp(dims[i], microSize)));
    }
    return microSize;
}

const Splat &MakeGrid::operator()(const Splat &splat)
{
    const float radius = splat.radius;
    if (first)
    {
        for (unsigned int i = 0; i < 3; i++)
        {
            low[i] = splat.position[i];
            bboxMin[i] = low[i] - radius;
            bboxMax[i] = low[i] + radius;
        }
        first = false;
    }
    else
    {
        for (unsigned int j = 0; j < 3; j++)
        {
            float p = splat.position[j];
            low[j] = std::min(low[j], p);
            bboxMin[j] = std::min(bboxMin[j], p - radius);
            bboxMax[j] = std::max(bboxMax[j], p + radius);
        }
    }
    return splat;
}

} // namespace internal

} // namespace Bucket
