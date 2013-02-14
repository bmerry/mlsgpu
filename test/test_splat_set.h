/**
 * @file
 *
 * Utility functions used by @ref TestSplatSet and other classes.
 */

#ifndef TEST_SPLAT_SET_H
#define TEST_SPLAT_SET_H

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <vector>
#include <boost/ptr_container/ptr_vector.hpp>
#include "../src/splat.h"
#include "../src/splat_set.h"

namespace SplatSet
{

/**
 * Implementation of the @ref SplatSet::SubsettableConcept that uses a
 * vector of vectors as the backing store, and assigns splat IDs in a similar
 * way to @ref SplatSet::FileSet.
 */
class VectorsSet : public std::vector<std::vector<Splat> >
{
public:
    static const unsigned int scanIdShift = 40;
    static const splat_id splatIdMask = (splat_id(1) << scanIdShift) - 1;

    splat_id maxSplats() const
    {
        splat_id total = 0;
        for (std::size_t i = 0; i < size(); i++)
        {
            total += at(i).size();
        }
        return total;
    }

    SplatStream *makeSplatStream(bool useOMP = true) const
    {
        return makeSplatStream(&detail::rangeAll, &detail::rangeAll + 1, useOMP);
    }

    BlobStream *makeBlobStream(const Grid &grid, Grid::size_type bucketSize) const
    {
        return new SimpleBlobStream(makeSplatStream(), grid, bucketSize);
    }

    template<typename RangeIterator>
    SplatStream *makeSplatStream(RangeIterator firstRange, RangeIterator lastRange, bool useOMP = false) const
    {
        (void) useOMP;
        return new MySplatStream<RangeIterator>(*this, firstRange, lastRange);
    }

private:
    template<typename RangeIterator>
    class MySplatStream : public SplatStream
    {
    public:
        virtual std::size_t read(Splat *splats, splat_id *splatIds, std::size_t count)
        {
            std::size_t oldCount = count;
            while (count > 0)
            {
                if (curRange == lastRange || cur >> scanIdShift >= owner.size())
                    return oldCount - count;

retry:
                if (cur >= curRange->second)
                {
                    ++curRange;
                    if (curRange == lastRange)
                        return oldCount - count;
                    cur = curRange->first;
                    goto retry;
                }
                std::size_t scan = cur >> scanIdShift;
                splat_id scanEnd = owner[scan].size() + (splat_id(scan) << scanIdShift);
                if (cur >= scanEnd)
                {
                    scan++;
                    if (scan >= owner.size())
                        return oldCount - count;
                    cur = splat_id(scan) << scanIdShift;
                    goto retry;
                }

                std::size_t n = std::min(splat_id(count), scanEnd - cur);
                std::size_t pos = cur & splatIdMask;
                for (std::size_t i = 0; i < n; i++)
                {
                    splats[i] = owner[scan][pos + i];
                    if (splatIds != NULL)
                        splatIds[i] = cur + i;
                }
                splats += n;
                if (splatIds != NULL)
                    splatIds += n;
                count -= n;
                cur += n;
            }
            return oldCount - count;
        }

        MySplatStream(const VectorsSet &owner, RangeIterator firstRange, RangeIterator lastRange)
            : owner(owner), curRange(firstRange), lastRange(lastRange)
        {
            if (curRange != lastRange)
                cur = curRange->first;
        }

    private:
        const VectorsSet &owner;
        splat_id cur;
        RangeIterator curRange, lastRange;
    };
};

} // namespace SplatSet

/**
 * Creates a sample set of splats for use in a test case. The resulting set of
 * splats is intended to be interesting when used with a grid spacing of 2.5
 * and an origin at the origin. Normally one would pass an instance of @ref
 * SplatSet::VectorsSet as @a splats.
 *
 * To make this easy to visualise, all splats are placed on a single Z plane.
 * This plane is along a major boundary, so when bucketing, each block can be
 * expected to appear twice (once on each side of the boundary).
 *
 * To see the splats graphically, save the following to a file and run gnuplot
 * over it. The coordinates are in grid space rather than world space:
 * <pre>
 * set xrange [0:16]
 * set yrange [0:20]
 * set size square
 * set xtics 4
 * set ytics 4
 * set grid
 * plot '-' with points
 * 4 8
 * 12 6.8
 * 12.8 4.8
 * 12.8 7.2
 * 14.8 7.2
 * 14 6.4
 * 4.8 14.8
 * 5.2 14.8
 * 4.8 15.2
 * 5.2 15.2
 * 6.8 12.8
 * 7.2 13.2
 * 10 18
 * e
 * pause -1
 * </pre>
 */
void createSplats(std::vector<std::vector<Splat> > &splats);

#endif /* !TEST_SPLAT_SET_H */
