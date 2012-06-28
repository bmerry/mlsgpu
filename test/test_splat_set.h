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

namespace detail
{

/**
 * Splat-only part of @ref SplatSet::VectorsSet.
 */
class SimpleVectorsSet : public std::vector<std::vector<Splat> >
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

    SplatStream *makeSplatStream() const
    {
        return makeSplatStream(&rangeAll, &rangeAll + 1);
    }

    template<typename RangeIterator>
    SplatStream *makeSplatStream(RangeIterator firstRange, RangeIterator lastRange) const
    {
        return new MySplatStream<RangeIterator>(*this, firstRange, lastRange);
    }

private:
    template<typename RangeIterator>
    class MySplatStream : public SplatStream
    {
    public:
        virtual Splat operator*() const
        {
            return owner.at(cur >> scanIdShift).at(cur & splatIdMask);
        }

        virtual SplatStream &operator++()
        {
            MLSGPU_ASSERT(!empty(), std::length_error);
            cur++;
            refill();
            return *this;
        }

        virtual bool empty() const
        {
            return curRange == lastRange;
        }

        virtual splat_id currentId() const
        {
            return cur;
        }

        MySplatStream(const SimpleVectorsSet &owner, RangeIterator firstRange, RangeIterator lastRange)
            : owner(owner), curRange(firstRange), lastRange(lastRange)
        {
            if (curRange != lastRange)
                cur = curRange->first;
            refill();
        }

    private:
        const SimpleVectorsSet &owner;
        splat_id cur;
        RangeIterator curRange, lastRange;

        /// Find the next usable element from the current range, if any
        bool refillRange()
        {
            std::size_t scan = cur >> scanIdShift;
            while (cur < curRange->second && scan < owner.size())
            {
                splat_id scanEnd = (scan << scanIdShift) + owner.at(scan).size();
                if (curRange->second < scanEnd)
                    scanEnd = curRange->second;
                while (cur < scanEnd)
                {
                    if (owner.at(scan).at(cur & splatIdMask).isFinite())
                        return true;
                    cur++;
                }
                scan++;
                cur = scan << scanIdShift;
            }
            return false;
        }

        void refill()
        {
            while (curRange != lastRange)
            {
                if (refillRange())
                    return;
                ++curRange;
                if (curRange != lastRange)
                    cur = curRange->first;
            }
        }
    };
};

} // namespace detail

/**
 * Implementation of the @ref SplatSet::SubsettableConcept that uses a
 * vector of vectors as the backing store, and assigns splat IDs in a similar
 * way to @ref SplatSet::FileSet.
 */
typedef detail::BlobbedSet<detail::SimpleVectorsSet> VectorsSet;

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
