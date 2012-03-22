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

namespace internal
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
        return new MySplatStream(*this, 0, splat_id(size()) << scanIdShift);
    }

    SplatStreamReset *makeSplatStreamReset() const
    {
        return new MySplatStream(*this, 0, 0);
    }

private:
    class MySplatStream : public SplatStreamReset
    {
    public:
        virtual const Splat &operator*() const
        {
            return owner.at(scan).at(offset);
        }

        virtual SplatStream &operator++()
        {
            MLSGPU_ASSERT(!empty(), std::runtime_error);
            offset++;
            refill();
            return *this;
        }

        virtual bool empty() const
        {
            return ((scan << scanIdShift) | offset) >= last;
        }

        virtual void reset(splat_id first, splat_id last)
        {
            MLSGPU_ASSERT(first <= last, std::invalid_argument);
            last = std::min(last, splat_id(owner.size()) << scanIdShift);
            first = std::min(first, last);
            scan = first >> scanIdShift;
            offset = first & splatIdMask;
            this->last = last;
            refill();
        }

        virtual splat_id currentId() const
        {
            return (scan << scanIdShift) | offset;
        }

        MySplatStream(const SimpleVectorsSet &owner, splat_id first, splat_id last)
            : owner(owner)
        {
            reset(first, last);
        }

    private:
        const SimpleVectorsSet &owner;
        splat_id scan, offset;
        SplatSet::splat_id last;

        void skipNonFiniteInScan()
        {
            while (scan < owner.size()
                   && offset < owner.at(scan).size()
                   && !owner.at(scan)[offset].isFinite())
                offset++;
        }

        void refill()
        {
            skipNonFiniteInScan();
            while (!MySplatStream::empty() && offset == owner.at(scan).size())
            {
                scan++;
                offset = 0;
                skipNonFiniteInScan();
            }
        }
    };
};

} // namespace internal

/**
 * Implementation of the @ref SplatSet::SubsettableConcept that uses a
 * vector of vectors as the backing store, and assigns splat IDs in a similar
 * way to @ref SplatSet::FileSet.
 */
typedef internal::BlobbedSet<internal::SimpleVectorsSet> VectorsSet;

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
