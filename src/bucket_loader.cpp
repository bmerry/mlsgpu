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
 * Loads buckets of data from a SplatSet and passes it to a worker group.
 */
#if HAVE_CONFIG_H
# include <config.h>
#endif

#include <boost/smart_ptr/shared_ptr.hpp>
#include <boost/smart_ptr/scoped_ptr.hpp>
#include <boost/foreach.hpp>
#include <cassert>
#include "workers.h"
#include "grid.h"
#include "statistics.h"
#include "splat_set.h"
#include "timeplot.h"
#include "bucket_loader.h"

BucketLoader::BucketLoader(
    std::size_t maxItemSplats, CopyGroup &outGroup, Timeplot::Worker &tworker)
    :
    maxItemSplats(maxItemSplats),
    outGroup(outGroup),
    tworker(tworker),
    super(NULL),
    splatBuffer("mem.BucketLoader.splatBuffer"),
    computeStat(Statistics::getStatistic<Statistics::Variable>("bucket.loader.compute")),
    loadStat(Statistics::getStatistic<Statistics::Variable>("bucket.loader.load")),
    writeStat(Statistics::getStatistic<Statistics::Variable>("bucket.loader.write"))
{
    splatBuffer.reserve(maxItemSplats);
}

void BucketLoader::operator()(const Statistics::Container::vector<BucketCollector::Bin> &bins)
{
    if (bins.empty())
        return;

    Statistics::Container::vector<range_type> ranges("mem.BucketLoader.ranges");
    {
        Timeplot::Action timer("compute", tworker, computeStat);
        /* Compute merged ranges */
        BOOST_FOREACH(const BucketCollector::Bin &bin, bins)
        {
            Statistics::Container::vector<range_type> tmp("mem.BucketLoader.ranges");
            SplatSet::merge(bin.ranges.begin(), bin.ranges.end(),
                            ranges.begin(), ranges.end(), std::back_inserter(tmp));
            tmp.swap(ranges);
        }
    }

    {
        Timeplot::Action timer("load", tworker, loadStat);
        boost::scoped_ptr<SplatSet::SplatStream> splatStream(super->makeSplatStream(ranges.begin(), ranges.end()));
        float invSpacing = 1.0f / fullGrid.getSpacing();
        std::size_t numRead = splatStream->read(&splatBuffer[0], NULL, maxItemSplats);
        for (std::size_t i = 0; i < numRead; i++)
        {
            Splat &splat = splatBuffer[i];
            /* Transform the splats into the grid's coordinate system */
            fullGrid.worldToVertex(splat.position, splat.position);
            splat.radius *= invSpacing;
        }
    }

    // Now process each bin, copying the relevant subset to the device
    BOOST_FOREACH(const BucketCollector::Bin &bin, bins)
    {
        /* We transformed splats from world space into fullGrid space, so we need to
         * construct a new grid for this coordinate system.
         */
        const float ref[3] = {0.0f, 0.0f, 0.0f};
        Grid subGrid(ref, 1.0f, 0, 1, 0, 1, 0, 1);
        for (unsigned int i = 0; i < 3; i++)
        {
            Grid::difference_type base = fullGrid.getExtent(i).first;
            Grid::difference_type low = bin.grid.getExtent(i).first - base;
            Grid::difference_type high = bin.grid.getExtent(i).second - base;
            subGrid.setExtent(i, low, high);
        }

        boost::shared_ptr<CopyGroup::WorkItem> item = outGroup.get(tworker, bin.ranges.numSplats());
        item->chunkId = bin.chunkId;
        item->grid = subGrid;

        Timeplot::Action timer("write", tworker, writeStat);
        timer.setValue(bin.ranges.numSplats() * sizeof(Splat));

        Statistics::Container::vector<range_type>::const_iterator p = ranges.begin();
        std::size_t pos = 0;
        Splat *splatPtr = (Splat *) item->getSplats();
        for (SplatSet::SubsetBase::const_iterator q = bin.ranges.begin(); q != bin.ranges.end(); ++q)
        {
            while (p->second < q->second)
            {
                pos += p->second - p->first;
                ++p;
            }
            assert(p->first <= q->first && p->second >= q->second);
            std::memcpy(splatPtr, &splatBuffer[pos + (q->first - p->first)],
                   (q->second - q->first) * sizeof(Splat));
            splatPtr += q->second - q->first;
        }
        outGroup.push(tworker, item);
    }
}

void BucketLoader::start(const Splats &super, const Grid &fullGrid)
{
    this->fullGrid = fullGrid;
    this->super = &super;
}
