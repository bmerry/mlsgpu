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
 * Accumulate buckets to process as a batch.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <boost/function.hpp>
#include <boost/ref.hpp>
#include "splat_set.h"
#include "statistics.h"
#include "allocator.h"
#include "timeplot.h"
#include "chunk_id.h"
#include "bucket.h"
#include "bucket_collector.h"

BucketCollector::BucketCollector(SplatSet::splat_id maxSplats, Functor functor)
    : maxSplats(maxSplats), functor(functor),
    bins("mem.BucketCollector.bins"), numSplats(0),
    binsStat(Statistics::getStatistic<Statistics::Variable>("bucket.collector.bins")),
    splatsStat(Statistics::getStatistic<Statistics::Variable>("bucket.collector.splats"))
{
}

void BucketCollector::operator()(
    const SplatSet::SubsetBase &splats,
    const Grid &grid,
    const Bucket::Recursion &recursionState)
{
    if (numSplats + splats.numSplats() > maxSplats)
        flush();

    if (recursionState.chunk != curChunkId.coords)
    {
        curChunkId.gen++;
        curChunkId.coords = recursionState.chunk;
    }

    bins.push_back(Bin());
    Bin &bin = bins.back();
    bin.ranges = splats;
    bin.grid = grid;
    bin.chunkId = curChunkId;

    numSplats += splats.numSplats();
}

void BucketCollector::flush()
{
    if (bins.empty())
        return;

    binsStat.add(bins.size());
    splatsStat.add(numSplats);

    boost::unwrap_ref(functor)(bins);

    bins.clear();
    numSplats = 0;
}
