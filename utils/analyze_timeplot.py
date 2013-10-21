#!/usr/bin/env python

# mlsgpu: surface reconstruction from point clouds
# Copyright (C) 2013  University of Cape Town
#
# This file is part of mlsgpu.
#
# mlsgpu is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import division, print_function
import sys
import timeplot

def analyze(groups):
    # Compute total time i.e. last endpoint
    total_time = 0
    for group in groups:
        for worker in group:
            for action in worker.actions:
                total_time = max(total_time, action.stop)

    for gid, group in enumerate(groups):
        for worker in group:
            print('{}: {}'.format(gid, worker.name))
            sums = {}
            start = total_time
            stop = 0.0
            active = 0.0
            for action in worker.actions:
                sums.setdefault(action.name, 0.0)
                sums[action.name] += action.stop - action.start
                start = min(start, action.start)
                stop = max(stop, action.stop)
                active += action.stop - action.start
            for key in sorted(list(sums.keys())):
                print('    {}: {:.2f}'.format(key, sums[key]))
            idle = stop - start - active
            print('    inactive: {:.2f} / {:.2f} / {:.2f}'.format(start, idle, total_time - stop))

def main():
    groups = []
    if len(sys.argv) > 1:
        for fname in sys.argv[1:]:
            with open(fname, 'r') as f:
                groups.append(timeplot.load_data(f))
    else:
        groups.append(timeplot.load_data(sys.stdin))
    analyze(groups)

if __name__ == '__main__':
    main()
