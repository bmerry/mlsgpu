#!/usr/bin/env python
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
