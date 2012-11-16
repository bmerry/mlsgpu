#!/usr/bin/env python
from __future__ import division, print_function
import sys
import re
from collections import OrderedDict

def parse_stats(f):
    values = {}
    for line in f:
        if re.match(r'[a-z]+ options: ', line):
            for m in re.findall(r' --([-a-zA-Z0-9]+)=(\S+)', line):
                try:
                    v = int(m[1])
                except ValueError:
                    try:
                        v = float(m[1])
                    except ValueError:
                        v = m[1]
                values[m[0]] = v
        else:
            m = re.match(r'\s*([a-zA-Z0-9.() /:]+?)\s*: (\S+)', line)
            if m:
                try:
                    v = int(m.group(2))
                except ValueError:
                    try:
                        v = float(m.group(2))
                    except ValueError:
                        v = None
                if v is not None:
                    values[m.group(1)] = v

            # No explicit counter for number of device threads, so tease it out
            m = re.match(r'device\.pop\.first: .* : .* \[(\d+)\]', line)
            if m:
                values['device-threads'] = int(m.group(1))
    return values

def print_breakdown(name, times, total = None):
    s = 0.0
    for (key, value) in times.items():
        s += value
    if total is None:
        total = s

    print(name)
    for (key, value) in times.items():
        perc = 100.0 * value / total
        print('    %6.2f%% %s' % (perc, key))
    if total != s:
        perc = 100.0 * (total - s) / total
        print('    %6.2f%% other' % (perc,))

def analyze_bucket_coarse(values):
    times = OrderedDict()
    times['compute'] = values['bucket.coarse.compute']
    times['load'] = values['bucket.coarse.load']
    times['wait-out'] = values['bucket.fine.get'] + values['bucket.fine.push']
    print_breakdown('CoarseBucket', times, values['pass1.time'])

def analyze_bucket_fine(values):
    times = OrderedDict()
    threads = values['bucket-threads']
    times['startup'] = values['bucket.fine.pop.first']
    times['compute'] = values['bucket.fine.compute']
    times['wait-in'] = values['bucket.fine.pop']
    times['wait-out'] = values['device.get'] + values['device.push']
    print_breakdown('FineBucket', times, values['pass1.time'] * threads)

def analyze_device(values):
    times = OrderedDict()
    threads = values['device-threads']
    times['startup'] = values['device.pop.first']
    times['compute'] = values['device.compute']
    times['wait-in'] = values['device.pop']
    times['wait-out'] = values['mesher.get'] + values['mesher.push']
    print_breakdown('Device', times, values['pass1.time'] * threads)

def analyze_mesher(values):
    times = OrderedDict()
    times['startup'] = values['mesher.pop.first']
    times['compute'] = values['mesher.compute']
    times['wait-in'] = values['mesher.pop']
    print_breakdown('Mesher', times, values['pass1.time'])

def analyze(values):
    analyze_bucket_coarse(values)
    analyze_bucket_fine(values)
    analyze_device(values)
    analyze_mesher(values)

def main():
    values = parse_stats(sys.stdin)
    analyze(values)

if __name__ == '__main__':
    main()
