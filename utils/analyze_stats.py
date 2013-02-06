#!/usr/bin/env python
from __future__ import division, print_function
import sys
import re
from collections import OrderedDict

def is_time_key(key):
    time_suffices = [
        '.compute',
        '.load',
        '.write',
        '.pop',
        '.pop.first',
        '.get',
        '.push',
        '.recv',
        '.send']
    for suffix in time_suffices:
        if key.endswith(suffix):
            return True
    return False

common_names = {
    'bucket.coarse.compute': ('main', 'compute'),
    'bucket.compute': ('main', 'compute'),
    'bucket.coarse.load': ('main', 'load'),
    'bucket.coarse.write': ('main', 'write'),

    'bucket.fine.compute': ('copy', 'compute'),
    'bucket.fine.pop.first': ('copy', 'startup'),
    'bucket.fine.pop': ('copy', 'wait-in'),
    'bucket.fine.write': ('copy', 'write'),
    'copy.compute': ('copy', 'compute'),
    'copy.pop.first': ('copy', 'startup'),
    'copy.pop': ('copy', 'wait-in'),
    'copy.write': ('copy', 'write'),

    'device.get': ('copy', 'wait-out'),
    'device.push': ('copy', 'wait-out'),
    'device.flush': ('copy', 'wait-out'),
    'device.compute': ('device', 'compute'),
    'device.pop.first': ('device', 'startup'),
    'device.pop': ('device', 'wait-in'),

    'mesher.compute': ('mesher', 'compute'),
    'mesher.pop.first': ('mesher', 'startup'),
    'mesher.pop': ('mesher', 'wait-in'),

    'tmpwriter.get': ('mesher', 'wait-out'),
    'tmpwriter.push': ('mesher', 'wait-out'),
    'tmpwriter.compute': ('tmpwriter', 'compute'),
    'tmpwriter.pop.first': ('tmpwriter', 'startup'),
    'tmpwriter.pop': ('tmpwriter', 'wait-in')
}

mpi_names = {
    'scatter.get': ('main', 'wait-out'),
    'scatter.push': ('main', 'wait-out'),
    'slave.pop.first': ('slave', 'startup'),
    'slave.pop': ('slave', 'wait-in'),
    'slave.recv': ('slave', 'receive'),
    'bucket.loader.load': ('slave', 'load'),
    'bucket.loader.write': ('slave', 'write'),
    'bucket.loader.compute': ('slave', 'compute'),
    'bucket.fine.get': ('slave', 'wait-out'),
    'bucket.fine.push': ('slave', 'wait-out'),
    'copy.get': ('slave', 'wait-out'),
    'copy.push': ('slave', 'wait-out'),
    'gather.get': ('device', 'wait-out'),
    'gather.compute': ('gather', 'compute'),
    'gather.pop.first': ('gather', 'startup'),
    'gather.pop': ('gather', 'wait-in'),
    'ReceiverGather.wait': ('receiver', 'wait-in'),
    'ReceiverGather.recv': ('receiver', 'receive'),
    'mesher.get': ('receiver', 'wait-out'),
    'mesher.push': ('receiver', 'wait-out'),
}

nompi_names = {
    'bucket.loader.compute': ('main', 'compute'),
    'bucket.loader.load': ('main', 'load'),
    'bucket.loader.write': ('main', 'write'),
    'bucket.fine.get': ('main', 'wait-out'),
    'bucket.fine.push': ('main', 'wait-out'),
    'copy.get': ('main', 'wait-out'),
    'copy.push': ('main', 'wait-out'),
    'mesher.get': ('device', 'wait-out'),
    'mesher.push': ('device', 'wait-out'),
}

order = ['startup', 'compute', 'receive', 'send', 'load', 'write', 'wait-in', 'wait-out']

def order_key(value):
    try:
        return order.index(value[0])
    except ValueError:
        return len(order)

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
                        v = m.group(2)
                        if v == '[0]':  # Happens if a statistic has 0 samples
                            v = 0.0
                if v is not None:
                    values[m.group(1)] = v

            # No explicit counters for these, so extract from elsewhere
            m = re.match(r'device\.pop\.first: .* : .* \[(\d+)\]', line)
            if m:
                values['total-device-threads'] = int(m.group(1))
            m = re.match(r'(?:bucket\.fine|copy)\.pop\.first: .* : .* \[(\d+)\]', line)
            if m:
                values['total-bucket-threads'] = int(m.group(1))
            m = re.match(r'gather\.pop\.first: .* : .* \[(\d+)\]', line)
            if m:
                values['total-slave-threads'] = int(m.group(1))
    return values

def print_breakdown(name, times, total = None):
    s = 0.0
    items = list(times.items())
    items.sort(key = order_key)

    for (key, value) in items:
        s += value
    if total is None:
        total = s

    print(name)
    for (key, value) in items:
        perc = 100.0 * value / total
        print('    %6.2f%% %s' % (perc, key))
    if total != s:
        perc = 100.0 * (total - s) / total
        print('    %6.2f%% other' % (perc,))

def extract_workers(values):
    workers = {}
    names = dict(common_names)
    if 'gather.pop' in values:
        names.update(mpi_names)
    else:
        names.update(nompi_names)

    for (key, value) in values.items():
        if key in names:
            (worker, action) = names[key]
            workers.setdefault(worker, {})
            workers[worker].setdefault(action, 0.0)
            workers[worker][action] += value
        elif is_time_key(key):
            print("WARNING: do not know what to do with", key, file = sys.stderr)
    return workers

def analyze_main(workers, values):
    print_breakdown('Main', workers['main'], values['pass1.time'])

def analyze_slave(workers, values):
    if 'slave' in workers:
        threads = values['total-slave-threads']
        print_breakdown('Slave', workers['slave'], values['pass1.time'] * threads)

def analyze_copy(workers, values):
    threads = values['total-bucket-threads']
    print_breakdown('Copy', workers['copy'], values['pass1.time'] * threads)

def analyze_device(workers, values):
    threads = values['total-device-threads']
    print_breakdown('Device', workers['device'], values['pass1.time'] * threads)

def analyze_gather(workers, values):
    if 'gather' in workers:
        threads = values['total-slave-threads']
        print_breakdown('Gather', workers['gather'], values['pass1.time'] * threads)

def analyze_receiver(workers, values):
    if 'ReceiverGather.wait' in values:
        print_breakdown('Receiver', workers['receiver'], values['pass1.time'])

def analyze_mesher(workers, values):
    print_breakdown('Mesher', workers['mesher'], values['pass1.time'])

def analyze_tmpwriter(workers, values):
    if 'tmpwriter' in workers:
        print_breakdown('TmpWriter', workers['tmpwriter'], values['pass1.time'])

def analyze(values):
    workers = extract_workers(values)
    analyze_main(workers, values)
    analyze_slave(workers, values)
    analyze_copy(workers, values)
    analyze_device(workers, values)
    analyze_gather(workers, values)
    analyze_receiver(workers, values)
    analyze_mesher(workers, values)
    analyze_tmpwriter(workers, values)

def main():
    values = parse_stats(sys.stdin)
    analyze(values)

if __name__ == '__main__':
    main()
