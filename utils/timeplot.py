#!/usr/bin/env python
from __future__ import division, print_function
import re

class Action(object):
    def __init__(self, name, start, stop):
        self.name = name
        self.start = start
        self.stop = stop
        self.value = None

class Worker(object):
    def __init__(self, name):
        self.name = name
        self.actions = []

    def sort_key(self):
        m = re.match(r'^(.*)\.(\d+)$', self.name)
        if m:
            return (m.group(1), int(m.group(2)))
        else:
            return (m,)

def load_data(f):
    workers = {}
    for line in f:
        m = re.match(r'^EVENT (\S+) (\S+) ([0-9.]+) ([0-9.]+)$', line)
        if m:
            worker_name = m.group(1)
            action_name = m.group(2)
            start_time = float(m.group(3))
            stop_time = float(m.group(4))
            if worker_name not in workers:
                workers[worker_name] = Worker(worker_name)
            worker = workers[worker_name]
            worker.actions.append(Action(action_name, start_time, stop_time))
        m = re.match(r'^VALUE ([0-9]+)$', line)
        if m:
            worker.actions[-1].value = int(m.group(1))
    workers = sorted(list(workers.values()), key = lambda x: x.sort_key())
    return workers
