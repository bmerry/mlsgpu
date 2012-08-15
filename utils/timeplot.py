#!/usr/bin/env python
from __future__ import division, print_function
import matplotlib.pyplot as plt
import re

class Action(object):
    color_map = {'compute': 'g', 'push' : 'y', 'get': 'y', 'pop': 'r', 'load': 'b'}

    def __init__(self, name, start, stop):
        self.name = name
        self.start = start
        self.stop = stop

    def get_color(self):
        return self.color_map[self.name]

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

workers = {}
with open('timeplot.log') as f:
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
workers = sorted(list(workers.values()), key = lambda x: x.sort_key())

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Time')
yticks = []
yticklabels = []
for i, w in enumerate(workers):
    xranges = []
    colors = []
    for a in w.actions:
        xranges.append((a.start, a.stop - a.start))
        colors.append(a.get_color())
    yrange = (i, 1)
    ax.broken_barh(xranges, yrange, facecolors = colors)
    yticks.append(i + 0.5)
    yticklabels.append(w.name)
ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)
plt.show()
