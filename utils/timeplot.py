#!/usr/bin/env python
from __future__ import division, print_function
import matplotlib.pyplot as plt
import re
import sys

class Action(object):
    color_map = {
            'compute': 'green',
            'bbox': 'gray',
            'push' : 'purple',
            'get': 'yellow',
            'pop': 'red',
            'load': 'blue',
            'write': 'orange',
            'send': 'purple',
            'recv': 'cyan',
            'wait': 'pink'
    }

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
    workers = sorted(list(workers.values()), key = lambda x: x.sort_key())
    return workers

def draw(workers):
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
        ax.broken_barh(xranges, yrange, facecolors = colors, antialiased = True)
        yticks.append(i + 0.5)
        yticklabels.append(w.name)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)

    legend_artists = []
    for value in Action.color_map.values():
        # Create dummy artists for the legend
        legend_artists.append(plt.Rectangle((0, 0), 1, 1, fc = value))
    ax.legend(legend_artists, Action.color_map.keys(),
            ncol = 4,
            bbox_to_anchor = (0.0, 1.02, 1, 0.05), loc = 'center left', mode = 'expand',
            borderaxespad = 0.0)
    plt.show()

def main():
    data = []
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as f:
            data = load_data(f)
    else:
        data = load_data(sys.stdin)
    draw(data)

if __name__ == '__main__':
    main()
