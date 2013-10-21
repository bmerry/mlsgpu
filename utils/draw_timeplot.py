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
import matplotlib.pyplot as plt
import sys
import timeplot

_color_map = {
    'init': 'white',
    'compute': 'green',
    'bbox': 'gray',
    'push' : 'purple',
    'get': 'yellow',
    'get.flush': 'yellow',
    'pop': 'red',
    'load': 'blue',
    'write': 'orange',
    'send': 'purple',
    'recv': 'cyan',
    'wait': 'pink'
}

def get_color(action):
    return _color_map[action.name]

def draw(workers):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Time')
    yticks = []
    yticklabels = []
    bias = 0
    for group in workers:
        for i, w in enumerate(group):
            xranges = []
            colors = []
            for a in w.actions:
                xranges.append((a.start, a.stop - a.start))
                colors.append(get_color(a))
            yrange = (i + bias, 1)
            ax.broken_barh(xranges, yrange, facecolors = colors, antialiased = True)
            yticks.append(i + bias + 0.5)
            yticklabels.append(w.name)
        bias += len(group) + 0.5
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)

    legend_artists = []
    for value in _color_map.values():
        # Create dummy artists for the legend
        legend_artists.append(plt.Rectangle((0, 0), 1, 1, fc = value))
    ax.legend(legend_artists, _color_map.keys(),
            ncol = 4,
            bbox_to_anchor = (0.0, 1.02, 1, 0.05), loc = 'center left', mode = 'expand',
            borderaxespad = 0.0)
    plt.show()

def main():
    groups = []
    if len(sys.argv) > 1:
        for fname in sys.argv[1:]:
            with open(fname, 'r') as f:
                groups.append(timeplot.load_data(f))
    else:
        groups.append(timeplot.load_data(sys.stdin))
    draw(groups)

if __name__ == '__main__':
    main()
