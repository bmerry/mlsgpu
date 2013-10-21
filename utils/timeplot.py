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
    event_re = re.compile(r'^EVENT (\S+) (\S+) ([0-9.]+) ([0-9.]+)$')
    value_re = re.compile(r'^VALUE ([0-9]+)$')
    for line in f:
        m = event_re.match(line)
        if m:
            worker_name = m.group(1)
            action_name = m.group(2)
            start_time = float(m.group(3))
            stop_time = float(m.group(4))
            if worker_name not in workers:
                workers[worker_name] = Worker(worker_name)
            worker = workers[worker_name]
            worker.actions.append(Action(action_name, start_time, stop_time))
        m = value_re.match(line)
        if m:
            worker.actions[-1].value = int(m.group(1))
    workers = sorted(list(workers.values()), key = lambda x: x.sort_key())
    return workers
