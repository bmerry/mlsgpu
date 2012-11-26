#!/usr/bin/env python
from __future__ import division, print_function
import sys
import heapq
import timeplot

class QItem(object):
    def __init__(self, parent, parent_get, parent_push):
        self.parent = parent
        self.size = 1
        self.finish = 0.0
        self.parent_get = parent_get
        self.parent_push = parent_push
        self.children = []

def process_worker(worker, pq):
    pqid = 0
    item = None
    cq = []
    for action in worker.actions:
        if action.name in ['bbox', 'pop']:
            if pqid == len(pq):
                break
            item = pq[pqid]
            pqid += 1
            base = action.stop
        elif action.name == 'get':
            parent_get = action.start - base
            base = action.stop
        elif action.name == 'push':
            parent_push = action.start - base
            base = action.stop
            child = QItem(item, parent_get, parent_push)
            item.children.append(child)
            cq.append(child)
            item.finish = 0.0
        elif action.name in ['compute', 'load']:
            item.finish += action.stop - action.start
        elif action.name in ['write']:
            pass
        else:
            raise ValueError('Unhandled action "' + action.name + '"')
    if pqid != len(pq):
        raise ValueError('Parent queue was not exhausted')
    return cq

def get_worker(group, name):
    for worker in group:
        if worker.name == name:
            return worker
    return None

class SimSem(object):
    def __init__(self, simulator, value = 0):
        self.simulator = simulator
        self.value = value
        self.waiters = []

    def post(self):
        self.value += 1
        if self.waiters:
            self.simulator.wakeup(self.waiters[0])

    def get(self, worker):
        if self.value > 0:
            self.value -= 1
            if worker in self.waiters:
                self.waiters.remove(worker)
            return True
        else:
            if worker not in self.waiters:
                self.waiters.append(worker)
            return False

class SimQueue(object):
    def __init__(self, simulator, pool_size):
        self.pool_size = pool_size
        self.queue_sem = SimSem(simulator, 0)
        self.pool_sem = SimSem(simulator, pool_size)
        self.queue = []
        self.pool = [0] * pool_size

    def pop(self, worker):
        if self.queue_sem.get(worker):
            return self.queue.pop(0)
        else:
            return None

    def get(self, worker):
        if self.pool_sem.get(worker):
            return self.pool.pop(0)
        else:
            return None

    def push(self, item):
        self.queue.append(item)
        self.queue_sem.post()

    def done(self, item):
        self.pool.append(item)
        self.pool_sem.post()

class SimWorker(object):
    def __init__(self, name, inq, outq):
        self.name = name
        self.inq = inq
        self.outq = outq
        self.generator = self.run()

    def run(self):
        time = 0.0
        yield
        while True:
            item = self.inq.pop(self)
            while item is None:
                time = yield None
                item = self.inq.pop(self)
            for child in item.children:
                time += child.parent_get
                time = yield time
                out = self.outq.get(self)
                while out is None:
                    time = yield None
                    out = self.outq.get(self)
                time += child.parent_push
                time = yield time
                self.outq.push(child)
            if item.finish > 0:
                time += item.finish
                time = yield time
            self.inq.done(item)

class Simulator(object):
    def __init__(self):
        self.workers = []
        self.wakeup_queue = []
        self.time = 0.0

    def add_worker(self, worker):
        self.workers.append(worker)
        worker.generator.send(None)
        self.wakeup(worker)

    def wakeup(self, worker, time = None):
        assert worker not in self.wakeup_queue
        if time is None:
            time = self.time
        assert time >= self.time
        heapq.heappush(self.wakeup_queue, (time, worker))

    def run(self):
        self.time = 0.0
        while self.wakeup_queue:
            (self.time, worker) = heapq.heappop(self.wakeup_queue)
            try:
                restart_time = worker.generator.send(self.time)
                if restart_time is not None:
                    self.wakeup(worker, restart_time)
            except StopIteration:
                pass

def load_items(group):
    all_queue = [QItem(None, 0.0, 0.0)]
    coarse_queue = process_worker(get_worker(group, 'main'), all_queue)
    fine_queue = process_worker(get_worker(group, 'bucket.fine.0'), coarse_queue)
    mesh_queue = process_worker(get_worker(group, 'device.0'), fine_queue)
    process_worker(get_worker(group, 'mesher.0'), mesh_queue)
    return all_queue[0]

def simulate(root):
    simulator = Simulator()

    all_queue = SimQueue(simulator, 1)
    coarse_queue = SimQueue(simulator, 2)
    fine_queue = SimQueue(simulator, 7)
    mesh_queue = SimQueue(simulator, 9)

    simulator.add_worker(SimWorker('coarse', all_queue, coarse_queue))
    simulator.add_worker(SimWorker('fine', coarse_queue, fine_queue))
    simulator.add_worker(SimWorker('device', fine_queue, mesh_queue))
    simulator.add_worker(SimWorker('mesher', mesh_queue, None))

    all_queue.get(None)
    all_queue.push(root)
    simulator.run()
    print(simulator.time)

def main():
    groups = []
    if len(sys.argv) > 1:
        for fname in sys.argv[1:]:
            with open(fname, 'r') as f:
                groups.append(timeplot.load_data(f))
    else:
        groups.append(timeplot.load_data(sys.stdin))
    if len(groups) != 1:
        print("Only one group is supported", file = sys.stderr)
        sys.exit(1)
    group = groups[0]
    for worker in group:
        if worker.name.endswith('.1'):
            print("Only one worker of each type is supported", file = sys.stderr)
            sys.exit(1)

    root = load_items(group)
    simulate(root)

if __name__ == '__main__':
    main()
