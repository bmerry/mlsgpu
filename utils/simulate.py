#!/usr/bin/env python
from __future__ import division, print_function
import sys
import heapq
import timeplot
from optparse import OptionParser

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
    def __init__(self, simulator, pool_size, sets = 1):
        self.pool_size = pool_size
        self.sets = sets
        self.queue_sem = []
        self.queue = [[] for i in range(sets)]
        self.pool_sem = SimSem(simulator, pool_size)
        for i in range(sets):
            self.queue_sem.append(SimSem(simulator, 0))
        self.pool = [(0, i % sets) for i in range(pool_size)]

    def pop(self, worker, qid):
        if self.queue_sem[qid].get(worker):
            return self.queue[qid].pop(0)
        else:
            return None

    def get(self, worker):
        if self.pool_sem.get(worker):
            return self.pool.pop(0)
        else:
            return (None, None)

    def push(self, item, qid):
        self.queue[qid].append(item)
        self.queue_sem[qid].post()

    def done(self, item, qid):
        self.pool.append((item, qid))
        self.pool_sem.post()

class SimWorker(object):
    def __init__(self, name, inq, outq, qid = 0):
        self.name = name
        self.inq = inq
        self.outq = outq
        self.qid = qid
        self.generator = self.run()

    def run(self):
        time = 0.0
        yield
        while True:
            item = self.inq.pop(self, self.qid)
            while item is None:
                time = yield None
                item = self.inq.pop(self, self.qid)
            for child in item.children:
                time += child.parent_get
                time = yield time
                (out, cqid) = self.outq.get(self)
                while out is None:
                    time = yield None
                    (out, cqid) = self.outq.get(self)
                time += child.parent_push
                time = yield time
                self.outq.push(child, cqid)
            if item.finish > 0:
                time += item.finish
                time = yield time
            self.inq.done(item, self.qid)

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

def simulate(root, options):
    simulator = Simulator()

    fine_threads = options.bucket_threads
    gpus = options.gpus

    coarse_spare = 1
    fine_spare = max(options.bucket_spare, fine_threads)
    mesher_spare = gpus * options.mesher_spare

    all_queue = SimQueue(simulator, 1)
    coarse_queue = SimQueue(simulator, fine_threads + options.coarse_spare)
    fine_queue = SimQueue(simulator, gpus * (1 + fine_spare), gpus)
    mesh_queue = SimQueue(simulator, 1 + mesher_spare)

    simulator.add_worker(SimWorker('coarse', all_queue, coarse_queue))
    for i in range(fine_threads):
        simulator.add_worker(SimWorker('fine', coarse_queue, fine_queue))
    for i in range(gpus):
        simulator.add_worker(SimWorker('device', fine_queue, mesh_queue, i))
    simulator.add_worker(SimWorker('mesher', mesh_queue, None))

    all_queue.get(None)
    all_queue.push(root, 0)
    simulator.run()
    print(simulator.time)

def main():
    parser = OptionParser()
    parser.add_option('--bucket-threads', type = 'int', default = 2)
    parser.add_option('--gpus', type = 'int', default = 1)
    parser.add_option('--bucket-spare', type = 'int', default = 6)
    parser.add_option('--mesher-spare', type = 'int', default = 8)
    parser.add_option('--coarse-spare', type = 'int', default = 1)
    (options, args) = parser.parse_args()

    groups = []
    if args:
        for fname in args:
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
    simulate(root, options)

if __name__ == '__main__':
    main()
