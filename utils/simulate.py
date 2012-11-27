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

    def total_time(self):
        ans = self.finish
        for x in self.children:
            ans += x.parent_get
            ans += x.parent_push
        return ans

class EndQItem(object):
    def __init__(self):
        pass

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
            if action.value is not None:
                item.size = action.value
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

    def find_waiter(self, waiter):
        for (i, w) in enumerate(self.waiters):
            if w[0] == waiter:
                return i
        return None

    def post(self, size):
        self.value += size
        if self.waiters and self.waiters[0][1] <= self.value:
            self.simulator.wakeup(self.waiters[0][0])

    def get(self, worker, size):
        if self.value >= size:
            self.value -= size
            idx = self.find_waiter(worker)
            if idx is not None:
                del self.waiters[idx]
            return True
        else:
            idx = self.find_waiter(worker)
            if idx is None:
                self.waiters.append((worker, size))
            else:
                self.waiters[idx] = (worker, size)
            return False

class SimQueue(object):
    def __init__(self, simulator, pool_size):
        self.pool_size = pool_size
        self.queue_sem = SimSem(simulator, 0)
        self.queue = []
        self.pool_sem = SimSem(simulator, pool_size)

    def spare(self):
        return self.pool_sem.value

    def pop(self, worker):
        if self.queue_sem.get(worker, 1):
            return self.queue.pop(0)
        else:
            return None

    def get(self, worker, size):
        assert size <= self.pool_size
        if self.pool_sem.get(worker, size):
            return True
        else:
            return False

    def push(self, item):
        self.queue.append(item)
        self.queue_sem.post(1)

    def done(self, size):
        self.pool_sem.post(size)

class SimWorker(object):
    def __init__(self, name, inq, outqs, options):
        self.name = name
        self.inq = inq
        self.outqs = outqs
        self.generator = self.run()
        self.by_size = options.by_size
        self.running = False

    def best_queue(self):
        return max(self.outqs, key = lambda x: x.spare())

    def run(self):
        running = True
        time = 0.0
        yield
        while True:
            item = self.inq.pop(self)
            while item is None:
                time = yield None
                item = self.inq.pop(self)
            if isinstance(item, EndQItem):
                for q in self.outqs:
                    q.push(item)
                self.inq.push(item) # Make sure it kills off all workers
                break
            for child in item.children:
                if self.by_size:
                    size = child.size
                else:
                    size = 1
                time += child.parent_get
                time = yield time
                outq = self.best_queue()
                success = outq.get(self, size)
                while not success:
                    time = yield None
                    success = outq.get(self, size)
                time += child.parent_push
                time2 = yield time
                assert time2 == time
                outq.push(child)
            if item.finish > 0:
                time += item.finish
                time2 = yield time
                assert time2 == time
            self.inq.done(item.size)
        running = False

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
        if time is None:
            time = self.time
        assert time >= self.time
        heapq.heappush(self.wakeup_queue, (time, worker))

    def run(self):
        self.time = 0.0
        running = len(self.workers)
        while self.wakeup_queue:
            (self.time, worker) = heapq.heappop(self.wakeup_queue)
            try:
                restart_time = worker.generator.send(self.time)
                if restart_time is not None:
                    assert restart_time >= self.time
                    self.wakeup(worker, restart_time)
            except StopIteration:
                running -= 1
        if running != 0:
            print("Workers still running: possible deadlock", file = sys.stderr)
            for w in self.workers:
                if w.running:
                    print("  " + w.name, file = sys.stderr)
            sys.exit(1)

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

    if options.infinite:
        big = 10**30
        coarse_cap = big
        fine_cap = big
        mesher_cap = big
    elif options.by_size:
        coarse_cap = options.coarse_cap * 1024 * 1024
        fine_cap = options.bucket_cap * 1024 * 1024
        mesher_cap = options.mesher_cap * 1024 * 1024
    else:
        coarse_cap = fine_threads + options.coarse_spare
        fine_cap = 1 + fine_spare
        mesher_cap = gpus * (1 + fine_spare)

    all_queue = SimQueue(simulator, 1)
    coarse_queue = SimQueue(simulator, coarse_cap)
    fine_queues = [SimQueue(simulator, fine_cap) for i in range(gpus)]
    mesh_queue = SimQueue(simulator, mesher_cap)

    simulator.add_worker(SimWorker('coarse', all_queue, [coarse_queue], options))
    for i in range(fine_threads):
        simulator.add_worker(SimWorker('fine', coarse_queue, fine_queues, options))
    for i in range(gpus):
        simulator.add_worker(SimWorker('device', fine_queues[i], [mesh_queue], options))
    simulator.add_worker(SimWorker('mesher', mesh_queue, [], options))

    all_queue.push(root)
    all_queue.push(EndQItem())
    simulator.run()
    print(simulator.time)

def main():
    parser = OptionParser()
    parser.add_option('--by-size', action = 'store_true')
    parser.add_option('--infinite', action = 'store_true')
    parser.add_option('--bucket-threads', type = 'int', metavar = 'THREADS', default = 2)
    parser.add_option('--gpus', type = 'int', default = 1)
    parser.add_option('--coarse-spare', type = 'int', metavar = 'SLOTS', default = 1)
    parser.add_option('--bucket-spare', type = 'int', metavar = 'SLOTS', default = 6)
    parser.add_option('--mesher-spare', type = 'int', metavar = 'SLOTS', default = 8)
    parser.add_option('--coarse-cap', type = 'int', metavar = 'MiB', default = 2 * 1024)
    parser.add_option('--bucket-cap', type = 'int', metavar = 'MiB', default = 512)
    parser.add_option('--mesher-cap', type = 'int', metavar = 'MiB', default = 256)
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
