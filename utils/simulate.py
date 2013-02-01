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
    get_size = None
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
            get_size = action.value
        elif action.name == 'push':
            parent_push = action.start - base
            base = action.stop
            child = QItem(item, parent_get, parent_push)
            if get_size is not None:
                child.size = get_size
                get_size = None
            item.children.append(child)
            cq.append(child)
            item.finish = 0.0
        elif action.name in ['compute', 'load', 'write']:
            if worker.name != 'main' or action.name != 'write':
                # Want to exclude phase 3
                item.finish += action.stop - action.start
        elif action.name in ['init']:
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

class SimPool(object):
    def __init__(self, simulator, size, inorder = True):
        self._size = size
        self._waiters = []
        self._watchers = []
        self._allocs = []
        self._spare = size
        self._inorder = inorder
        self._simulator = simulator

    def spare(self):
        return self._spare

    def _biggest(self):
        """Maximum possible allocation without blocking"""
        if not self._inorder:
            return self._spare
        elif not self._allocs:
            return self._size
        else:
            start = self._allocs[0][0]
            end = self._allocs[-1][1]
            if end > start:
                return max(self._size - end, start)
            else:
                return start - end

    def get(self, worker, size):
        if not self._inorder:
            size = 1
        assert size > 0
        assert size <= self._size
        self._waiters.append((worker, size))
        self._do_wakeups()

    def can_get(self, size):
        if not self._inorder:
            size = 1
        return size <= self._biggest()

    def watch(self, worker):
        '''Request to be woken up when free space increases'''
        self._watches.append(worker)

    def unwatch(self, worker):
        '''Cancel a previous watch request'''
        self._watches.remove(worker)

    def _do_wakeups(self):
        while self._waiters:
            (w, size) = self._waiters[0]
            if size > self._biggest():
                break
            elif not self._allocs:
                start = 0
            elif not self._inorder:
                start = self._allocs[-1][1]
            else:
                cur_start = self._allocs[0][0]
                cur_end = self._allocs[-1][1]
                cur_limit = self._size
                if cur_end <= cur_start:
                    limit = cur_start
                if cur_limit - cur_end >= size:
                    start = cur_end
                else:
                    start = 0
            a = (start, start + size)
            self._allocs.append(a)
            self._spare -= size
            del self._waiters[0]
            self._simulator.wakeup(w, value = a)
        while self._watchers:
            w = self._watchers.pop(0)
            self._simulator.wakeup(w)

    def done(self, alloc):
        self._allocs.remove(alloc)
        self._spare += alloc[1] - alloc[0]
        self._do_wakeups()

class SimSimpleQueue(object):
    """
    Queue without associated pool. Just accepts objects and provides
    a blocking pop.
    """
    def __init__(self, simulator):
        self._queue = []
        self._waiters = []
        self._simulator = simulator
        self._running = True

    def _do_wakeups(self):
        while self._waiters and self._queue:
            item = self._queue.pop(0)
            worker = self._waiters.pop(0)
            self._simulator.wakeup(worker, value = item)
        while self._waiters and not self._running:
            worker = self._waiters.pop(0)
            self._simulator.wakeup(worker, value = EndQItem())

    def pop(self, worker):
        self._waiters.append(worker)
        self._do_wakeups()

    def push(self, item):
        self._queue.append(item)
        self._do_wakeups()

    def stop(self):
        self._running = False
        self._do_wakeups()

class SimQueue(object):
    def __init__(self, simulator, pool_size, inorder = True):
        self._pool = SimPool(simulator, pool_size, inorder)
        self._queue = SimSimpleQueue(simulator)

    def spare(self):
        return self._pool.spare()

    def pop(self, worker):
        self._queue.pop(worker)

    def get(self, worker, size):
        self._pool.get(worker, size)

    def can_get(self, size):
        return self._pool.can_get(worker, size)

    def watch(self, worker):
        self._pool.watch(worker)

    def unwatch(self, worker):
        self._pool.unwatch(worker)

    def push(self, item, alloc):
        self._queue.push(item)

    def done(self, alloc):
        self._pool.done(alloc)

    def watch(self, alloc):
        self._pool

    def stop(self):
        self._queue.stop()

class SimWorker(object):
    def __init__(self, simulator, name, inq, outqs, options):
        self.simulator = simulator
        self.name = name
        self.inq = inq
        self.outqs = outqs
        self.generator = self.run()

    def best_queue(self, size):
        if len(self.outqs) > 1:
            valid_queues = [q for q in self.outqs if q.can_get(size)]
            if valid_queues:
                return max(valid_queues, key = lambda x: x.spare())
            else:
                return None
        else:
            return self.outqs[0]

    def run(self):
        yield
        while True:
            self.inq.pop(self)
            item = yield
            if isinstance(item, EndQItem):
                if self.simulator.count_running_workers(self.name) == 1:
                    # We are the last worker from the set
                    for q in self.outqs:
                        q.stop()
                break
            print(self.name, self.simulator.time, item.total_time())
            for child in item.children:
                size = child.size

                yield child.parent_get

                while True:
                    outq = self.best_queue(size)
                    if outq is not None:
                        break
                    for q in self.outqs:
                        q.watch(self)
                    yield
                    for q in self.outqs:
                        q.unwatch(self)

                outq.get(self, size)
                child.alloc = yield

                yield child.parent_push
                outq.push(child, child.alloc)
            if item.finish > 0:
                yield item.finish
            if hasattr(item, 'alloc'):
                self.inq.done(item.alloc)

class Simulator(object):
    def __init__(self):
        self.workers = []
        self.wakeup_queue = []
        self.time = 0.0
        self.running = set()

    def add_worker(self, worker):
        self.workers.append(worker)
        worker.generator.send(None)
        self.wakeup(worker)

    def wakeup(self, worker, time = None, value = None):
        if time is None:
            time = self.time
        assert time >= self.time
        for (t, w, v) in self.wakeup_queue:
            assert w != worker
        heapq.heappush(self.wakeup_queue, (time, worker, value))

    def count_running_workers(self, name):
        ans = 0
        for w in self.running:
            if w.name == name:
                ans += 1
        return ans

    def run(self):
        self.time = 0.0
        self.running = set(self.workers)
        while self.wakeup_queue:
            (self.time, worker, value) = heapq.heappop(self.wakeup_queue)
            assert worker in self.running
            try:
                compute_time = worker.generator.send(value)
                if compute_time is not None:
                    assert compute_time >= 0
                    self.wakeup(worker, self.time + compute_time)
            except StopIteration:
                self.running.remove(worker)
        if self.running:
            print("Workers still running: possible deadlock", file = sys.stderr)
            for w in self.running:
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

    gpus = options.gpus

    if options.infinite:
        big = 10**30
        coarse_cap = big
        fine_cap = big
        mesher_cap = big
    else:
        coarse_cap = options.coarse_cap * 1024 * 1024
        fine_cap = 2
        mesher_cap = options.mesher_cap * 1024 * 1024

    all_queue = SimQueue(simulator, 1)
    coarse_queue = SimQueue(simulator, coarse_cap)
    fine_queues = [SimQueue(simulator, fine_cap, inorder = False) for i in range(gpus)]
    mesh_queue = SimQueue(simulator, mesher_cap)

    simulator.add_worker(SimWorker(simulator, 'coarse', all_queue, [coarse_queue], options))
    simulator.add_worker(SimWorker(simulator, 'fine', coarse_queue, fine_queues, options))
    for i in range(gpus):
        simulator.add_worker(SimWorker(simulator, 'device', fine_queues[i], [mesh_queue], options))
    simulator.add_worker(SimWorker(simulator, 'mesher', mesh_queue, [], options))

    all_queue.push(root, None)
    all_queue.stop()
    simulator.run()
    print(simulator.time)

def main():
    parser = OptionParser()
    parser.add_option('--infinite', action = 'store_true')
    parser.add_option('--gpus', type = 'int', default = 1)
    parser.add_option('--coarse-cap', type = 'int', metavar = 'MiB', default = 512)
    parser.add_option('--bucket-cap', type = 'int', metavar = 'MiB', default = 128)
    parser.add_option('--mesher-cap', type = 'int', metavar = 'MiB', default = 512)
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
