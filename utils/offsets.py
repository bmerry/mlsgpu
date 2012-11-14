#!/usr/bin/env python
from __future__ import print_function, division
import sys
import random

mib = 1024 * 1024
gib = 1024 * 1024 * 1024

class Page(object):
    def __init__(self, start, time):
        self.start = start
        self.last_used = time

class PageTable(object):
    def __init__(self, page_blocks, slots, block_size):
        self.slots = slots
        self.page_size = block_size * page_blocks
        self.pages = []
        self.hits = 0
        self.misses = 0
        self.loaded = 0
        self.time = 0

    def access(self, addr):
        self.time += 1
        for p in self.pages:
            if p.start <= addr and addr < p.start + self.page_size:
                self.hits += 1
                p.last_used = self.time
                return
        if len(self.pages) == self.slots:
            oldest = None
            for p in self.pages:
                if oldest is None or p.last_used < oldest.last_used:
                    oldest = p
            # Enable for random paging strategy:
            # oldest = self.pages[random.randint(0, self.slots - 1)]
            self.pages.remove(oldest)
        start = addr // self.page_size * self.page_size
        self.pages.append(Page(start, self.time))
        self.misses += 1
        self.loaded += self.page_size

    def access_range(self, start, end):
        first = start // self.page_size
        last = (end + self.page_size - 1) // self.page_size
        for i in range(first, last):
            self.access(i * self.page_size)

class Access(object):
    def __init__(self, chunk, first_vertex, last_vertex, first_triangle, last_triangle):
        self.chunk = chunk
        self.first_vertex = first_vertex
        self.last_vertex = last_vertex
        self.first_triangle = first_triangle
        self.last_triangle = last_triangle

    def vertices(self):
        return self.last_vertex - self.first_vertex

    def triangles(self):
        return self.last_triangle - self.first_triangle

    def size(self):
        return 12 * (self.vertices() + self.triangles())

    def adjust(self, nfv, nft):
        dv = nfv - self.first_vertex
        self.first_vertex += dv
        self.last_vertex += dv
        dt = nft - self.first_triangle
        self.first_triangle += dt
        self.last_triangle += dt

    def __str__(self):
        return "{}: {}-{}   {}-{}".format(self.chunk, self.first_vertex, self.last_vertex, self.first_triangle, self.last_triangle)

    def __cmp__(self, other):
        if self.chunk != other.chunk:
            return self.chunk - other.chunk
        else:
            return self.first_vertex - other.first_vertex

def flush(accesses, chunk, va, ta):
    for v, t in zip(va, ta):
        accesses.append(Access(chunk, v[0], v[1], t[0], t[1]))

def load_accesses(f):
    chunk = 0
    va = []
    ta = []
    accesses = []
    for line in f:
        fields = line.split()
        if fields[0] == 'CHUNK':
            flush(accesses, chunk, va, ta)
            va = []
            ta = []
            chunk = int(fields[1])
        elif fields[0] == 'VERTICES':
            first = int(fields[1])
            last = int(fields[2])
            va.append((first, last))
        elif fields[0] == 'TRIANGLES':
            first = int(fields[1])
            last = int(fields[2])
            ta.append((first, last))
    flush(accesses, chunk, va, ta)
    return accesses

class Emitter(object):
    def __init__(self):
        self.vpos = 0
        self.tpos = 0
        self.out = []
        self.last_chunk = -1

    def emit(self, a):
        a.adjust(self.vpos, self.tpos)
        self.out.append(a)
        self.vpos += a.vertices()
        self.tpos += a.triangles()
        if self.last_chunk != a.chunk:
            # print("Chunk:", a.chunk)
            self.last_chunk = a.chunk

class Chunk(object):
    def __init__(self, chunk):
        self.accesses = []
        self.chunk = chunk
        self.last_used = -1

    def append(self, access, timestamp):
        assert access.chunk == self.chunk
        self.accesses.append(access)
        self.last_used = timestamp

    def __cmp__(self, other):
        return self.chunk - other.chunk

class ChunkCache(object):
    def __init__(self):
        self.time = 0
        self.chunks = {}

    def append(self, access):
        self.time += 1
        if access.chunk not in self.chunks:
            self.chunks[access.chunk] = Chunk(access.chunk)
        self.chunks[access.chunk].append(access, self.time)

    def pop(self):
        best = None
        # for c in sorted(self.chunks.values()):
        #     if best is None or c.last_used < best.last_used:
        #         best = c
        best = min(self.chunks.values())
        del self.chunks[best.chunk]
        return best

def reorder_buffer(accesses, capacity):
    ordered = sorted(accesses, key = lambda x: x.first_vertex)
    e = Emitter()
    cache = ChunkCache()
    qsize = 0
    for a in ordered:
        while qsize > 0 and a.size() + qsize > capacity:
            while qsize > 0:
                c = cache.pop()
                for dump in c.accesses:
                    e.emit(dump)
                    qsize -= dump.size()
        cache.append(a)
        qsize += a.size()
    while qsize > 0:
        c = cache.pop()
        for dump in c.accesses:
            e.emit(dump)
            qsize -= dump.size()
    return sorted(e.out)

def main():
    vertex_table = PageTable(1, 16, 8 * 1024 * 1024)
    triangle_table = PageTable(1, 32, 8 * 1024 * 1024)
    with open(sys.argv[1], 'r') as f:
        accesses = load_accesses(f)

    accesses = reorder_buffer(accesses, 1024 * mib)

    for a in accesses:
        vertex_table.access_range(a.first_vertex * 12, a.last_vertex * 12)
        triangle_table.access_range(a.first_triangle * 12, a.last_triangle * 12)
    print('Vertex traffic (GiB):', vertex_table.loaded / gib)
    print('Triangle traffic (GiB):', triangle_table.loaded / gib)
    print('Total traffic (GiB):', (vertex_table.loaded + triangle_table.loaded) / gib)

if __name__ == '__main__':
    main()
