"""Async SSD IO: priority-queue thread pool for prefetch, plus the filler
fill loop (spec sections 6.5 and 7.3). Prefetch requests carry priority > 0;
filler requests use priority 0 and are naturally served last.
"""

import itertools
import queue
import random
import threading

from . import cache as cache_mod
from .shards import read_shard, shard_path


class IOPool:
    def __init__(self, shard_root, cache: cache_mod.ExpertCache, n_threads: int = 4):
        self.shard_root = shard_root
        self.cache = cache
        self.q = queue.PriorityQueue()
        self.in_flight = set()
        self.lock = threading.Lock()
        self._tie = itertools.count()
        self._stop = threading.Event()
        self.threads = [
            threading.Thread(target=self._worker, daemon=True, name=f"io-{i}")
            for i in range(n_threads)
        ]
        for t in self.threads:
            t.start()

    def enqueue(self, key, priority: float, tier: str):
        """key = (layer, expert_id). Higher priority is served first."""
        with self.lock:
            if key in self.in_flight:
                return
            self.in_flight.add(key)
        if not self.cache.contains(key):
            self.q.put((-priority, next(self._tie), key, tier))
        else:
            with self.lock:
                self.in_flight.discard(key)

    def load_sync(self, key):
        """Blocking load used on cache miss in the hot path."""
        _, arrays = read_shard(shard_path(self.shard_root, *key))
        return arrays

    def is_saturated(self) -> bool:
        return self.q.qsize() > 2 * len(self.threads)

    def wait_idle(self, key) -> bool:
        with self.lock:
            return key in self.in_flight

    def _worker(self):
        while not self._stop.is_set():
            try:
                _, _, key, tier = self.q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if not self.cache.contains(key):
                    arrays = self.load_sync(key)
                    self.cache.put(key, arrays, tier=tier)
            except FileNotFoundError:
                pass
            finally:
                with self.lock:
                    self.in_flight.discard(key)
                self.q.task_done()

    def stop(self):
        self._stop.set()


class FillerLoop:
    """Low-priority background thread that opportunistically warms the filler
    cache with randomly selected experts (spec section 7.3)."""

    MIN_FREE = 64 * 1024 * 1024  # only fill when at least this much budget free

    def __init__(self, all_keys, io: IOPool, cache: cache_mod.ExpertCache,
                 batch: int = 8):
        self.all_keys = list(all_keys)
        self.io = io
        self.cache = cache
        self.batch = batch
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name="filler")

    def start(self):
        self.thread.start()

    def stop(self):
        self._stop.set()

    def _run(self):
        while not self._stop.is_set():
            if (self.cache.filler_free_bytes() > self.MIN_FREE
                    and not self.io.is_saturated()):
                cached = self.cache.cached_keys()
                pool = [k for k in self.all_keys if k not in cached]
                if pool:
                    for key in random.sample(pool, min(self.batch, len(pool))):
                        self.io.enqueue(key, priority=0.0, tier=cache_mod.FILLER)
            self._stop.wait(0.05)
