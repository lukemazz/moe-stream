"""Three-tier expert cache (spec sections 5 and 7).

Tiers and strict eviction order (evicted first -> last):
  filler (random warm fill)  ->  prefetch staging  ->  LRU.
Lookup order on a request: LRU -> prefetch -> filler -> SSD (sync, counted as miss).
A hit in prefetch/filler promotes the entry to the LRU tier.
"""

import threading
from collections import OrderedDict

LRU, PREFETCH, FILLER = "lru", "prefetch", "filler"


def entry_bytes(arrays: dict) -> int:
    return sum(a.nbytes for a in arrays.values())


class ExpertCache:
    def __init__(self, lru_bytes: int, prefetch_bytes: int, filler_bytes: int):
        self.lock = threading.Lock()
        self.budget = {LRU: lru_bytes, PREFETCH: prefetch_bytes, FILLER: filler_bytes}
        self.used = {LRU: 0, PREFETCH: 0, FILLER: 0}
        self.tiers = {LRU: OrderedDict(), PREFETCH: OrderedDict(), FILLER: OrderedDict()}
        # stats
        self.hits = {LRU: 0, PREFETCH: 0, FILLER: 0}
        self.misses = 0

    # ------------------------------------------------------------ lookup

    def get(self, key):
        """Returns the arrays dict or None. Promotes prefetch/filler hits to LRU."""
        with self.lock:
            for tier in (LRU, PREFETCH, FILLER):
                d = self.tiers[tier]
                if key in d:
                    self.hits[tier] += 1
                    arrays = d[key]
                    if tier == LRU:
                        d.move_to_end(key)
                    else:
                        nb = entry_bytes(arrays)
                        del d[key]
                        self.used[tier] -= nb
                        self._insert(LRU, key, arrays, nb)
                    return arrays
            self.misses += 1
            return None

    def contains(self, key) -> bool:
        with self.lock:
            return any(key in self.tiers[t] for t in (LRU, PREFETCH, FILLER))

    def put(self, key, arrays, tier=LRU):
        nb = entry_bytes(arrays)
        with self.lock:
            if any(key in self.tiers[t] for t in (LRU, PREFETCH, FILLER)):
                return
            self._insert(tier, key, arrays, nb)

    # ------------------------------------------------------------ internals
    # caller must hold self.lock

    def _insert(self, tier, key, arrays, nb):
        if nb > self.budget[tier]:
            return  # would never fit; drop
        self._make_room(tier, nb)
        self.tiers[tier][key] = arrays
        self.used[tier] += nb

    def _make_room(self, tier, nb):
        """Evict to fit nb bytes into tier. Lower-priority tiers are robbed first."""
        donors = {LRU: (FILLER, PREFETCH, LRU), PREFETCH: (FILLER, PREFETCH), FILLER: (FILLER,)}[tier]
        while self.used[tier] + nb > self.budget[tier]:
            # tier over its own budget: evict its own oldest entries
            d = self.tiers[tier]
            if not d:
                break
            _, old = d.popitem(last=False)
            self.used[tier] -= entry_bytes(old)
        # global memory pressure is handled by per-tier budgets, but if a
        # higher tier was granted extra space, steal it back from donors
        for donor in donors:
            if self.used[tier] + nb <= self.budget[tier]:
                break
            d = self.tiers[donor]
            while d and self.used[tier] + nb > self.budget[tier]:
                _, old = d.popitem(last=False)
                self.used[donor] -= entry_bytes(old)

    # ------------------------------------------------------------ stats

    def filler_free_bytes(self) -> int:
        with self.lock:
            return self.budget[FILLER] - self.used[FILLER]

    def cached_keys(self) -> set:
        with self.lock:
            ks = set()
            for t in (LRU, PREFETCH, FILLER):
                ks |= set(self.tiers[t])
            return ks

    def stats(self) -> dict:
        with self.lock:
            total = sum(self.hits.values()) + self.misses
            return {
                "hits": dict(self.hits),
                "misses": self.misses,
                "hit_rate": (sum(self.hits.values()) / total) if total else 0.0,
                "used_mb": {t: round(self.used[t] / 1e6, 1) for t in self.used},
            }
