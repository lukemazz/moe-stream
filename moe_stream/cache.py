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


class DecodeArena:
    """Per-layer stacked expert buffers for single-dispatch decode (gather_qmm).

    Each layer owns preallocated tensors [H, ...] per weight name plus an
    expert->slot LRU map. Inserting an expert copies its tensors into a slot
    once; from then on decode addresses it by slot index, so the k active
    experts of a layer run as ONE gather_qmm per projection instead of k
    quantized_matmul launches. All mutation happens on the decode thread; the
    slice-assign follows the KV-cache in-place update pattern (scatter with
    buffer donation), so steady-state inserts do not copy the whole buffer.
    """

    def __init__(self, n_layers: int, total_bytes: int, n_experts: int = 0):
        self.n_layers = n_layers
        self.total_bytes = total_bytes
        self.n_experts = n_experts
        self.h = 0  # slots per layer, sized on first insert from expert nbytes
        self.layers = [None] * n_layers  # layer -> {name: stacked buffer}
        self.maps = [OrderedDict() for _ in range(n_layers)]  # expert -> slot
        # specchio GPU della mappa esperto->slot (-1 = assente), per la bozza
        # auto-speculativa che seleziona gli esperti senza readback CPU
        self.slot_tables = [None] * n_layers
        self.hits = 0
        self.inserts = 0
        self.layer_inserts = [0] * n_layers  # probe: churn per layer

    def lookup(self, layer: int, expert: int):
        m = self.maps[layer]
        slot = m.get(expert)
        if slot is not None:
            m.move_to_end(expert)
            self.hits += 1
        return slot

    def contains(self, layer: int, expert: int) -> bool:
        return expert in self.maps[layer]

    def insert(self, layer: int, expert: int, arrays: dict) -> int:
        import mlx.core as mx
        if self.h == 0:
            self.h = max(8, int(self.total_bytes
                                // (self.n_layers * entry_bytes(arrays))))
        bufs = self.layers[layer]
        if bufs is None:
            bufs = {name: mx.zeros((self.h, *a.shape), dtype=a.dtype)
                    for name, a in arrays.items()}
            self.layers[layer] = bufs
        m = self.maps[layer]
        if len(m) < self.h:
            slot = len(m)
            evicted = None
        else:
            evicted, slot = m.popitem(last=False)  # reuse the LRU slot
        for name, a in arrays.items():
            bufs[name][slot] = a
        m[expert] = slot
        if self.n_experts:
            st = self.slot_tables[layer]
            if st is None:
                st = mx.full((self.n_experts,), -1, dtype=mx.int32)
                self.slot_tables[layer] = st
            if evicted is not None:
                st[evicted] = -1
            st[expert] = slot
        self.inserts += 1
        self.layer_inserts[layer] += 1
        return slot

    def stats(self) -> dict:
        return {"arena_hits": self.hits, "arena_inserts": self.inserts,
                "arena_slots_per_layer": self.h,
                "arena_layer_inserts": list(self.layer_inserts)}


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

    def remove(self, key):
        """Drop an entry from whatever tier holds it (no hit/miss counted).
        Used when an expert moves to the decode arena, so it is not held twice."""
        with self.lock:
            for tier in (LRU, PREFETCH, FILLER):
                d = self.tiers[tier]
                if key in d:
                    self.used[tier] -= entry_bytes(d.pop(key))
                    return

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
