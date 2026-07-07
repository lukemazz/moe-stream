"""MLX model wrapper: replaces SwitchGLU in each MoE layer with a streamed
version that pulls expert weights through the three-tier cache and triggers
predictive prefetch after each router decision (spec sections 3, 6, 8).
"""

import json
import os
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .cache import DecodeArena, ExpertCache, LRU, PREFETCH
from .io_pool import FillerLoop, IOPool
from .predictor import Predictor


def _swiglu(gate, up):
    return nn.silu(gate) * up


class StreamedSwitchGLU(nn.Module):
    """Drop-in replacement for switch_layers.SwitchGLU on the decode path.

    Expert weights are fetched per call from the cache manager; misses are
    loaded synchronously from SSD. After computing, the predictor enqueues
    prefetch for the next layer(s).
    """

    def __init__(self, layer_idx: int, runtime: "StreamRuntime", bits: int, group_size: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.rt = runtime
        self.bits = bits
        self.group_size = group_size

    def _expert_slot(self, expert_id: int) -> int:
        """Decode path: resolve an expert to its arena slot, inserting it on
        first touch (the one-time copy that per-token stacking would pay
        every token). The dict-cache entry is dropped after the move so the
        expert is not held in RAM twice."""
        ar = self.rt.arena
        slot = ar.lookup(self.layer_idx, expert_id)
        if slot is None:
            arrays = self._expert_arrays(expert_id)
            slot = ar.insert(self.layer_idx, expert_id, arrays)
            self.rt.cache.remove((self.layer_idx, expert_id))
        return slot

    def _expert_arrays(self, expert_id: int):
        key = (self.layer_idx, expert_id)
        ar = self.rt.arena
        if ar is not None:
            slot = ar.lookup(self.layer_idx, expert_id)
            if slot is not None:  # lazy row views, no copy
                bufs = ar.layers[self.layer_idx]
                return {name: bufs[name][slot] for name in bufs}
        arrays = self.rt.cache.get(key)
        if arrays is None:
            # if a prefetch for this key is in flight, wait briefly for it
            deadline = time.monotonic() + 0.05
            while self.rt.io.wait_idle(key) and time.monotonic() < deadline:
                arrays = self.rt.cache.get(key)
                if arrays is not None:
                    return arrays
                time.sleep(0.0005)
            arrays = self.rt.cache.get(key)
            if arrays is None:
                self.rt.sync_loads += 1
                prev = self.rt._last_active.get(self.layer_idx - 1)
                if self.layer_idx == 0 or prev is None:
                    self.rt.pred_na += 1
                elif expert_id in self.rt.predictor.predict(
                        self.layer_idx - 1, prev, self.rt.prefetch_width):
                    self.rt.pred_hit += 1
                else:
                    self.rt.pred_miss += 1
                _t = time.monotonic()
                arrays = self.rt.io.load_sync(key)
                self.rt.sync_secs += time.monotonic() - _t
                self.rt.cache.put(key, arrays, tier=LRU)
        return arrays

    def _qmm(self, x, arrays, part):
        return mx.quantized_matmul(
            x,
            arrays[f"{part}.weight"],
            scales=arrays[f"{part}.scales"],
            biases=arrays[f"{part}.biases"],
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
        )

    def _gather_qmm(self, x, stacked, part, rhs_indices):
        return mx.gather_qmm(
            x,
            stacked[f"{part}.weight"],
            stacked[f"{part}.scales"],
            stacked[f"{part}.biases"],
            rhs_indices=rhs_indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
        )

    def _draft_gpu(self, xt, indices, lead, k, D):
        """Forward bozza SENZA readback: fino a draft_n esperti tra i residenti
        in arena, scelti per punteggio del router interamente su GPU tramite la
        slot-table. Può essere 'sbagliato' (esperti assenti -> contributo zero,
        layer mai inseriti -> solo shared expert): è la bozza — la correttezza
        la garantisce il forward di verifica sul percorso pieno."""
        ar = self.rt.arena
        st = ar.slot_tables[self.layer_idx] if ar is not None else None
        if st is None:  # layer senza arena (es. 0-2 anti-thrash)
            return mx.zeros((*lead, k, D), dtype=xt.dtype)
        bufs = ar.layers[self.layer_idx]
        n = min(self.rt.draft_n, k)
        idxf = indices.reshape(-1)                  # [k]
        slots = st[idxf]                            # [k], -1 = assente
        g = self.rt.gates[self.layer_idx](xt)[0]    # [n_experts] logits
        s = mx.where(slots >= 0, g[idxf].astype(mx.float32),
                     mx.full((k,), -1e30))
        keep = mx.argsort(-s)[:n]                   # posizioni dei migliori n
        sel = slots[keep]                           # [n] slot scelti (o -1)
        ridx = mx.maximum(sel, 0).astype(mx.uint32).reshape(1, n)
        xe = xt.reshape(1, 1, 1, D)
        gu = self._gather_qmm(xe, bufs, "gateup_proj", ridx)
        h = gu.shape[-1] // 2
        y = self._gather_qmm(_swiglu(gu[..., :h], gu[..., h:]),
                             bufs, "down_proj", ridx).squeeze(-2)  # [1, n, D]
        y = y * (sel >= 0).astype(y.dtype).reshape(1, n, 1)
        onehot = (mx.arange(k).reshape(k, 1) == keep.reshape(1, n))
        out = onehot.astype(y.dtype) @ y[0]         # [k, D] nei loro slot
        return out.reshape(*lead, k, D)

    def __call__(self, x, indices) -> mx.array:
        # x: [..., D], indices: [..., k]
        *lead, D = x.shape
        k = indices.shape[-1]
        xt = x.reshape(-1, D)
        if self.rt.draft_gpu and xt.shape[0] == 1:
            return self._draft_gpu(xt, indices, lead, k, D)
        idx = np.array(indices.reshape(-1, k))  # forces eval of router output
        T = xt.shape[0]
        self.rt._last_active[self.layer_idx] = np.unique(idx)  # for pred probe

        # enqueue prefetch for the next layers BEFORE computing this layer's
        # experts, so SSD IO overlaps with the GPU work below
        self.rt.prefetch_hook(self.layer_idx, xt, idx)

        # Hybrid dispatch. Decode (few tokens): loop over the k experts with
        # plain quantized_matmul — stacking weights would cost more in copies
        # than it saves in kernel launches. Prefill (many tokens): stack the
        # active experts once and run three batched gather_qmm kernels, where
        # the copy is amortized over all tokens.
        if T == 1 and self.rt.draft_top1:
            # DRAFT auto-speculativa: solo il miglior esperto (preferendo i
            # residenti in arena), col suo peso vero — l'output va nel suo
            # slot e gli altri restano zero, così il caller ricostruisce
            # esattamente w1*y1 + shared_expert. Correttezza NON richiesta:
            # è la bozza; il forward di verifica usa il percorso pieno.
            ar = self.rt.arena
            sc = np.array(
                self.rt.gates[self.layer_idx](xt).astype(mx.float32))[0]
            cand = [int(e) for e in idx[0]]
            pool = [e for e in cand
                    if ar is not None and ar.contains(self.layer_idx, e)] or cand
            chosen = set(sorted(pool, key=lambda q: sc[q],
                                reverse=True)[:self.rt.draft_n])
            outs = {}
            for e in chosen:
                arrays = self._expert_arrays(e)
                gu = self._qmm(xt, arrays, "gateup_proj")
                h = gu.shape[-1] // 2
                outs[e] = self._qmm(_swiglu(gu[..., :h], gu[..., h:]),
                                    arrays, "down_proj")
            z = mx.zeros_like(next(iter(outs.values())))
            ys = [outs.get(c, z) for c in cand]
            return mx.stack(ys, axis=1).reshape(*lead, k, D)

        if (self.rt.arena is not None
                and self.layer_idx not in self.rt.arena_skip
                and T * k <= 4096):
            # decode E verifica speculativa (T piccolo): gli esperti attivi come
            # UN gather_qmm per proiezione, indirizzati per slot di arena.
            # Guardia LRU: tutti gli esperti unici devono poter risiedere
            # simultaneamente negli slot senza sfrattarsi a vicenda.
            uniq, inv = np.unique(idx, return_inverse=True)
            ar = self.rt.arena
            if ar.h == 0:
                self._expert_slot(int(uniq[0]))  # dimensiona l'arena (bootstrap)
            if T == 1 or len(uniq) <= ar.h // 2:
                slots = np.array([self._expert_slot(int(e)) for e in uniq])
                ridx = mx.array(slots[inv].reshape(T, k).astype(np.uint32))
                bufs = ar.layers[self.layer_idx]
                xe = xt.reshape(T, 1, 1, D)
                gu = self._gather_qmm(xe, bufs, "gateup_proj", ridx)
                h = gu.shape[-1] // 2
                y = self._gather_qmm(_swiglu(gu[..., :h], gu[..., h:]),
                                     bufs, "down_proj", ridx)
                return y.squeeze(-2).reshape(*lead, k, D)

        if T == 1:
            # single token: router experts are distinct, no scatter needed
            ys = []
            for e in idx[0]:
                arrays = self._expert_arrays(int(e))
                gu = self._qmm(xt, arrays, "gateup_proj")
                h = gu.shape[-1] // 2
                ys.append(self._qmm(_swiglu(gu[..., :h], gu[..., h:]),
                                    arrays, "down_proj"))
            return mx.stack(ys, axis=1).reshape(*lead, k, D)

        if T * k <= 4096:
            out = mx.zeros((T, k, D), dtype=x.dtype)
            for e in np.unique(idx):
                rows, slots = np.nonzero(idx == e)
                arrays = self._expert_arrays(int(e))
                xe = xt[mx.array(rows)]
                gu = self._qmm(xe, arrays, "gateup_proj")
                h = gu.shape[-1] // 2
                ye = self._qmm(_swiglu(gu[..., :h], gu[..., h:]),
                               arrays, "down_proj")
                out[mx.array(rows), mx.array(slots)] = ye
            return out.reshape(*lead, k, D)

        uniq, inv = np.unique(idx, return_inverse=True)
        experts = [self._expert_arrays(int(e)) for e in uniq]
        stacked = {name: mx.stack([e[name] for e in experts])
                   for name in experts[0]}
        ridx = mx.array(inv.reshape(T, k).astype(np.uint32))

        xe = xt.reshape(T, 1, 1, D)
        gu = self._gather_qmm(xe, stacked, "gateup_proj", ridx)
        h = gu.shape[-1] // 2
        y = self._gather_qmm(_swiglu(gu[..., :h], gu[..., h:]),
                             stacked, "down_proj", ridx)

        return y.squeeze(-2).reshape(*lead, k, D)


class _EmbedHook(nn.Module):
    """Cross-token prefetch: as soon as the freshly sampled token is embedded,
    predict the experts for the first layers by applying their routers to the
    embedding (the residual stream at the start of the network is dominated by
    it), so SSD loads overlap with layer-0 attention instead of stalling the
    first MoE layers cold."""

    def __init__(self, embed, rt: "StreamRuntime"):
        super().__init__()
        self.embed = embed
        self.rt = rt

    def __call__(self, x):
        y = self.embed(x)
        if y.size // y.shape[-1] <= 8:  # decode only; prefill warms itself
            rt = self.rt
            if rt.draft_gpu:
                return y  # bozza: niente lookahead/probe, zero Python
            if rt.arena is not None:
                # PROBE temporaneo (design no-readback): un token che ha
                # causato >=1 insert in arena sarebbe stato un token-miss
                # (da ricalcolare) nel decode ottimistico.
                ins = rt.arena.inserts
                if rt._probe_last_ins is not None and ins > rt._probe_last_ins:
                    rt.miss_tokens += 1
                rt._probe_last_ins = ins
                rt.decode_tokens += 1
            self.rt.lookahead_from(-1, y.reshape(-1, y.shape[-1]))
        return y

    def as_linear(self, x):
        return self.embed.as_linear(x)


class StreamRuntime:
    """Shared state: cache, IO pool, predictor, filler, activation recorder."""

    def __init__(self, shard_root: Path, n_layers: int, n_experts: int,
                 lru_bytes: int, prefetch_bytes: int, filler_bytes: int,
                 table_path=None, prefetch_depth: int = 2, prefetch_width: int = 8,
                 io_threads: int = 4):
        self.cache = ExpertCache(lru_bytes, prefetch_bytes, filler_bytes)
        self.io = IOPool(shard_root, self.cache, n_threads=io_threads)
        self.predictor = Predictor(n_layers, n_experts, table_path)
        self.prefetch_depth = prefetch_depth
        self.prefetch_width = prefetch_width
        self.n_layers = n_layers
        self.n_experts = n_experts
        self.sync_loads = 0
        # sync-load predictability probe: was a synchronously-loaded expert one
        # the transition-table predictor would have named from the prev layer's
        # active set? pred_hit = predictable (timing/eviction gap); pred_miss =
        # predictor didn't know it; pred_na = layer 0, no prev MoE layer.
        self.pred_hit = self.pred_miss = self.pred_na = 0
        self.sync_secs = 0.0  # wall time blocked on synchronous SSD reads
        self._last_active = {}  # moe layer_idx -> active expert ids this step
        self.arena = None  # DecodeArena, set by load_streamed_model
        # anti-thrash: layer il cui working set eccede gli slot (misurato:
        # 0-2 fanno churn oltre il ritmo dei token) girano sul percorso dict,
        # risparmiando insert-copy inutili (+10-12% misurato sui turni caldi).
        # Override con MOE_ARENA_SKIP (es. "" per disattivare, "0,1,2,3" ...).
        self.arena_skip = frozenset(
            int(x) for x in os.environ.get("MOE_ARENA_SKIP", "0,1,2").split(",")
            if x.strip())
        self.draft_top1 = False  # bozza CPU (solo per acceptance_bench)
        self.draft_gpu = False  # bozza GPU no-readback (loop auto-speculativo)
        self.draft_n = int(os.environ.get("MOE_DRAFT_N", "1"))
        self.miss_tokens = 0  # probe: token con insert arena (no-readback)
        self.decode_tokens = 0
        self._probe_last_ins = None
        self.recorder = None  # set by profiler: fn(layer, np_indices)
        self.gates = None  # routers of all layers, set by load_streamed_model
        self.use_lookahead = True
        # deferred lookahead pipeline: scores are scheduled with async_eval
        # and read back (numpy) one layer later, when they are already
        # computed, keeping graph syncs off the decode critical path
        self._la_pending: list[tuple[int, mx.array]] = []
        all_keys = [(l, e) for l in range(n_layers) for e in range(n_experts)]
        self.filler = FillerLoop(all_keys, self.io, self.cache)
        if filler_bytes > 0:
            self.filler.start()

    def prefetch_hook(self, layer: int, xt: mx.array, idx: np.ndarray):
        if self.recorder is not None:
            self.recorder(layer, idx)
        if self.use_lookahead and self.gates is not None:
            self.lookahead_from(layer, xt)
        else:
            self._table_prefetch(layer, idx)

    def lookahead_from(self, layer: int, xt: mx.array):
        if self.use_lookahead and self.gates is not None:
            self._lookahead(layer, xt)

    def _lookahead(self, layer: int, xt: mx.array):
        """Router lookahead (pre-gating): apply the routers of the next layers
        to the current hidden state. The residual stream changes slowly across
        adjacent layers, so this predicts upcoming experts far better than
        static co-occurrence statistics. The top-k index arrays are scheduled
        asynchronously and consumed on the next call, when the GPU has already
        produced them — the numpy readback then costs ~nothing."""
        # 1. drain predictions scheduled on the previous layer
        for nxt, d, preds_arr in self._la_pending:
            prio = 10.0 / d
            for rank, e in enumerate(np.array(preds_arr).tolist()):
                if self.arena is not None and self.arena.contains(nxt, e):
                    continue  # already resident in the decode arena
                self.io.enqueue((nxt, e), priority=prio - 0.01 * rank,
                                tier=PREFETCH)
        self._la_pending.clear()

        # 2. schedule this layer's predictions without syncing
        w = self.prefetch_width
        depth = self.prefetch_depth if layer >= 0 else 2
        scheduled = []
        for d in range(1, depth + 1):
            nxt = layer + d
            if nxt >= self.n_layers or self.gates[nxt] is None:
                break
            # raw logits: softmax is monotonic, ranking does not need it
            scores = self.gates[nxt](xt).sum(axis=0)
            preds = mx.argpartition(scores, kth=-w)[-w:]
            scheduled.append((nxt, d, preds))
        if scheduled:
            mx.async_eval([p for _, _, p in scheduled])
            self._la_pending = scheduled

    def _table_prefetch(self, layer: int, idx: np.ndarray):
        active = np.unique(idx).tolist()
        # chain predictions for depth layers ahead
        for d in range(1, self.prefetch_depth + 1):
            preds = self.predictor.predict(layer + d - 1, active, self.prefetch_width)
            if not preds:
                break
            prio = 10.0 / d
            for rank, e in enumerate(preds):
                if self.arena is not None and self.arena.contains(layer + d, e):
                    continue
                self.io.enqueue((layer + d, e), priority=prio - 0.01 * rank,
                                tier=PREFETCH)
            active = preds

    def stats(self):
        s = self.cache.stats()
        if self.arena is not None:
            s.update(self.arena.stats())
            s["miss_tokens"] = self.miss_tokens
            s["decode_tokens"] = self.decode_tokens
        s["sync_loads"] = self.sync_loads
        s["sync_predictable"] = self.pred_hit
        s["sync_unpredictable"] = self.pred_miss
        s["sync_layer0"] = self.pred_na
        s["sync_secs"] = round(self.sync_secs, 2)
        return s

    def stop(self):
        self.filler.stop()
        self.io.stop()


def load_streamed_model(model_dir: Path, shard_dir: Path, *,
                        lru_bytes: int, prefetch_bytes: int, filler_bytes: int,
                        table_path=None, prefetch_depth: int = 2,
                        prefetch_width: int = 8, io_threads: int = 4):
    """Load fixed parts into RAM, swap every MoE switch_mlp for the streamed
    version. Expert weights from the safetensors are never materialized."""
    from mlx_lm.utils import load_model

    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / "config.json").read_text())
    quant = cfg.get("quantization") or cfg.get("quantization_config")
    text_cfg = cfg.get("text_config", cfg)
    n_layers = text_cfg["num_hidden_layers"]
    n_experts = text_cfg["num_experts"]

    model, _ = load_model(model_dir, lazy=True)

    # The decode arena takes most of the LRU budget: the dict cache then only
    # stages prefetch/prefill entries on their way in. MOE_NO_ARENA=1 restores
    # the per-expert loop with the full budget on the dict LRU (for A/B).
    arena_bytes = 0 if os.environ.get("MOE_NO_ARENA") else int(lru_bytes * 0.85)

    rt = StreamRuntime(Path(shard_dir), n_layers, n_experts,
                       lru_bytes - arena_bytes, prefetch_bytes, filler_bytes,
                       table_path=table_path, prefetch_depth=prefetch_depth,
                       prefetch_width=prefetch_width, io_threads=io_threads)
    if arena_bytes:
        rt.arena = DecodeArena(n_layers, arena_bytes, n_experts)

    inner = model.language_model.model if hasattr(model, "language_model") else model.model
    n_swapped = 0
    gates = []
    for l, layer in enumerate(inner.layers):
        mlp = layer.mlp
        if hasattr(mlp, "switch_mlp"):
            mlp.switch_mlp = StreamedSwitchGLU(l, rt, quant["bits"], quant["group_size"])
            n_swapped += 1
            gates.append(mlp.gate)
        else:
            gates.append(None)
    assert n_swapped == n_layers, f"swapped {n_swapped}/{n_layers} MoE layers"
    rt.gates = gates
    inner.embed_tokens = _EmbedHook(inner.embed_tokens, rt)

    mx.eval(model.parameters())  # loads only the fixed (non-expert) weights
    return model, rt
