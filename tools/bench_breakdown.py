"""Measure where decode time goes: MoE expert compute vs rest (attention,
norms, sampling) vs sync SSD loads. Run: python -m tools.bench_breakdown"""

import sys
import time
from pathlib import Path

import mlx.core as mx

from moe_stream import model as mst
from moe_stream.generate import budget_split

MODEL = Path.home() / ".cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/38740b847e4cb78f352aba30aa41c76e08e6eb46"
SHARDS = Path(__file__).resolve().parent.parent / "experts"

acc = {"moe": 0.0, "sync_io": 0.0, "lookahead": 0.0, "n_moe_calls": 0}

orig_call = mst.StreamedSwitchGLU.__call__
orig_load = None
orig_la = mst.StreamRuntime._lookahead


def timed_call(self, x, indices):
    mx.eval(x)
    t0 = time.perf_counter()
    y = orig_call(self, x, indices)
    mx.eval(y)
    acc["moe"] += time.perf_counter() - t0
    acc["n_moe_calls"] += 1
    return y


def timed_la(self, layer, xt):
    t0 = time.perf_counter()
    orig_la(self, layer, xt)
    acc["lookahead"] += time.perf_counter() - t0


mst.StreamedSwitchGLU.__call__ = timed_call
mst.StreamRuntime._lookahead = timed_la


def main():
    from mlx_lm.generate import stream_generate
    from mlx_lm.utils import load_tokenizer

    gb = 1 << 30
    lru_b, pre_b, fill_b = budget_split(24, 4, (0.87, 0.13, 0.0))
    model, rt = mst.load_streamed_model(
        MODEL, SHARDS, lru_bytes=lru_b, prefetch_bytes=pre_b,
        filler_bytes=fill_b, prefetch_depth=3, prefetch_width=16, io_threads=8)
    tok = load_tokenizer(MODEL)

    orig_sync = rt.io.load_sync
    def timed_sync(key):
        t0 = time.perf_counter()
        r = orig_sync(key)
        acc["sync_io"] += time.perf_counter() - t0
        return r
    rt.io.load_sync = timed_sync

    text = tok.apply_chat_template(
        [{"role": "user", "content": "Write a detailed essay about the Roman Empire."}],
        add_generation_prompt=True, tokenize=False)

    n = 0
    t0 = time.perf_counter()
    for r in stream_generate(model, tok, text, max_tokens=192):
        n += 1
    total = time.perf_counter() - t0

    per_tok = total / n * 1000
    moe = acc["moe"] / n * 1000
    print(f"\ntokens: {n}  total: {total:.1f}s  -> {per_tok:.1f} ms/token ({n/total:.2f} tok/s)")
    print(f"  MoE layers (incl. sync loads + lookahead): {moe:.1f} ms/token")
    print(f"    of which sync SSD loads: {acc['sync_io']/n*1000:.1f} ms/token")
    print(f"    of which lookahead:      {acc['lookahead']/n*1000:.1f} ms/token")
    print(f"  rest (attention/deltanet, norms, sampling): {per_tok-moe:.1f} ms/token")
    print(f"  stats: {rt.stats()}")
    rt.stop()


if __name__ == "__main__":
    main()
