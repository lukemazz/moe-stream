"""CLI generation with streamed experts (spec sections 8, 9, 11.8).

Usage:
  python -m moe_stream.generate MODEL_DIR SHARD_DIR -p "prompt" [options]
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx


def budget_split(total_ram_gb: float, context_k: int):
    """Spec section 9 memory budget calculator."""
    avail = total_ram_gb - 4 - 4 - 0.125 * context_k - 1
    avail = max(avail, 1.0)
    gb = 1 << 30
    return (int(avail * 0.27 * gb),   # LRU
            int(avail * 0.09 * gb),   # prefetch staging
            int(avail * 0.64 * gb))   # filler


def main():
    from mlx_lm.generate import stream_generate
    from mlx_lm.utils import load_tokenizer

    from .model import load_streamed_model

    p = argparse.ArgumentParser()
    p.add_argument("model_dir", type=Path)
    p.add_argument("shard_dir", type=Path)
    p.add_argument("-p", "--prompt", default="What is 2+2?")
    p.add_argument("-n", "--max-tokens", type=int, default=128)
    p.add_argument("--ram-gb", type=float, default=24.0)
    p.add_argument("--context-k", type=int, default=4)
    p.add_argument("--table", type=Path, default=None,
                   help="transition_table.npy from the profiler")
    p.add_argument("--prefetch-depth", type=int, default=2)
    p.add_argument("--prefetch-width", type=int, default=8)
    p.add_argument("--io-threads", type=int, default=4)
    args = p.parse_args()

    lru_b, pre_b, fill_b = budget_split(args.ram_gb, args.context_k)
    print(f"budgets: LRU={lru_b/1e9:.1f}GB prefetch={pre_b/1e9:.1f}GB "
          f"filler={fill_b/1e9:.1f}GB")

    t0 = time.time()
    model, rt = load_streamed_model(
        args.model_dir, args.shard_dir,
        lru_bytes=lru_b, prefetch_bytes=pre_b, filler_bytes=fill_b,
        table_path=args.table, prefetch_depth=args.prefetch_depth,
        prefetch_width=args.prefetch_width, io_threads=args.io_threads)
    tokenizer = load_tokenizer(args.model_dir)
    print(f"fixed parts loaded in {time.time()-t0:.1f}s "
          f"(RAM: {mx.get_active_memory()/1e9:.2f} GB)")

    msgs = [{"role": "user", "content": args.prompt}]
    text = tokenizer.apply_chat_template(msgs, add_generation_prompt=True,
                                         tokenize=False)

    n_tok = 0
    t0 = time.time()
    for resp in stream_generate(model, tokenizer, text, max_tokens=args.max_tokens):
        print(resp.text, end="", flush=True)
        n_tok += 1
    dt = time.time() - t0
    print(f"\n\n--- {n_tok} tokens in {dt:.1f}s = {n_tok/dt:.2f} tok/s")
    print(json.dumps(rt.stats(), indent=2))
    print(f"peak RAM: {mx.get_peak_memory()/1e9:.2f} GB")
    rt.stop()


if __name__ == "__main__":
    main()
