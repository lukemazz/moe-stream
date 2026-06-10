"""Interactive multi-turn chat with streamed experts.

Usage: python -m moe_stream.chat MODEL_DIR SHARD_DIR [--table T] [options]
Commands inside the chat: /stats, /reset, /quit
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx


def main():
    from mlx_lm.generate import stream_generate
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.utils import load_tokenizer

    from .generate import budget_split
    from .model import load_streamed_model

    p = argparse.ArgumentParser()
    p.add_argument("model_dir", type=Path)
    p.add_argument("shard_dir", type=Path)
    p.add_argument("-n", "--max-tokens", type=int, default=1024)
    p.add_argument("--ram-gb", type=float, default=24.0)
    p.add_argument("--context-k", type=int, default=8)
    p.add_argument("--table", type=Path, default=None)
    p.add_argument("--prefetch-depth", type=int, default=3)
    p.add_argument("--prefetch-width", type=int, default=16)
    p.add_argument("--io-threads", type=int, default=8)
    args = p.parse_args()

    lru_b, pre_b, fill_b = budget_split(args.ram_gb, args.context_k)
    print("Caricamento modello (parti fisse)...", flush=True)
    model, rt = load_streamed_model(
        args.model_dir, args.shard_dir,
        lru_bytes=lru_b, prefetch_bytes=pre_b, filler_bytes=fill_b,
        table_path=args.table, prefetch_depth=args.prefetch_depth,
        prefetch_width=args.prefetch_width, io_threads=args.io_threads)
    tokenizer = load_tokenizer(args.model_dir)
    print(f"Pronto. RAM fissa: {mx.get_active_memory()/1e9:.2f} GB. "
          f"Comandi: /stats /reset /effort high|low /quit "
          f"(Ctrl+C ferma la generazione)\n")

    prompt_cache = make_prompt_cache(model)
    thinking = True  # effort: high = thinking abilitato, low = disabilitato

    while True:
        try:
            user = input("tu> ").strip()
        except KeyboardInterrupt:
            print("\n(usa /quit o Ctrl+D per uscire)")
            continue
        except EOFError:
            print()
            break
        if not user:
            continue
        if user == "/quit":
            break
        if user.startswith("/effort"):
            arg = user.removeprefix("/effort").strip().lower()
            if arg in ("high", "on"):
                thinking = True
            elif arg in ("low", "off"):
                thinking = False
            else:
                print("uso: /effort high|low")
                continue
            print(f"(effort: {'high — thinking abilitato' if thinking else 'low — thinking disabilitato'})")
            continue
        if user == "/stats":
            print(json.dumps(rt.stats(), indent=2))
            print(f"RAM attiva: {mx.get_active_memory()/1e9:.2f} GB, "
                  f"picco: {mx.get_peak_memory()/1e9:.2f} GB")
            continue
        if user == "/reset":
            prompt_cache = make_prompt_cache(model)
            print("(conversazione azzerata)")
            continue

        # persistent KV cache holds the whole conversation; feed only the new turn
        new_tokens = tokenizer.apply_chat_template(
            [{"role": "user", "content": user}],
            add_generation_prompt=True, tokenize=True,
            enable_thinking=thinking)

        n_tok = 0
        t0 = time.time()
        print("ai> ", end="", flush=True)
        try:
            for resp in stream_generate(model, tokenizer, new_tokens,
                                        max_tokens=args.max_tokens,
                                        prompt_cache=prompt_cache):
                print(resp.text, end="", flush=True)
                n_tok += 1
        except KeyboardInterrupt:
            print("\n   [generazione interrotta]", end="")
        dt = time.time() - t0
        print(f"\n   [{n_tok} token, {n_tok/dt:.2f} tok/s]\n")

    rt.stop()


if __name__ == "__main__":
    main()
