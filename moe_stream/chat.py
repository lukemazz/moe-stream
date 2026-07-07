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
    p.add_argument("-n", "--max-tokens", type=int, default=-1,
                   help="massimo token per risposta (-1 = illimitato: si "
                        "ferma all'EOS; Ctrl+C interrompe comunque)")
    p.add_argument("--ram-gb", type=float, default=24.0)
    p.add_argument("--context-k", type=int, default=8)
    p.add_argument("--table", type=Path, default=None)
    p.add_argument("--prefetch-depth", type=int, default=3)
    p.add_argument("--prefetch-width", type=int, default=16)
    p.add_argument("--io-threads", type=int, default=8)
    p.add_argument("--split", default="0.87,0.13,0.0",
                   help="RAM fractions for lru,prefetch,filler")
    p.add_argument("--self-spec", type=int, default=0, metavar="K",
                   help="auto-speculativa: K token di bozza per ciclo (0=off)")
    p.add_argument("--draft-n", type=int, default=8,
                   help="esperti residenti usati dalla bozza")
    p.add_argument("--mtp", type=Path, default=None, metavar="SAFETENSORS",
                   help="testa MTP nativa come drafter (es. mtp_head."
                        "safetensors); richiede --self-spec")
    args = p.parse_args()
    max_toks = args.max_tokens if args.max_tokens > 0 else (1 << 30)

    fracs = tuple(float(x) for x in args.split.split(","))
    lru_b, pre_b, fill_b = budget_split(args.ram_gb, args.context_k, fracs)
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
    mtp_head = mtp_cache = None
    if args.mtp:
        from .mtp import load_mtp_head, make_mtp_cache
        mtp_head = load_mtp_head(args.mtp, model.language_model)
        mx.eval(mtp_head.parameters())
        mtp_cache = make_mtp_cache(model)
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
            if args.mtp:
                from .mtp import make_mtp_cache
                mtp_cache = make_mtp_cache(model)
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
            if args.self_spec:
                from .self_spec import mtp_spec_generate, self_spec_generate
                detok = tokenizer.detokenizer
                detok.reset()
                if mtp_head is not None:
                    gen = mtp_spec_generate(
                        model, rt, mtp_head, new_tokens, cache=prompt_cache,
                        mtp_cache=mtp_cache, max_tokens=max_toks,
                        k=args.self_spec, eos=set(tokenizer.eos_token_ids))
                else:
                    gen = self_spec_generate(
                        model, rt, new_tokens, cache=prompt_cache,
                        max_tokens=max_toks, k=args.self_spec,
                        draft_n=args.draft_n,
                        eos=set(tokenizer.eos_token_ids))
                for tok in gen:
                    detok.add_token(tok)
                    print(detok.last_segment, end="", flush=True)
                    n_tok += 1
                detok.finalize()
                print(detok.last_segment, end="", flush=True)
            else:
                for resp in stream_generate(model, tokenizer, new_tokens,
                                            max_tokens=max_toks,
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
