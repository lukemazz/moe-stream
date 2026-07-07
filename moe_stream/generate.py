"""CLI generation with streamed experts (spec sections 8, 9, 11.8).

Usage:
  python -m moe_stream.generate MODEL_DIR SHARD_DIR -p "prompt" [options]
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx


def budget_split(total_ram_gb: float, context_k: int,
                 fractions=(0.87, 0.13, 0.0)):
    """Spec section 9 memory budget calculator. fractions = (lru, prefetch, filler)."""
    avail = total_ram_gb - 4 - 4 - 0.125 * context_k - 1
    avail = max(avail, 1.0)
    gb = 1 << 30
    return tuple(int(avail * f * gb) for f in fractions)


def main():
    from mlx_lm.generate import stream_generate
    from mlx_lm.utils import load_tokenizer

    from .model import load_streamed_model

    p = argparse.ArgumentParser()
    p.add_argument("model_dir", type=Path)
    p.add_argument("shard_dir", type=Path)
    p.add_argument("-p", "--prompt", default="What is 2+2?")
    p.add_argument("-n", "--max-tokens", type=int, default=128)
    p.add_argument("--ram-gb", type=float, default=0,
                   help="RAM totale per i budget (0 = auto dalla RAM fisica)")
    p.add_argument("--context-k", type=int, default=4)
    p.add_argument("--table", type=Path, default=None,
                   help="transition_table.npy from the profiler")
    p.add_argument("--prefetch-depth", type=int, default=3)
    p.add_argument("--prefetch-width", type=int, default=16)
    p.add_argument("--io-threads", type=int, default=8)
    p.add_argument("--split", default="0.87,0.13,0.0",
                   help="RAM fractions for lru,prefetch,filler")
    p.add_argument("--self-spec", type=int, default=3, metavar="K",
                   help="auto-speculativa: K token di bozza per ciclo (default 3; 0=off)")
    p.add_argument("--draft-n", type=int, default=8,
                   help="esperti residenti usati dalla bozza")
    p.add_argument("--mtp", type=Path, default=None, metavar="SAFETENSORS",
                   help="testa MTP nativa come drafter (es. mtp_head."
                        "safetensors); richiede --self-spec")
    p.add_argument("--batch", type=Path, default=None, metavar="FILE",
                   help="modalità batch: un prompt per riga, generati in "
                        "lockstep a wave da --batch-size (~100 tok/s "
                        "aggregati a wave 32; disattiva --self-spec)")
    p.add_argument("--batch-size", type=int, default=32,
                   help="ampiezza della wave (32 = sweet spot misurato: "
                        "~104 tok/s aggregati, ~3 tok/s per prompt)")
    args = p.parse_args()

    if not args.ram_gb:
        import os as _os
        args.ram_gb = (_os.sysconf("SC_PAGE_SIZE")
                       * _os.sysconf("SC_PHYS_PAGES")) / 1e9
    fracs = tuple(float(x) for x in args.split.split(","))
    lru_b, pre_b, fill_b = budget_split(args.ram_gb, args.context_k, fracs)
    print(f"budgets: LRU={lru_b/1e9:.1f}GB prefetch={pre_b/1e9:.1f}GB "
          f"filler={fill_b/1e9:.1f}GB (ram={args.ram_gb:.0f}GB)")

    try:
        mx.set_wired_limit(mx.metal.device_info()["max_recommended_working_set_size"])
    except Exception:
        pass

    t0 = time.time()
    model, rt = load_streamed_model(
        args.model_dir, args.shard_dir,
        lru_bytes=lru_b, prefetch_bytes=pre_b, filler_bytes=fill_b,
        table_path=args.table, prefetch_depth=args.prefetch_depth,
        prefetch_width=args.prefetch_width, io_threads=args.io_threads)
    tokenizer = load_tokenizer(args.model_dir)
    print(f"fixed parts loaded in {time.time()-t0:.1f}s "
          f"(RAM: {mx.get_active_memory()/1e9:.2f} GB)")

    if args.batch:
        from mlx_lm.generate import batch_generate
        prompts = [ln.strip() for ln in args.batch.read_text().splitlines()
                   if ln.strip()]
        print(f"[batch] {len(prompts)} prompt, wave da {args.batch_size}")
        t0 = time.time()
        n_tok = 0
        for w0 in range(0, len(prompts), args.batch_size):
            wave = prompts[w0:w0 + args.batch_size]
            ids = [tokenizer.apply_chat_template(
                       [{"role": "user", "content": p}],
                       add_generation_prompt=True) for p in wave]
            res = batch_generate(model, tokenizer, prompts=ids,
                                 max_tokens=args.max_tokens, verbose=False)
            for p, out in zip(wave, res.texts):
                n_tok += len(tokenizer.encode(out))
                print(f"\n===== PROMPT: {p}\n{out}")
        dt = time.time() - t0
        print(f"\n--- [batch] {n_tok} token in {dt:.1f}s = "
              f"{n_tok/dt:.2f} tok/s aggregati")
        print(json.dumps(rt.stats(), indent=2))
        print(f"peak RAM: {mx.get_peak_memory()/1e9:.2f} GB")
        rt.stop()
        return

    msgs = [{"role": "user", "content": args.prompt}]
    text = tokenizer.apply_chat_template(msgs, add_generation_prompt=True,
                                         tokenize=False)

    n_tok = 0
    t0 = time.time()
    resp = None
    if args.self_spec:
        from .self_spec import mtp_spec_generate, self_spec_generate
        prompt_ids = tokenizer.encode(text)
        detok = tokenizer.detokenizer
        detok.reset()
        if args.mtp:
            from .mtp import load_mtp_head
            head = load_mtp_head(args.mtp, model.language_model)
            mx.eval(head.parameters())
            gen = mtp_spec_generate(model, rt, head, prompt_ids,
                                    max_tokens=args.max_tokens,
                                    k=args.self_spec,
                                    eos=set(tokenizer.eos_token_ids))
        else:
            gen = self_spec_generate(model, rt, prompt_ids,
                                     max_tokens=args.max_tokens,
                                     k=args.self_spec, draft_n=args.draft_n,
                                     eos=set(tokenizer.eos_token_ids))
        for tok in gen:
            detok.add_token(tok)
            print(detok.last_segment, end="", flush=True)
            n_tok += 1
        detok.finalize()
        print(detok.last_segment, end="", flush=True)
        dt = time.time() - t0
        s = rt.spec_stats
        print(f"\n\n--- {n_tok} tokens in {dt:.1f}s = {n_tok/dt:.2f} tok/s")
        print(f"[spec] {s['accepted']}/{s['drafted']} bozze accettate "
              f"({s['accepted']/max(s['drafted'],1):.0%}), "
              f"{s['cycles']} cicli, {s['partial']} parziali")
    else:
        for resp in stream_generate(model, tokenizer, text, max_tokens=args.max_tokens):
            print(resp.text, end="", flush=True)
            n_tok += 1
        dt = time.time() - t0
        print(f"\n\n--- {n_tok} tokens in {dt:.1f}s = {n_tok/dt:.2f} tok/s")
    if resp is not None:
        print(f"prefill: {resp.prompt_tokens} tokens at {resp.prompt_tps:.1f} tok/s")
    print(json.dumps(rt.stats(), indent=2))
    print(f"peak RAM: {mx.get_peak_memory()/1e9:.2f} GB")
    rt.stop()


if __name__ == "__main__":
    main()
