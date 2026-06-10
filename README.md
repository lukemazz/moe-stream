# moe-stream — Predictive Expert Prefetching for MoE Models on Apple Silicon

Run Mixture-of-Experts models **larger than your Mac's RAM** by streaming expert
weights from the SSD, with predictive prefetching that loads experts *before*
the model needs them.

Reference setup: **Qwen3.6-35B-A3B 4-bit MLX (19 GB)** running on a **24 GB**
Apple Silicon Mac with a peak RAM usage of ~10 GB and only ~1.6 GB of fixed
weights resident in memory.

```
Measured on M4 / 24 GB / NVMe:
  LRU cache only (no prediction)        1.8 tok/s   (45% cache hit rate)
  + transition-table prefetch + filler  3.7 tok/s   (61%+)
  + router lookahead (pre-gating)       3.9 tok/s   (up to 89% hit rate)
```

---

## Why this is a big deal

A 35-billion-parameter model is simply **not supposed to run on a 24 GB
laptop**. The 4-bit checkpoint alone is 19 GB; add the OS, the KV cache and
the working buffers and the math stops working — by the conventional rules,
this model belongs on a workstation with 32–64 GB of unified memory, or on a
GPU server. The standard answers for consumer hardware are all compromises:
use a much smaller (and dumber) model, quantize so aggressively that quality
collapses, or let the OS swap and watch generation crawl at seconds per token.

This project takes none of those compromises. The **full 35B model, at a
healthy 4-bit quantization, runs on a base-spec Mac** — with only ~1.6 GB of
weights permanently resident and a peak of ~10 GB of RAM — at interactive
speeds. Nothing about the model was shrunk, distilled or trimmed: all 10,240
experts are there, fetched from the SSD at the exact moment the router asks
for them, and increasingly *before* it asks, thanks to predictive prefetching.

What makes this exciting is the shift in perspective it represents:

- **The SSD becomes part of the memory hierarchy.** Modern NVMe storage on
  Apple Silicon is fast enough (multiple GB/s) that, with the right access
  pattern — one expert, one sequential read — it can serve as a *lazy tier of
  RAM* rather than a place where models go to die in swap.
- **MoE sparsity is turned from a training trick into a deployment
  superpower.** Only ~3% of the expert weights are needed per token; this
  system is built entirely around exploiting that asymmetry.
- **Prediction hides the latency.** The transition table learns the model's
  own routing habits and overlaps SSD reads with GPU compute, so the
  bottleneck that should make this approach unusable largely disappears —
  in our measurements it *doubled* throughput.
- **It scales with intelligence-per-byte.** The same machinery applies to any
  MoE checkpoint with this layout: the bigger and sparser the model, the more
  you gain. The RAM in your laptop stops being the ceiling on the size of the
  model you can run.

In short: a consumer laptop just ran a model from a weight class it had no
right to touch, and it did so by being clever about *when* weights are loaded
rather than ruthless about *which* weights are kept.

## Why this works: MoE sparsity

In a dense transformer every FFN weight is needed for every token. In an MoE
model like Qwen3.6-35B-A3B, each of the 40 layers has **256 experts** but the
router activates only **8 of them per token** (~3%). The other 248 experts are
dead weight for that token — so they can live on the SSD instead of in RAM.

The catch is IO latency: even loading just the active experts per layer stalls
the GPU if done synchronously. This project removes that stall by overlapping
SSD reads with GPU compute:

```
without prefetch:  [GPU layer N] → wait SSD → [GPU layer N+1] → wait SSD → ...
with prefetch:     [GPU layer N]                [GPU layer N+1]
                   [SSD: load predicted experts for N+1]   ← in parallel
```

On Apple Silicon, GPU compute (MLX/Metal) and NVMe IO run on separate
subsystems, so the overlap is real.

## Architecture

```
                       ┌──────────────────────┐
                       │   MoE layer forward  │
                       └──────────┬───────────┘
                                  │ lookup: LRU → prefetch → filler → SSD
                       ┌──────────▼───────────┐
                       │  three-tier cache    │
                       │  (ExpertCache)       │
                       └──────────┬───────────┘
            ┌─────────────────────┼─────────────────────┐
   ┌────────▼────────┐  ┌─────────▼────────┐  ┌─────────▼────────┐
   │ LRU cache       │  │ prefetch staging │  │ filler cache     │
   │ recent experts  │  │ predicted experts│  │ random warm fill │
   │ evicted last    │  │                  │  │ evicted FIRST    │
   └─────────────────┘  └─────────┬────────┘  └─────────┬────────┘
                        ┌─────────▼─────────────────────▼────────┐
                        │  async IO thread pool (priority queue) │
                        └─────────────────┬──────────────────────┘
                                  ┌───────▼────────┐
                                  │ NVMe SSD       │
                                  │ 10240 expert   │
                                  │ shard files    │
                                  └────────────────┘
```

**Fixed parts always in RAM** (~1.6 GB): embeddings, attention / GatedDeltaNet
weights, layer norms, MoE routers, shared experts. Everything accessed on every
token, intolerant to SSD latency.

**Expert shards on SSD**: every expert of every layer is a standalone `.bin`
file (`experts/layer_LL/expert_EEE.bin`, ~1.8 MB each) containing the
4-bit-quantized `gate/up/down` projections plus scales and biases. One expert =
one `read()` — no seeks, no read amplification.

### The three cache tiers

1. **LRU cache** (reactive). Recently used experts. Exploits temporal locality:
   experts used for recent tokens tend to fire again.
2. **Prefetch staging** (predictive). While the GPU computes layer *N*, the
   experts most likely needed at the next layers are loaded at high priority,
   in parallel with compute. Prediction uses **router lookahead
   (pre-gating)**: the routers of layers *N+1..N+depth* — tiny, always
   resident — are applied directly to the current hidden state. Because the
   residual stream changes slowly across adjacent layers, this predicts
   upcoming experts with high accuracy (~89% end-to-end cache hit rate in our
   tests). A statistical **transition table** built offline by
   `moe_stream.profiler` is used as automatic fallback when lookahead is
   disabled (`rt.use_lookahead = False`).
3. **Filler cache** (opportunistic). Any leftover RAM is filled in the
   background with *randomly chosen* experts. Expert activation is far from
   uniform, so a random resident expert has a better-than-uniform chance of
   being useful — and it costs nothing, because filler entries have the lowest
   priority and are always evicted first.

Strict eviction order: **filler → prefetch → LRU**. Lookup order on the hot
path: **LRU → prefetch → filler → SSD (sync load, counted as a miss)**.

The transition table is tiny (40×256×256 float16 ≈ 5 MB) and pays for itself
immediately: in our tests it roughly doubled throughput.

---

## Installation

Requires an Apple Silicon Mac, Python ≥ 3.10 and an NVMe SSD with ~37 GB free
(model download + extracted shards).

```bash
pip install -r requirements.txt
```

## Usage

### 1. Download the model (~19 GB)

```bash
python -c "from huggingface_hub import snapshot_download; \
  print(snapshot_download('mlx-community/Qwen3.6-35B-A3B-4bit'))"
```

The printed path is your `MODEL_DIR`.

### 2. Split the experts into shards (~17 GB, a few minutes)

```bash
python -m moe_stream.shards MODEL_DIR ./experts
```

### 3. (Optional but recommended) Build the transition table

Profiles the model on a small built-in corpus and writes
`transition_table.npy`. Without it, prefetching is disabled and you fall back
to LRU + filler only.

```bash
python -m moe_stream.profiler MODEL_DIR ./experts -o transition_table.npy
```

### 4. Chat

```bash
python -m moe_stream.chat MODEL_DIR ./experts --table transition_table.npy
```

Multi-turn conversation with a persistent KV cache (each turn extends the
previous one — no re-prefill of the history). **Ctrl+C stops the current
generation** without leaving the chat. In-chat commands: `/stats` (cache hit
rates, RAM), `/effort high|low` (enable/disable the model's thinking mode),
`/reset`, `/quit`.

### One-shot generation / benchmark

```bash
python -m moe_stream.generate MODEL_DIR ./experts \
    -p "Explain why the sky is blue." -n 256 --table transition_table.npy
```

Prints tokens/sec, per-tier hit rates and peak RAM at the end.

`run.sh` and `chat.sh` are convenience launchers with the paths pre-filled —
edit the variables at the top to match your machine.

### Tuning

| Flag | Default | Meaning |
|---|---|---|
| `--ram-gb` | 24 | total machine RAM; drives the budget calculator |
| `--context-k` | 4/8 | context length (k tokens); reserves KV-cache RAM |
| `--prefetch-depth` | 3 | how many layers ahead to predict |
| `--prefetch-width` | 16 | experts prefetched per predicted layer |
| `--io-threads` | 8 | SSD reader threads |

RAM budget split (after subtracting OS, fixed weights, KV cache, safety
margin): 27% LRU, 9% prefetch staging, 64% filler.

## Repository layout

```
moe_stream/
  shards.py     expert shard format + safetensors → per-expert .bin converter
  cache.py      three-tier cache (LRU / prefetch / filler), strict eviction order
  io_pool.py    priority-queue thread pool + background filler loop
  predictor.py  transition-table next-layer expert prediction
  profiler.py   offline activation profiling → transition_table.npy
  model.py      MLX wrapper: swaps each MoE SwitchGLU for a streamed version
  generate.py   one-shot generation CLI with stats
  chat.py       interactive multi-turn chat (persistent KV cache)
run.sh, chat.sh launchers
```

## Limitations / non-goals

- Apple Silicon + MLX only (the IO/compute overlap assumptions are
  NVMe-on-Mac specific).
- Single-stream inference; no request batching.
- Works with MLX-quantized MoE checkpoints that use the
  `switch_mlp.{gate,up,down}_proj` layout (Qwen3.6 / Qwen3-Next family in
  `mlx-lm`).

## Acknowledgements

Inspired by activation-aware expert offloading work such as PowerInfer (SJTU)
and the MLX expert-streaming ecosystem.


Note: this app is partially vibe-coded with claude fable5, if you notice a bug please report it.
