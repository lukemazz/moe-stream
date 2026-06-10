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
  + predictive prefetch + filler cache  3.7 tok/s   (61%+, keeps rising)
```

---

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
2. **Prefetch staging** (predictive). While the GPU computes layer *N*, a
   **transition table** — `P(expert j active at layer N+1 | expert i active at
   layer N)`, built offline by profiling the model on a small corpus — ranks
   the experts most likely needed at layer *N+1* (and *N+2*, chained), and the
   IO pool loads them at high priority in parallel with compute.
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
| `--prefetch-depth` | 2 | how many layers ahead to predict |
| `--prefetch-width` | 8 | experts prefetched per predicted layer |
| `--io-threads` | 4 | SSD reader threads |

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
