# CLAUDE.md — moe_stream

## What this project is

`moe_stream` is an MLX-native inference engine for MoE models **larger than RAM**.
Expert weights live as per-expert shards on SSD and are streamed in on demand,
with predictive prefetch hiding the IO behind compute. Target hardware: Mac
M4, 24 GB. Reference model: `mlx-community/Qwen3.6-35B-A3B-4bit` (20.4 GB),
256 experts/layer, top-8, 40 MoE layers. Current speed: ~12.4 tok/s one-shot,
19-22 tok/s warm multi-turn chat (decode arena: per-layer stacked expert
slots + gather_qmm at T=1; MOE_NO_ARENA=1 restores the per-expert loop).
Remaining structural costs: per-layer router readback + attention.
NB benchmarks on this machine: run-to-run variance up to 2× (thermal + OS
page cache) — only trust sandwiched A/B runs.

The design constraint that defines the project: **model > RAM, stream experts
from SSD — never shrink the model to fit.**

## Where we're going (current direction)

We are contributing `moe_stream` to **Odysseus**
(github.com/pewdiepie-archdaemon/odysseus), a self-hosted AI workspace, so its
expert-streaming capability is available behind Odysseus's chat UI. Odysseus
today serves local models only as whole models via standard runners
(llama-server, vllm, ollama) — it has **no** way to run a MoE bigger than RAM.
That gap is our pitch.

User's working copy of Odysseus: `/Users/luca/Desktop/odysseus` (branch `dev`).

### Architecture decision (settled)

Integrate as a **subprocess runner**, not in-process. Reasons:
- mlx is macOS/Apple-Silicon only; Odysseus also runs on Linux/Docker/NVIDIA.
  In-process import would break non-Mac users and bloat the image.
- Odysseus already serves every model as a separate process exposing an HTTP
  endpoint. moe_stream fits that pattern exactly.
- "Separate process" is NOT "external system the user babysits": Odysseus
  launches and supervises it (like llama-server). UX is seamless.

So: moe_stream exposes an OpenAI-compatible HTTP endpoint; Odysseus's Cookbook
launches it as a runner and the chat talks to it via `endpoint_url`.

## What's done

- `moe_stream/serve.py` — OpenAI-compatible server (`/v1/models`,
  `/v1/chat/completions` with SSE streaming). Stdlib `http.server` only, mlx
  imported lazily so the self-check runs without mlx. Generation serialized
  behind a lock (Metal can't encode concurrently). **Tested working** against
  the Qwen model, streaming + non-streaming, and via Odysseus chat.
- The package is vendored into `/Users/luca/Desktop/odysseus/moe_stream/`
  (a copy; the canonical source is here in this repo).
- `tools/fit_check.py` — given a model's `config.json`, estimates MB streamed
  per token vs SSD bandwidth and returns GOOD / MARGINAL / BAD. Use it before
  sharding any new model to know if streaming is worth it. Handles Qwen and
  Gemma field names. Run: `python3 tools/fit_check.py <model_dir> [--bits N]`.
- `--max-ram-gb` flag in `generate.py` and `serve.py` — hard RAM cap via
  `mx.set_memory_limit`, for testing the model under a chosen ceiling.

## Multi-model generalization

The streaming code already generalizes across the **Qwen3 MoE family** (reads
`config.json`; uses mlx_lm's generic `switch_mlp` naming). No per-architecture
naming abstraction is needed until a non-Qwen-style model (Mixtral, DeepSeek)
is actually targeted — that's YAGNI for now. The one real code gap is
per-tensor mixed-precision quant (`model.py` passes global bits/group_size);
add it only when a target model needs it.

Whether a model is worth streaming depends on **bytes-per-token / SSD
bandwidth**, not active-param count per se. fit_check encodes this. Gemma-4-26B
-A4B measured MARGINAL (3.7 tok/s worst-case) — heavier per expert than Qwen.

## Next steps

1. (current) This CLAUDE.md.
2. Write the **GitHub issue** (English) for Odysseus proposing the feature —
   MUST come before any PR (their CONTRIBUTING auto-closes agent PRs that have
   no prior issue).
3. After maintainer buy-in: PR A (backend runner, no visual changes), then
   PR B (Cookbook card, matching their visual style).

## Hard constraints for the Odysseus contribution

From their `CONTRIBUTING.md` — violating these gets the PR closed unread:
- **Open an issue first.** Agent-generated PRs without a prior issue are closed
  without review. Keep PRs small, one feature each, against `dev` (not `main`).
- **Visual style is enforced.** Reuse existing CSS vars
  (`--fg`, `--bg`, `--card`, `--border`, `--red`), reuse existing
  button/input/card classes, extend the Cookbook rather than adding parallel
  components. **No Unicode emoji** in UI or code — inline monochrome SVG only.
  `Fira Code` monospace, dark theme default. Screenshot required for any visual
  change.
- License: Odysseus is **AGPL-3.0**. Code we put into Odysseus becomes AGPL.
  Keep the canonical moe_stream repo separate to avoid relicensing it.
- Their checks: `python -m pytest`, `python -m py_compile`,
  `node --check static/js/<file>.js`, `docker compose config`.

## Key paths

- Engine: `moe_stream/` (model.py, serve.py, generate.py, shards.py, cache.py,
  io_pool.py, predictor.py, profiler.py).
- Model: `~/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit/`
- Shards: `./experts` (10240 .bin, ~17 GB). Transition table:
  `./transition_table.npy`.
- Framework python WITH mlx: `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3`
  (MacPorts python3 lacks mlx). Run helpers: `run.sh`, `chat.sh`.

## Notes / dead ends

- Prompt-lookup speculative decoding (`--spec` in generate.py) is a **net
  slowdown** on this streamed MoE — see the memory note. Don't re-pursue it as
  a speedup. (The verdict does NOT extend to model-based drafts: expert-subsampled
  self-speculation — `--self-spec 4`, moe_stream/self_spec.py — measures **+21%**
  with token-identical output. Draft = top-4 arena-resident experts chosen on
  GPU via slot tables, zero readback; verify = full path over k tokens in one
  forward. k=4 is the sweet spot: acceptance falls off a cliff at k=5.
  Not yet wired into chat.py.)
- **Gemma-4-26B-A4B: run it stock, nothing to port.** 4-bit = 15 GB, fits in
  24 GB RAM → streaming pointless. Measured (mlx-lm 0.31.3 + mlx 0.32 +
  transformers 5.3.0 venv; model in
  `~/Library/Application Support/DnD/engine/gemma_moe/`): ~28-33 tok/s
  one-shot, batch 75/84/95 tok/s aggregate at B=8/16/32 (peak 21.4 GB).
  Expert-subsampled self-spec is a **net slowdown** in-RAM (best 22.9 vs ~31
  baseline despite 61-77% acceptance): the Qwen win came from skipping SSD IO;
  in RAM the draft still pays attention + router + shared dense MLP, so it
  costs ~a full forward regardless of draft top_k. MTP: no head in checkpoint.
- Gemma IS in the app now: `serve.py --plain` (whole model in RAM via mlx_lm,
  same endpoints) run with `./venv-gemma/bin/python3`; the app's sidebar has a
  model picker (Catalog in Backend.swift) showing only models installed on
  disk. All mlx calls in serve.py go through a single GPU thread
  (engine.gpu): MLX streams are thread-affine and gemma4 MoE crashes if the
  forward runs on an HTTP handler thread. Gemma's `<|channel>thought ...
  <channel|>` reasoning is stripped server-side (_strip_thought_stream).
- App agent modes (serve.py): "agent" = solo agent with real tools
  (list/read/write/run sandboxed in a user-picked workdir, ```tool JSON
  blocks, pi.dev-style prompt, max 16 steps, thinking off); "orchestrate" =
  planner (_PROMPT_ORCH, token-thrifty splitting) -> wave -> reviewer.
  Workdir is required by the app; solo requires it server-side too. The app
  no longer loads a model at startup — welcome screen asks first.
  Memory guards (root cause CONFIRMED 2026-07-08: Metal command-buffer OOM
  kIOGPUCommandBufferCallbackErrorOutOfMemory kills the Qwen server as an
  uncatchable C++ abort): serve.py always sets mx.set_memory_limit(82% of
  physical RAM) unless --max-ram-gb; expert budget baseline is 80% of
  physical RAM (was 100% — that oversizing caused the OOM; peak on the big
  prompt dropped 17.7→13.0 GB, tps unchanged); on streamed models the expert
  LRU shrinks ~150 KB/token past 2048 ctx tokens (ExpertCache.shrink_lru,
  one-way ratchet, floor = half budget) — re-checked every 512 generated
  tokens DURING generation, not just at start; "unlimited" max_tokens is
  hard-capped at 32768. The app now logs server
  stdout/stderr to ~/Library/Logs/moe-stream.log (truncated per launch) and
  a terminationHandler surfaces unexpected server death in the UI; clicking
  the same model restarts it.
  Stop & unlimited: POST /api/stop aborts the running generation
  (engine.abort Event, checked per token) and cancels all pending jobs;
  max_tokens <= 0 means unlimited (normalized to 1<<30 for self_spec, 4096
  cap in waves). Chat sends max_tokens -1; Invia/Lancia become red
  Stop/Ferma while running. Waves now use BatchGenerator incrementally
  (per-agent live text + tok/s in job items, aggregate in job.tps,
  interruptible mid-wave); AgentsView has a grid/list toggle with live
  agent cards and a Σ tok/s header.
  Orchestrator v2: decides its own coder count (UI n = cap), keeps a
  persistent lean conversation (plan + one-line ESITO reports only, never
  coder outputs); coders are stateless; on failures it issues one round of
  "RIFAI n: …" corrections (single round by design); reviewer is stateless
  and sees full outputs. Phases: pianificazione/coding/verifica/correzioni/
  revisione.
