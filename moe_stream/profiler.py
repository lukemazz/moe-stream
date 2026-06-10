"""Offline profiler (spec section 6.2): runs the streamed model on a small
corpus, records expert activations per MoE layer, and builds the per-layer
transition matrix transition[l][e_src][e_dst] saved as transition_table.npy.
"""

import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT_CORPUS = [
    "Write a Python function that parses a CSV file and returns a dict.",
    "Solve step by step: if 3x + 7 = 25, what is x?",
    "Summarize the plot of Romeo and Juliet in three sentences.",
    "Explain how a hash map handles collisions.",
    "Translate to French: 'The weather is beautiful today.'",
    "What are the trade-offs between TCP and UDP?",
    "Write a SQL query to find the top 5 customers by revenue.",
    "Describe the water cycle for a 10-year-old.",
]


class ActivationRecorder:
    def __init__(self, n_layers: int, n_experts: int):
        self.counts = np.zeros((n_layers, n_experts, n_experts), dtype=np.float64)
        self._token_layers: dict[int, np.ndarray] = {}

    def __call__(self, layer: int, idx: np.ndarray):
        # idx: [T, k] expert ids for T tokens at this layer
        prev = self._token_layers.get(layer - 1)
        if prev is not None and prev.shape[0] == idx.shape[0]:
            for t in range(idx.shape[0]):
                self.counts[layer - 1][np.ix_(prev[t], idx[t])] += 1.0
        self._token_layers[layer] = idx

    def table(self) -> np.ndarray:
        sums = self.counts.sum(axis=-1, keepdims=True)
        sums[sums == 0] = 1.0
        return (self.counts / sums).astype(np.float16)


def main():
    from mlx_lm.generate import generate
    from mlx_lm.utils import load_tokenizer

    from .model import load_streamed_model

    p = argparse.ArgumentParser()
    p.add_argument("model_dir", type=Path)
    p.add_argument("shard_dir", type=Path)
    p.add_argument("-o", "--out", type=Path, default=Path("transition_table.npy"))
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--corpus", type=Path, help="optional text file, one prompt per line")
    args = p.parse_args()

    prompts = (args.corpus.read_text().splitlines()
               if args.corpus else DEFAULT_CORPUS)

    gb = 1 << 30
    model, rt = load_streamed_model(
        args.model_dir, args.shard_dir,
        lru_bytes=4 * gb, prefetch_bytes=0, filler_bytes=0)
    tokenizer = load_tokenizer(args.model_dir)

    rec = ActivationRecorder(rt.n_layers, rt.n_experts)
    rt.recorder = rec

    for i, prompt in enumerate(prompts):
        msgs = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(msgs, add_generation_prompt=True,
                                             tokenize=False)
        generate(model, tokenizer, text, max_tokens=args.max_tokens)
        print(f"[{i+1}/{len(prompts)}] profiled: {prompt[:50]}", flush=True)

    np.save(args.out, rec.table())
    print(f"saved {args.out} shape={rec.table().shape}")
    rt.stop()


if __name__ == "__main__":
    main()
