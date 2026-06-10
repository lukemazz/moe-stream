"""Transition-table expert predictor (spec sections 6.2-6.3).

transition[l][e_src][e_dst] = P(e_dst active at MoE layer l+1 | e_src active at l).
Built offline by profiler.py; float16 npy of shape [n_layers, E, E].
Falls back to uniform popularity (column marginals) when no table exists.
"""

from pathlib import Path

import numpy as np


class Predictor:
    def __init__(self, n_layers: int, n_experts: int, table_path: Path | None = None):
        self.n_layers = n_layers
        self.n_experts = n_experts
        self.table = None
        if table_path and Path(table_path).exists():
            self.table = np.load(table_path).astype(np.float32)
            assert self.table.shape == (n_layers, n_experts, n_experts), \
                f"transition table shape {self.table.shape} != {(n_layers, n_experts, n_experts)}"

    def predict(self, layer: int, active_experts, top_k: int = 8) -> list[int]:
        """Experts most likely active at MoE layer `layer`+1."""
        if self.table is None or layer + 1 >= self.n_layers:
            return []
        scores = self.table[layer][list(active_experts)].sum(axis=0)
        k = min(top_k, self.n_experts)
        return np.argpartition(scores, -k)[-k:][np.argsort(scores[np.argpartition(scores, -k)[-k:]])[::-1]].tolist()
