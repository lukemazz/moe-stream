"""Is an MLX MoE model worth streaming experts from SSD?

Reads config.json, estimates the bytes pulled from disk per decoded token, and
turns that into the throughput ceiling the SSD alone allows. If that ceiling is
already usable, the model is a good streaming fit; if it isn't, no amount of
prefetch code will save you — you'd be SSD-bound.

    python3 tools/fit_check.py <model_dir> [--ssd-gbps 3.0] [--bits N]

The number is the worst case: every routed expert a cache miss, no reuse across
tokens or adjacent layers. Real throughput is higher because the LRU holds hot
experts — so a passing worst case means "definitely streamable", a failing one
means "you'd be betting everything on the cache hit rate".
"""

import argparse
import json
from pathlib import Path

MB = 1024 * 1024


def expert_bytes(hidden, moe_inter, bits, group_size):
    # one expert = gate + up + down, each hidden*moe_inter params
    params = 3 * hidden * moe_inter
    quant = params * bits / 8
    # per quant group: one fp16 scale + one fp16 bias
    overhead = (params / group_size) * 2 * 2
    return quant + overhead


def fit(cfg, ssd_gbps, bits_override):
    t = cfg.get("text_config", cfg)
    n_layers = t["num_hidden_layers"]
    dense = set(t.get("mlp_only_layers") or [])
    n_moe = n_layers - len(dense)
    top_k = t.get("num_experts_per_tok") or t["top_k_experts"]  # gemma names it differently
    hidden = t["hidden_size"]
    moe_inter = t["moe_intermediate_size"]

    quant = cfg.get("quantization") or cfg.get("quantization_config") or {}
    bits = bits_override or quant.get("bits")
    assert bits, "no quantization in config; pass --bits"
    group_size = quant.get("group_size", 64)

    eb = expert_bytes(hidden, moe_inter, bits, group_size)
    per_token = n_moe * top_k * eb
    ceiling = ssd_gbps * 1e9 / per_token  # tok/s if fully SSD-bound

    verdict = ("GOOD — streamable on worst case alone" if ceiling >= 10 else
               "MARGINAL — depends on cache hit rate" if ceiling >= 3 else
               "BAD — SSD-bound, streaming won't help")
    return {
        "moe_layers": n_moe, "top_k": top_k,
        "expert_MB": round(eb / MB, 2),
        "bytes_per_token_MB": round(per_token / MB, 1),
        "ssd_gbps": ssd_gbps,
        "tok_s_ceiling_worstcase": round(ceiling, 1),
        "verdict": verdict,
    }


def _selfcheck():
    # Qwen3.6-35B-A3B at 4 bits: ~1.7 MB/expert (17 GB / 10240 shards), and a
    # worst-case ceiling in the low single-digit-to-teens range.
    r = fit({"num_hidden_layers": 40, "num_experts_per_tok": 8,
             "hidden_size": 2048, "moe_intermediate_size": 512,
             "quantization": {"bits": 4, "group_size": 64}}, 3.0, None)
    assert 1.5 < r["expert_MB"] < 2.0, r
    assert r["moe_layers"] == 40 and r["tok_s_ceiling_worstcase"] > 0, r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model_dir", type=Path)
    p.add_argument("--ssd-gbps", type=float, default=3.0,
                   help="sustained SSD read bandwidth; measure yours, don't trust the spec sheet")
    p.add_argument("--bits", type=int, default=None,
                   help="override quant bits if config.json has none")
    args = p.parse_args()
    cfg = json.loads((args.model_dir / "config.json").read_text())
    for k, v in fit(cfg, args.ssd_gbps, args.bits).items():
        print(f"{k:28} {v}")


if __name__ == "__main__":
    _selfcheck()
    main()
