"""Scarica SOLO i tensori mtp.* dal checkpoint Qwen/Qwen3.6-35B-A3B via
HTTP range-request sul formato safetensors (header JSON con offset per
tensore), e li salva in un unico mtp_head.safetensors bf16 locale.
"""
import json
import struct
import subprocess
import sys

import mlx.core as mx
import numpy as np

BASE = "https://huggingface.co/Qwen/Qwen3.6-35B-A3B/resolve/main"
SHARDS = ["model-00025-of-00026.safetensors",
          "model-00026-of-00026.safetensors"]
OUT = sys.argv[1] if len(sys.argv) > 1 else "mtp_head.safetensors"

DTYPES = {"BF16": (np.uint16, mx.bfloat16), "F16": (np.uint16, mx.float16),
          "F32": (np.uint32, mx.float32)}


def fetch_range(url, start, end):
    return subprocess.run(
        ["curl", "-sfL", "-r", f"{start}-{end}", url],
        capture_output=True, check=True).stdout


out = {}
total = 0
for shard in SHARDS:
    url = f"{BASE}/{shard}"
    (hlen,) = struct.unpack("<Q", fetch_range(url, 0, 7))
    header = json.loads(fetch_range(url, 8, 8 + hlen - 1))
    base_off = 8 + hlen
    for name, meta in header.items():
        if not name.startswith("mtp."):
            continue
        s, e = meta["data_offsets"]
        raw = fetch_range(url, base_off + s, base_off + e - 1)
        np_dt, mx_dt = DTYPES[meta["dtype"]]
        arr = np.frombuffer(raw, np_dt).reshape(meta["shape"])
        out[name] = mx.array(arr).view(mx_dt)
        total += len(raw)
        print(f"  {name}  {meta['dtype']} {meta['shape']} "
              f"({len(raw)/1e6:.1f} MB)", flush=True)

mx.save_safetensors(OUT, out)
print(f"\nsalvati {len(out)} tensori, {total/1e9:.2f} GB -> {OUT}")
