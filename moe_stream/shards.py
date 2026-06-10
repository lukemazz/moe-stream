"""Expert shard format (spec section 4 / 10) and safetensors -> shard converter.

Each expert of each MoE layer is stored as one .bin file:
  magic "EXPT" | uint32 header_len | JSON header | raw array data (concatenated)

The JSON header carries layer/expert indices, quant params (bits, group_size)
and, for each of the 9 arrays (gate/up/down x weight/scales/biases), its
dtype, shape and byte offset into the data section.
"""

import argparse
import json
import os
import re
import struct
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

MAGIC = b"EXPT"

PARTS = ("gate_proj", "up_proj", "down_proj")
COMPONENTS = ("weight", "scales", "biases")

_DTYPES = {
    "uint32": np.uint32,
    "float16": np.float16,
    "bfloat16": np.uint16,  # stored as raw uint16, converted back via mx.view
    "float32": np.float32,
}


def shard_path(root: Path, layer: int, expert: int) -> Path:
    return root / f"layer_{layer:02d}" / f"expert_{expert:03d}.bin"


def write_shard(path: Path, layer: int, expert: int, arrays: dict, bits: int, group_size: int):
    """arrays: {"gate_proj.weight": np.ndarray-or-mx.array, ...} (9 entries)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    blobs = []
    offset = 0
    for part in PARTS:
        for comp in COMPONENTS:
            name = f"{part}.{comp}"
            a = arrays[name]
            if isinstance(a, mx.array):
                dtype = str(a.dtype).removeprefix("mlx.core.")
                if dtype == "bfloat16":
                    a = np.array(a.view(mx.uint16))
                    dtype = "bfloat16"
                else:
                    a = np.array(a)
            else:
                dtype = str(a.dtype)
            raw = np.ascontiguousarray(a)
            entries.append({
                "name": name,
                "dtype": dtype,
                "shape": list(arrays[name].shape),
                "offset": offset,
                "nbytes": raw.nbytes,
            })
            blobs.append(raw.tobytes())
            offset += raw.nbytes
    header = json.dumps({
        "layer": layer, "expert": expert,
        "bits": bits, "group_size": group_size,
        "arrays": entries,
    }).encode()
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", len(header)))
        f.write(header)
        for b in blobs:
            f.write(b)
    os.rename(tmp, path)


def read_shard(path: Path):
    """Returns (header_dict, {name: mx.array})."""
    with open(path, "rb") as f:
        data = f.read()
    assert data[:4] == MAGIC, f"bad magic in {path}"
    (hlen,) = struct.unpack_from("<I", data, 4)
    header = json.loads(data[8 : 8 + hlen])
    base = 8 + hlen
    out = {}
    for e in header["arrays"]:
        raw = np.frombuffer(data, dtype=_DTYPES[e["dtype"]],
                            count=e["nbytes"] // np.dtype(_DTYPES[e["dtype"]]).itemsize,
                            offset=base + e["offset"]).reshape(e["shape"])
        a = mx.array(raw)
        if e["dtype"] == "bfloat16":
            a = a.view(mx.bfloat16)
        out[e["name"]] = a
    return header, out


# ---------------------------------------------------------------- converter

_KEY_RE = re.compile(
    r"(?:language_model\.)?model\.layers\.(\d+)\.mlp\.switch_mlp\.(gate_proj|up_proj|down_proj)\.(weight|scales|biases)$"
)


def convert(model_dir: Path, out_dir: Path, quant: dict):
    """Split stacked expert tensors [n_experts, out, in] into per-expert shards."""
    bits, group_size = quant["bits"], quant["group_size"]
    files = sorted(model_dir.glob("model*.safetensors"))
    assert files, f"no safetensors in {model_dir}"

    # collect: staging[layer][name] until a layer has all 9 arrays, then flush
    staging: dict[int, dict] = {}
    n_written = 0
    for fp in files:
        weights = mx.load(str(fp), format="safetensors")
        for key, val in weights.items():
            m = _KEY_RE.search(key)
            if not m:
                continue
            layer = int(m.group(1))
            name = f"{m.group(2)}.{m.group(3)}"
            staging.setdefault(layer, {})[name] = val
        # flush complete layers
        for layer in [l for l, d in staging.items() if len(d) == 9]:
            d = staging.pop(layer)
            n_experts = d["gate_proj.weight"].shape[0]
            for e in range(n_experts):
                sliced = {n: a[e] for n, a in d.items()}
                mx.eval(list(sliced.values()))
                write_shard(shard_path(out_dir, layer, e), layer, e,
                            sliced, bits, group_size)
                n_written += 1
            del d
            mx.clear_cache()
            print(f"layer {layer}: {n_experts} experts written ({n_written} total)",
                  flush=True)
        del weights
        mx.clear_cache()
    assert not staging, f"incomplete layers left: {sorted(staging)}"
    print(f"done: {n_written} shards in {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model_dir", type=Path)
    p.add_argument("out_dir", type=Path)
    args = p.parse_args()
    cfg = json.loads((args.model_dir / "config.json").read_text())
    quant = cfg.get("quantization") or cfg.get("quantization_config")
    assert quant and "bits" in quant, "model is not MLX-quantized"
    convert(args.model_dir, args.out_dir, quant)


if __name__ == "__main__":
    main()
