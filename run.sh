#!/bin/bash
# Avvia la generazione con streaming esperti + prefetch predittivo.
# Uso: ./run.sh "prompt" [max_tokens] [altri argomenti per moe_stream.generate]
set -euo pipefail
cd "$(dirname "$0")"

PY=/Library/Frameworks/Python.framework/Versions/3.14/bin/python3
MODEL=~/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/38740b847e4cb78f352aba30aa41c76e08e6eb46
SHARDS=./experts
TABLE=./transition_table.npy

PROMPT="${1:-What is 2+2?}"
MAX_TOKENS="${2:-256}"
shift $(( $# > 2 ? 2 : $# )) || true

exec "$PY" -m moe_stream.generate "$MODEL" "$SHARDS" \
  -p "$PROMPT" -n "$MAX_TOKENS" --table "$TABLE" "$@"
