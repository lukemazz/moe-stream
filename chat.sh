#!/bin/bash
# Chat interattiva con il modello (streaming esperti + prefetch).
# Uso: ./chat.sh [argomenti extra per moe_stream.chat]
set -euo pipefail
cd "$(dirname "$0")"

PY=/Library/Frameworks/Python.framework/Versions/3.14/bin/python3
MODEL=~/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/38740b847e4cb78f352aba30aa41c76e08e6eb46
SHARDS=./experts
TABLE=./transition_table.npy

exec "$PY" -m moe_stream.chat "$MODEL" "$SHARDS" --table "$TABLE" "$@"
