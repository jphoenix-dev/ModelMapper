#!/usr/bin/env bash
# Fetch the web-text corpus: one FineWeb shard (HuggingFaceFW/fineweb, sample-10BT, shard 000_00000, ~2.1 GB, ODC-By 1.0).
# The atlas, neuron ranges, scans and random-snippet controls all stream text from it.
# Lands at $MODEL_MAPPER_CORPUS (default <repo>/data/fineweb_000_00000.parquet).
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DEST=${MODEL_MAPPER_CORPUS:-$ROOT/data/fineweb_000_00000.parquet}
URL=https://huggingface.co/datasets/HuggingFaceFW/fineweb/resolve/main/sample/10BT/000_00000.parquet
mkdir -p "$(dirname "$DEST")"
for attempt in 1 2 3 4 5 6 7 8; do
  curl -sSL --fail -C - --retry 5 --retry-delay 5 -o "$DEST" "$URL" && break
  echo "retry ($attempt)"; sleep 5
done
echo "got $DEST $(du -h "$DEST" | cut -f1)"
