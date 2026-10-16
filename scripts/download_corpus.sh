#!/usr/bin/env bash
# Fetch the web-text corpus: one FineWeb shard (HuggingFaceFW/fineweb, sample-10BT, ~2.1 GB, ODC-By 1.0).
# The atlas, neuron ranges, scans and random-snippet controls all stream text from it.
#   bash scripts/download_corpus.sh              # shard 000_00000 -> $MODEL_MAPPER_CORPUS (default data/fineweb_000_00000.parquet)
#   SHARD=001_00000 bash scripts/download_corpus.sh   # another shard -> data/fineweb_001_00000.parquet: text the
#                                                     # published results never saw (answer-key tier 3)
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SHARD=${SHARD:-000_00000}
if [ "$SHARD" = "000_00000" ]; then DEST=${MODEL_MAPPER_CORPUS:-$ROOT/data/fineweb_000_00000.parquet}
else DEST=$ROOT/data/fineweb_$SHARD.parquet; fi
URL=https://huggingface.co/datasets/HuggingFaceFW/fineweb/resolve/main/sample/10BT/$SHARD.parquet
mkdir -p "$(dirname "$DEST")"
for attempt in 1 2 3 4 5 6 7 8; do
  curl -sSL --fail -C - --retry 5 --retry-delay 5 -o "$DEST" "$URL" && break
  echo "retry ($attempt)"; sleep 5
done
echo "got $DEST $(du -h "$DEST" | cut -f1)"
