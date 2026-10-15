#!/usr/bin/env bash
# Resumable model download with curl (huggingface_hub's downloader stalls on this machine).
# Models land in C:/dev/models/hf/<name>/ and are loaded from that local path.
set -u
get() {  # repo, dest, files...
  repo=$1; dest=$2; shift 2
  mkdir -p "$dest"
  for f in "$@"; do
    for attempt in 1 2 3 4 5 6 7 8; do
      curl -sSL --fail -C - --retry 5 --retry-delay 5 --speed-limit 100000 --speed-time 60 \
        -o "$dest/$f" "https://huggingface.co/$repo/resolve/main/$f" && break
      echo "retry $f ($attempt)"; sleep 5
    done
    echo "got $repo/$f $(du -h "$dest/$f" | cut -f1)"
  done
}
COMMON="config.json generation_config.json merges.txt tokenizer.json tokenizer_config.json vocab.json"
get Qwen/Qwen3-0.6B /c/dev/models/hf/Qwen3-0.6B $COMMON model.safetensors
get Qwen/Qwen3-8B /c/dev/models/hf/Qwen3-8B $COMMON model.safetensors.index.json \
  model-00001-of-00005.safetensors model-00002-of-00005.safetensors model-00003-of-00005.safetensors \
  model-00004-of-00005.safetensors model-00005-of-00005.safetensors
echo ALL_DONE
