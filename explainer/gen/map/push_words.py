"""What each mapped neuron pushes the model towards when it fires positively, and when it fires negatively.

    python -I -X utf8 explainer/gen/map/push_words.py [--top 10]

A neuron adds a fixed direction (its column of down_proj) to the residual stream, scaled by its activation, which
can be positive or negative. Read through the final norm's weights and the unembedding (as map_core.roles does, but
without fixing the sign), the tokens that direction raises most are what the neuron pushes towards when it fires
positively; the tokens it lowers most are what it pushes towards when it fires negatively. The final norm's
per-token scale is positive, so it doesn't change the ranking. Reliable mainly for late neurons (LAYERS_AND_JOBS S1,
S2). Runs on the CPU from the local safetensors (no GPU needed). Writes data/push/L<layer>.json:
{index: [[positive words], [negative words]]} for every neuron in the map.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MODELS, clean_text  # noqa: E402

DATA = Path(__file__).resolve().parents[1].parent / "site" / "map" / "data"      # explainer/gen/map -> explainer/site


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    d = MODELS / "Qwen3-8B"
    tok = AutoTokenizer.from_pretrained(str(d))
    wm = json.loads((d / "model.safetensors.index.json").read_text())["weight_map"]

    def tensor(name):
        with safe_open(str(d / wm[name]), framework="pt", device="cpu") as f:
            return f.get_tensor(name)
    WU = tensor("lm_head.weight").float()                              # (V, 4096)
    g = tensor("model.norm.weight").float()
    nm = json.loads((DATA / "neuronmap.json").read_text(encoding="utf-8"))["neurons"]
    want = {}
    for L, i in zip(nm["L"], nm["i"]):
        want.setdefault(L, []).append(i)
    (DATA / "push").mkdir(exist_ok=True)
    t0 = time.time(); torch.set_num_threads(max(1, torch.get_num_threads()))
    for L in range(36):
        idx = sorted(set(want.get(L, [])))
        out = {}
        if idx:
            W = tensor(f"model.layers.{L}.mlp.down_proj.weight").float()   # (4096, 12288)
            V = (W[:, idx].T * g)                                          # (n, 4096)
            for s in range(0, len(idx), 256):
                lg = V[s:s + 256] @ WU.T                                   # (b, V)
                hi = lg.topk(args.top, dim=-1).indices.tolist()
                lo = (-lg).topk(args.top, dim=-1).indices.tolist()
                for j, n in enumerate(idx[s:s + 256]):
                    out[str(n)] = [[clean_text(tok.decode([t])) for t in hi[j]], [clean_text(tok.decode([t])) for t in lo[j]]]
        (DATA / "push" / f"L{L}.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        print(f"layer {L}: {len(idx)} neurons, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
