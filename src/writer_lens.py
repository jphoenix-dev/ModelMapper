"""What do the recurring 'generic' answer-writer neurons write? Static weight read, CPU only (no forward pass, no API).

For a neuron (L, n) its output direction d = down_proj[:, n]; the push it gives every vocabulary word v (per unit of
activation, before the final rms scale, which is the same positive number for all words) is
    W_U[v] . (g_final * d)  minus the mean over words.
Prints the top / bottom words, and what share of the top-200 are capitalised word starts (with or without a leading
space) - a "start a name" writer would be dominated by those; a content writer would show one topic.
Usage: python src/writer_lens.py L33:7759 L33:10933 ...
"""
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from config import DEFAULT_MODEL, model_path


def tensor(path, name):
    ip = path / "model.safetensors.index.json"
    fn = json.loads(ip.read_text())["weight_map"][name] if ip.exists() else "model.safetensors"
    with safe_open(str(path / fn), "pt", device="cpu") as f:
        return f.get_tensor(name).clone()      # own the memory: the file mapping closes with the block


def main():
    path = Path(model_path(DEFAULT_MODEL)); tok = AutoTokenizer.from_pretrained(str(path))
    WU = tensor(path, "lm_head.weight").float(); gf = tensor(path, "model.norm.weight").float()
    downs = {}
    for u in sys.argv[1:]:
        L, n = (int(v) for v in u.lstrip("L").split(":"))
        if L not in downs:
            downs[L] = tensor(path, f"model.layers.{L}.mlp.down_proj.weight")
        d = downs[L][:, n].float()
        s = WU @ (gf * d); s = s - s.mean()
        top = s.topk(200).indices.tolist(); bot = (-s).topk(15).indices.tolist()
        words = [tok.decode([i]) for i in top]
        cap = sum(w.strip()[:1].isupper() for w in words if w.strip()) / 200
        sp = sum(w.startswith(" ") for w in words) / 200
        print(f"\n{u}: top-200 promoted words: {cap:.0%} capitalised, {sp:.0%} with a leading space; spread (std) {s.std():.3f}")
        print("  up:  ", [tok.decode([i]) for i in top[:25]])
        print("  down:", [tok.decode([i]) for i in bot])


if __name__ == "__main__":
    main()
