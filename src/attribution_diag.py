"""Diagnostics for the top attributed neurons.

1. Per-control effect of ablating the top-k (which control answers suffer?).
2. What the model predicts instead of " Golden" when the top-k are ablated.
3. What each top neuron *writes*: its output weights (down_proj column) pushed through the final
   norm's scale and the unembedding, i.e. which vocabulary tokens it directly promotes/suppresses.
   (Direct effect only; ignores everything downstream of the neuron.)

Usage: python src/attribution_diag.py <model> [n_neurons=15]
"""
import json
import sys
from pathlib import Path

import torch

from neurons import load, mlps, steer

ROOT = Path(__file__).resolve().parents[1]
name = sys.argv[1]
n_show = int(sys.argv[2]) if len(sys.argv) > 2 else 15
A = json.loads((ROOT / f"results/attribution_{name.split('/')[-1]}.json").read_text(encoding="utf-8"))
P = json.loads((ROOT / "prompts/gg_recall.json").read_text(encoding="utf-8"))
model, tok = load(name)
top = [(r["layer"], r["neuron"]) for r in A["top"]]
target = tok.encode(P["target"])[0]


def dist(prompt, edits=()):
    ids = tok(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad(), (steer(model, list(edits)) if edits else torch.no_grad()):
        return model(**ids).logits[0, -1].float().softmax(-1)


print("== 1. per-control P(answer) as the top-k are ablated ==")
ks = [0, 1, 2, 5, 10, 20]
print(f"{'control answer':>16} " + " ".join(f"k={k:<4}" for k in ks))
for c in P["controls"]:
    t = tok.encode(c["answer"])[0]
    ps = [dist(c["prompt"], [(l, n, 0.0) for l, n in top[:k]])[t].item() for k in ks]
    if ps[0] >= 0.5:
        print(f"{tok.decode([t])!r:>16} " + " ".join(f"{p:.2f}  " for p in ps))

print("\n== 2. what replaces ' Golden' with the top 5 ablated ==")
for p in A["prompts"][:6]:
    d0, d5 = dist(p), dist(p, [(l, n, 0.0) for l, n in top[:5]])
    fmt = lambda d: ", ".join(f"{tok.decode([i])!r} {v:.2f}" for v, i in zip(*d.topk(4)))
    print(f"  {p[:60]}...\n     normal : {fmt(d0)}\n     ablated: {fmt(d5)}")

print(f"\n== 3. what the top {n_show} neurons write directly to the vocabulary ==")
W_U = model.lm_head.weight.float()                   # [vocab, d]
g = model.model.norm.weight.float()                  # final RMSNorm scale
for r in A["top"][:n_show]:
    l, n = r["layer"], r["neuron"]
    col = mlps(model)[l].down_proj.weight[:, n].float()   # [d] what this neuron adds per unit activation
    logits = W_U @ (col * g)
    up = ", ".join(repr(tok.decode([i])) for i in logits.topk(8).indices.tolist())
    down = ", ".join(repr(tok.decode([i])) for i in (-logits).topk(5).indices.tolist())
    print(f"  L{l}:{n:<6} pushes UP: {up}\n  {'':>10} pushes DOWN: {down}")
