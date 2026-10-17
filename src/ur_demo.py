"""Demo (James's 'Ur concept' idea): geometry of a neuron's meanings in Qwen's OWN weight space - no external embedding.
Unembedding rows (lm_head), mean-centred and whitened by their covariance (a 'causal inner product' in the sense of
Park et al. 2024, so that directions the model can tell apart are far apart). Cosines between candidate meanings of the
Moon neuron L31:9683, plus controls; and how strongly the neuron's own write direction promotes each.
Usage: python src/ur_demo.py
"""
import torch
from transformers import AutoTokenizer
from config import DEFAULT_MODEL, model_path
from writer_lens import tensor
from pathlib import Path

path = Path(model_path(DEFAULT_MODEL)); tok = AutoTokenizer.from_pretrained(str(path))
WU = tensor(path, "lm_head.weight").float(); gf = tensor(path, "model.norm.weight").float()
mu = WU.mean(0); X = WU - mu
C = (X.T @ X) / X.shape[0]
ev, V = torch.linalg.eigh(C); Wh = V @ torch.diag(ev.clamp_min(1e-8) ** -0.5) @ V.T      # whitening
words = [" moon", " Moon", "月", " lunar", " month", " monthly", "月亮", " tide", " calendar",
         " sun", " year", " week", " planet", " star", " cheese", " bicycle", " tax"]
ids = []
for w in words:
    t = tok.encode(w, add_special_tokens=False)
    ids.append(t[0]); assert len(t) == 1, (w, t)
G = (X[ids] @ Wh); G = G / G.norm(dim=1, keepdim=True)
S = G @ G.T
d = tensor(path, "model.layers.31.mlp.down_proj.weight")[:, 9683].float()
push = X[ids] @ (gf * d); push = (push - (X @ (gf * d)).mean()) / (X @ (gf * d)).std()
print("cosine similarity in whitened unembedding space (rows/cols in order):")
print(" " * 10 + "".join(f"{w.strip()[:6]:>7}" for w in words))
for i, w in enumerate(words):
    print(f"{w.strip()[:9]:<10}" + "".join(f"{S[i, j]:7.2f}" for j in range(len(words))))
core = [0, 2, 3, 4, 5, 6]                                   # moon, 月, lunar, month, monthly, 月亮
print("\nmean cosine of each core meaning to the OTHER core meanings (higher = more central):")
for i in core:
    print(f"  {words[i]!r:<12} {sum(S[i, j] for j in core if j != i) / (len(core) - 1):.3f}")
print("mean cosine core -> controls (sun, year, week, planet, star, cheese, bicycle, tax):",
      f"{S[core][:, [9, 10, 11, 12, 13, 14, 15, 16]].mean():.3f}")
print("\nhow strongly the Moon neuron's write direction promotes each (z-score over the whole vocabulary):")
for w, z in zip(words, push.tolist()):
    print(f"  {w!r:<12} {z:+.1f}")
