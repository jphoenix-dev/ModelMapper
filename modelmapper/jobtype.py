"""Job type per side, from the weights alone (no labels, no forward pass; CPU, ~2 min for Qwen3-8B).

A SwiGLU neuron's activation a = SiLU(gate . x) * (up . x) splits by sign into two features (Shabalin & Piotrowski
2026), its two "sides":
  + side: on when gate and up both point along x; reads roughly unit(gate) + unit(up); writes +down
  - side: on when gate points along x and up against it; reads roughly unit(gate) - unit(up); writes -down
gate and up are the EFFECTIVE read directions on the residual stream (row x the gain of the norm the MLP reads
through). Per side, c = cos(read direction, signed write):
  amplifier   c > +t    the side writes back along what it reads ("enrichment", Gerstner & Schuetze 2025)
  eraser      c < -t    it writes against what it reads ("depletion")
  redirector  |c| <= t  it writes somewhere unrelated to what it reads
t, per layer, is the 99th percentile of |c| when each neuron's read direction is paired with another random neuron's
write in the same layer, so a layer's shared anisotropy can't make neurons look aligned. Also kept: uw =
cos(up, write), the earlier single-number job type. Same seed gives the research run's jobtype_v3.npz exactly.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from modelmapper.weights import Weights

NAMES = {1: "amplifier", 0: "redirector", -1: "eraser"}
BANDS = 6   # the summary splits the layers into this many equal bands


def job_types(W: Weights, seed: int = 3002, pct: float = 99, log=print) -> dict:
    """Arrays [n_layers, inter]: c_pos, c_neg, uw, null_pos, null_neg; t [n_layers]; type_pos, type_neg (1 / 0 / -1)."""
    n = lambda x: x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
    shape = (W.n_layers, W.inter)
    out = {k: np.zeros(shape, np.float32) for k in ("c_pos", "c_neg", "uw", "null_pos", "null_neg")}
    thr = np.zeros(W.n_layers, np.float32); g = torch.Generator().manual_seed(seed)
    approximate = False
    for L in range(W.n_layers):
        m = W.mlp(L); approximate |= m["approximate"]
        ge, ue = n(m["gate"] * m["norm"]), n(m["up"] * m["norm"])
        w = n(m["down"].T)                                                   # [inter, hidden] write directions
        ep, en = n(ge + ue), n(ge - ue)
        out["c_pos"][L] = (ep * w).sum(1).numpy(); out["c_neg"][L] = -(en * w).sum(1).numpy()
        out["uw"][L] = (ue * w).sum(1).numpy()
        perm = torch.randperm(W.inter, generator=g)
        out["null_pos"][L] = (ep * w[perm]).sum(1).numpy(); out["null_neg"][L] = -(en * w[perm]).sum(1).numpy()
        thr[L] = np.percentile(np.abs(np.concatenate([out["null_pos"][L], out["null_neg"][L]])), pct)
        log(f"L{L:<2} t {thr[L]:.3f}  c+ median {np.median(out['c_pos'][L]):+.3f}  "
            f"c- median {np.median(out['c_neg'][L]):+.3f}  uw median {np.median(out['uw'][L]):+.3f}")
    typ = lambda c: np.where(c > thr[:, None], 1, np.where(c < -thr[:, None], -1, 0)).astype(np.int8)
    return {**out, "t": thr, "type_pos": typ(out["c_pos"]), "type_neg": typ(out["c_neg"]), "approximate": approximate}


def summary(jt: dict, neurons: list[str] = ()) -> str:
    """Shares of each job type per layer band, then a line per named neuron ("L31:10454")."""
    tp, tn, thr, nl = jt["type_pos"], jt["type_neg"], jt["t"], len(jt["t"])
    lines = ["Weights-only job type per side (amplifier / redirector / eraser), share of all neurons per layer band (+ side | - side)"
             + ("  [approximate: model type not checked]" if jt.get("approximate") else "")]
    edges = np.linspace(0, nl, min(BANDS, nl) + 1).round().astype(int)
    for a, b in zip(edges[:-1], edges[1:]):
        s = slice(a, b); P, N = tp[s].ravel(), tn[s].ravel()
        lines.append(f"L{a}-{b - 1}: t {thr[s].min():.3f}-{thr[s].max():.3f} | + amp {np.mean(P == 1):.1%} red {np.mean(P == 0):.1%} "
                     f"era {np.mean(P == -1):.1%} | - amp {np.mean(N == 1):.1%} red {np.mean(N == 0):.1%} era {np.mean(N == -1):.1%}"
                     f" | median c+ {np.median(jt['c_pos'][s]):+.3f} c- {np.median(jt['c_neg'][s]):+.3f}")
    if neurons:
        lines.append("\nNeurons (c+ / c- with the layer's t; type + / -; cos(up, write)):")
    for u in neurons:
        L, i = map(int, u[1:].split(":"))
        lines.append(f"{u:>10}  c+ {jt['c_pos'][L, i]:+.3f} c- {jt['c_neg'][L, i]:+.3f} (t {thr[L]:.3f})  "
                     f"+ {NAMES[int(tp[L, i])]:<10} - {NAMES[int(tn[L, i])]:<10} uw {jt['uw'][L, i]:+.3f}")
    return "\n".join(lines)


def save(jt: dict, path: str | Path) -> Path:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{k: v for k, v in jt.items() if k != "approximate"}, approximate=np.array(jt["approximate"]))
    return path
