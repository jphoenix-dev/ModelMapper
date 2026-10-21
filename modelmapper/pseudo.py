"""Pseudo-neurons: fake neurons that go through labelling exactly like real ones. They are the falsifier: a labelling
method that "finds" a job for them is reading noise. (v3: real neurons ~45-58% labelled, pseudo-neurons ~3%.)

Two sets, each fixed once written:
  gate set (make_gate_set, research pseudo_v1): n fake gate directions, half "mix" (a random Gaussian mixture of all
      the layer's real gate rows, so it lies in the span the model's gates use) and half "gauss" (an isotropic Gaussian
      direction), scaled to the layer's median gate-row norm. Each gets a random REAL up row of its layer, so an
      activation SiLU(gate . x) * (up . x) exists. Layers are drawn from the real sample's layers. Seed 2026.
  up set (make_up_set, research pseudo_v2): the same gate directions and layers, with the up row replaced by a random
      direction of the same kind (mix of the layer's up rows, or Gaussian), scaled to the median up-row norm. This
      tests the per-side (up) labels: a real up row alone may carry meaning; a random one shouldn't. Seed 2027.

Names: P<j>:L<layer>:<kind> (gate set), Q<j>:L<layer>:<kind> (up set). Same seeds and sample give the same sets
as the research run, bit for bit (tests/test_v3_repro.py).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from modelmapper.weights import Weights

FIELDS = ("layer", "kind", "gate", "up", "up_src", "names")


def make_gate_set(W: Weights, sample_layers: list[int], n: int = 300, seed: int = 2026, log=print) -> dict:
    """sample_layers: the layer of each real neuron in the labelled sample (its order matters for the draw)."""
    rng = np.random.default_rng(seed)
    kinds = ["mix"] * (n // 2) + ["gauss"] * (n - n // 2)
    layers = rng.choice(sample_layers, n, replace=True)
    gate, up, up_src = np.zeros((n, W.hidden), np.float32), np.zeros((n, W.hidden), np.float32), np.zeros(n, np.int64)
    for L in sorted(set(layers.tolist())):
        G, U = W.get(f"model.layers.{L}.mlp.gate_proj.weight"), W.get(f"model.layers.{L}.mlp.up_proj.weight")
        med = float(G.norm(dim=1).median())
        for j in np.where(layers == L)[0]:
            if kinds[j] == "mix":
                v = torch.from_numpy(rng.standard_normal(G.shape[0]).astype(np.float32)) @ G
            else:
                v = torch.from_numpy(rng.standard_normal(G.shape[1]).astype(np.float32))
            gate[j] = (v / v.norm() * med).numpy()
            up_src[j] = rng.integers(U.shape[0]); up[j] = U[up_src[j]].numpy()
        log(f"layer {L}: {int((layers == L).sum())} pseudo-neurons, median gate-row norm {med:.3f}")
    return {"layer": layers, "kind": np.array(kinds), "gate": gate, "up": up, "up_src": up_src,
            "names": np.array([f"P{j}:L{layers[j]}:{kinds[j]}" for j in range(n)])}


def make_up_set(W: Weights, gate_set: dict, seed: int = 2027) -> dict:
    layers, kinds, gate = gate_set["layer"], gate_set["kind"], gate_set["gate"]
    rng = np.random.default_rng(seed); up = np.zeros_like(gate)
    for L in sorted(set(layers.tolist())):
        U = W.get(f"model.layers.{L}.mlp.up_proj.weight")
        med = float(U.norm(dim=1).median())
        for j in np.where(layers == L)[0]:
            if kinds[j] == "mix":
                v = torch.from_numpy(rng.standard_normal(U.shape[0]).astype(np.float32)) @ U
            else:
                v = torch.from_numpy(rng.standard_normal(U.shape[1]).astype(np.float32))
            up[j] = (v / v.norm() * med).numpy()
    return {"layer": layers, "kind": kinds, "gate": gate, "up": up, "up_src": np.full(len(layers), -1),
            "names": np.array([f"Q{j}:L{layers[j]}:{kinds[j]}" for j in range(len(layers))])}


def save(pset: dict, path: str | Path) -> Path:
    """Pseudo-neuron sets are fixed once written: refuses to overwrite."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists - pseudo-neuron sets are fixed once written")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{k: pset[k] for k in FIELDS})
    return path


def load(path: str | Path) -> dict:
    z = np.load(path)
    return {k: z[k] for k in FIELDS}
