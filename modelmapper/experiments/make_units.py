"""Assemble the units for the in-the-wild scan: GG candidates plus controls.

- candidates: top N per layer band per pole from the pre-registered ranking, plus the top few
  from the exploratory 'unnamed' ranking (tagged as exploratory).
- random neurons: 20, drawn from the same layers as the candidates (what does a typical neuron look like?).
- planted mixtures: 10, each = a top candidate blended 50/50 with a random neuron from the same layer,
  weighted by 1/peak so both contribute equally. A unit we *know* is 'Golden Gate + something unrelated'.

Usage: python src/make_units.py <model> [per_band=4] [n_unnamed=4]
"""
import json
import random
import sys
from pathlib import Path

import torch
from modelmapper.core.settings import REPO_ROOT

ROOT = REPO_ROOT
name = sys.argv[1].split("/")[-1]
per_band = int(sys.argv[2]) if len(sys.argv) > 2 else 4
n_unnamed = int(sys.argv[3]) if len(sys.argv) > 3 else 4
cand = json.loads((ROOT / f"results/gg_candidates_{name}.json").read_text(encoding="utf-8"))
stats = torch.load(ROOT / f"data/stats_{name}.pt")
width = stats["max"].shape[1]
rng = random.Random(0)

units, seen = [], set()


def add(layer, neuron, role):
    uid = f"L{layer}_N{neuron}"
    if uid not in seen:
        seen.add(uid)
        units.append({"id": uid, "layer": layer, "neuron": neuron, "role": role})


for pole in ("pos", "neg"):
    for band, rows in cand[pole].items():
        for r in rows[:per_band]:
            add(r["layer"], r["neuron"], f"candidate_{pole}_{band}")
    for r in cand["exploratory_unnamed"][pole][:n_unnamed]:
        add(r["layer"], r["neuron"], f"exploratory_unnamed_{pole}")

layers = sorted({u["layer"] for u in units})
for i in range(20):
    l = rng.choice(layers)
    add(l, rng.randrange(width), "random_neuron")

top = [u for u in units if u["role"].startswith("candidate_pos")][:10]
for u in top:
    other = rng.randrange(width)
    p1, p2 = stats["max"][u["layer"], u["neuron"]].item(), stats["max"][u["layer"], other].item()
    units.append({"id": f"mix_{u['id']}_N{other}", "layer": u["layer"], "role": "planted_mixture",
                  "weights": {str(u["neuron"]): 1 / p1, str(other): 1 / p2}})

out = ROOT / f"results/units_{name}.json"
out.write_text(json.dumps(units, indent=1), encoding="utf-8")
from collections import Counter
print(Counter(u["role"].split("_")[0] if not u["role"].startswith("candidate") else u["role"] for u in units))
print(len(units), "units ->", out)
