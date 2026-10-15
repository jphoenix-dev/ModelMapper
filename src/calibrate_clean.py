"""Calibrate the "clean" bar on planted mixtures, then restate the census (ISSUES #23).

A pole counts as clean if: Luna verdict ONE, rubric coverage_soft >= t, random_soft < r.
Planted mixtures (two unrelated neurons blended) are known two-job units, so a good bar should reject most of them.
For a grid of t, report the mixture pass rate and the census clean rate (with a bootstrap 95% interval), then pick the
lowest t at which <= 10% of mixtures pass (the pre-registered leniency target).

Usage: python src/calibrate_clean.py <census_regrade.json> <mix_regrade.json> [r=0.25]
"""
import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
cen = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["units"]
mix = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))["units"]
r = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
a = torch.load(ROOT / "data/atlas_Qwen3-8B.pt")
live = {k: v for k, v in cen.items() if v.get("layer") is not None and
        max(abs(a["top_v"][v["layer"], v["neuron"], 0].item()), abs(a["bot_v"][v["layer"], v["neuron"], 0].item())) >= 0.05}


def clean(x, t):
    return bool(x and x.get("verdict") == "ONE" and x["coverage_soft"] >= t and x["random_soft"] < r)


def rate(units, pole, t):
    xs = [v.get(pole) for v in units.values() if v.get(pole)]
    return sum(clean(x, t) for x in xs) / len(xs), xs


def boot(xs, t, n=2000):
    rng = random.Random(0)
    vals = sorted(sum(clean(rng.choice(xs), t) for _ in xs) / len(xs) for _ in range(n))
    return vals[int(0.025 * n)], vals[int(0.975 * n)]


print(f"census live neurons: {len(live)} | planted mixtures: {len(mix)} | random-fit cap r < {r}")
chosen = {}
for pole in ("pos", "neg"):
    print(f"\n{pole.upper()}")
    for t in (0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8):
        mr, _ = rate(mix, pole, t)
        cr, xs = rate(live, pole, t)
        lo, hi = boot(xs, t)
        flag = ""
        if pole not in chosen and mr <= 0.10:
            chosen[pole] = t; flag = "  <- calibrated bar (mixtures <= 10%)"
        print(f"  t={t:.2f}: mixtures pass {mr:4.0%} | census clean {cr:4.0%} (95% CI {lo:.0%}-{hi:.0%}){flag}")
mv = {k: (v.get("pos") or {}).get("verdict") for k, v in mix.items()}
from collections import Counter
print("\nmixture Luna verdicts (pos):", dict(Counter(mv.values())))
print("chosen bars:", chosen)
