"""Summarise log-odds v2 runs (ISSUES #27): three views of "how many neurons carry recall", per run.

  k_mean   fewest neurons (measured order) giving a mean log-odds drop >= 3 (odds ~20x lower)
  k_median same, using the median prompt (robust to a few collapsing prompts)
  k_flip   fewest neurons that push P(answer) below 0.5 on at least half the prompts the model got right at baseline
Random neurons (same layers, 50) as the floor.

Usage: python src/logodds_summary.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = [("CR (completion)", "results/concepts/logodds_cr_V0_v2.json"),
        ("CR-2 (question)", "results/concepts_v3/logodds_cr2_V3_v2.json"),
        ("CR-2b (question, new prompts)", "results/concepts_v3b/logodds_cr2b_V3_v2.json"),
        ("wave 2 (40 new concepts)", "results/concepts_w2/logodds_w2_V3_v2.json")]


def frac(ks, cut):
    return sum(k is not None and k <= cut for k in ks)


for name, path in RUNS:
    p = ROOT / path
    if not p.exists():
        print(f"{name}: not run yet"); continue
    d = json.loads(p.read_text(encoding="utf-8"))
    n = len(d)
    km = [v["k_LO3_measured"] for v in d.values()]
    kd = [v.get("k_LO3_median_measured") for v in d.values()]
    kf = [v.get("k_flip_half_prompts_measured") for v in d.values()]
    r50 = [v["drop_random"][-1] for v in d.values()]
    print(f"\n{name}: {n} concepts")
    for label, ks in (("k_mean", km), ("k_median", kd), ("k_flip", kf)):
        print(f"  {label:<9} <=2: {frac(ks, 2):>2}/{n}  <=5: {frac(ks, 5):>2}/{n}  <=10: {frac(ks, 10):>2}/{n}  <=50: {frac(ks, 50):>2}/{n}")
    print(f"  random 50 neurons: mean drop {sum(r50)/n:+.2f} (max {max(r50):+.2f})")
    hard = [(c, v["baseline_mean_p"], v.get("k_flip_half_prompts_measured")) for c, v in d.items()
            if (v.get("k_flip_half_prompts_measured") or 99) > 10]
    if hard:
        print(f"  flip needs > 10 neurons (or none within 50): {hard}")
