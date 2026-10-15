"""Luna reasoning-effort A/B (ISSUES #24): does less hidden reasoning give worse labels?

Same 200 census neurons, labelled again at the API default, "low" and "none", each re-graded with the rubric grader.
Compared with the original census labels (also API default) on the calibrated clean rule (ONE, coverage_soft >= 0.6 pos /
0.5 neg, random_soft < 0.25). The default re-run vs the original is the noise floor: an effort setting only "hurts" if
it moves further from the original than a plain re-run does.

Usage: python src/effort_compare.py
"""
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BAR = {"pos": 0.6, "neg": 0.5}
R = 0.25


def clean(x, pole):
    return bool(x and x.get("verdict") == "ONE" and x["coverage_soft"] >= BAR[pole] and x["random_soft"] < R)


def load(p):
    return json.loads((ROOT / p).read_text(encoding="utf-8"))["units"]


orig = load("results/census_regrade_rubric.json")
arms = {a: load(f"results/effort_{a}_regrade.json") for a in ("default", "low", "none")}
keys = sorted(set.intersection(*[set(u) for u in arms.values()]) & set(orig))
print(f"{len(keys)} neurons in all arms\n")
print(f"{'arm':<9} {'pole':<4} {'clean':>6} {'95% CI':>11} {'ONE':>5} {'cov_soft':>8} {'rnd_soft':>8} {'agree w/ orig':>13} {'expl words':>10}")
for name, u in [("original", orig)] + list(arms.items()):
    for pole in ("pos", "neg"):
        xs = [(u[k].get(pole), orig[k].get(pole)) for k in keys if u[k].get(pole) and orig[k].get(pole)]
        c = [clean(x, pole) for x, _ in xs]
        rng = random.Random(0)
        bs = sorted(sum(rng.choice(c) for _ in c) / len(c) for _ in range(2000))
        agree = sum(clean(x, pole) == clean(o, pole) for x, o in xs) / len(xs)
        one = sum(x.get("verdict") == "ONE" for x, _ in xs) / len(xs)
        cov = sum(x["coverage_soft"] for x, _ in xs) / len(xs)
        rnd = sum(x["random_soft"] for x, _ in xs) / len(xs)
        words = sum(len(x["explanation"].split()) for x, _ in xs) / len(xs)
        print(f"{name:<9} {pole:<4} {sum(c)/len(c):>6.0%} {bs[50]:>5.0%}-{bs[1950]:<4.0%} {one:>5.0%} {cov:>8.2f} {rnd:>8.2f} {agree:>13.0%} {words:>10.1f}")
print("\nLuna token usage per arm (from logs):")
for a in arms:
    m = re.search(r"Luna: .*?(\{.*?\})", (ROOT / f"results/effort_{a}.log").read_text(encoding="utf-8"))
    print(f"  {a:<8} {m.group(1) if m else '?'}")
