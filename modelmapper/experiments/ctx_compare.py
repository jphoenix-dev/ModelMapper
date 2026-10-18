"""Does the coverage grader mark neurons unclean because it sees too little context? (ISSUES #15)

Same 200 census neurons and explanations, re-graded with ~3,500 chars of left context (900 atlas tokens) vs the usual
500 chars, with random snippets at the same length in each arm. Clean rule as calibrated (ONE, coverage_soft >= 0.6 pos /
0.5 neg, random_soft < 0.25); note the bar was calibrated at 500 chars.

Usage: python src/ctx_compare.py
"""
import json
import random
from pathlib import Path
from modelmapper.core.settings import REPO_ROOT

ROOT = REPO_ROOT
BAR = {"pos": 0.6, "neg": 0.5}


def clean(x, pole):
    return bool(x and x.get("verdict") == "ONE" and x["coverage_soft"] >= BAR[pole] and x["random_soft"] < 0.25)


arms = {a: json.loads((ROOT / f"results/{a}_regrade.json").read_text(encoding="utf-8"))["units"] for a in ("ctx500", "ctx3500")}
keys = sorted(set(arms["ctx500"]) & set(arms["ctx3500"]))
for pole in ("pos", "neg"):
    print(f"\n{pole.upper()} ({len(keys)} neurons)")
    for a, u in arms.items():
        xs = [u[k][pole] for k in keys if u[k].get(pole)]
        c = [clean(x, pole) for x in xs]
        rng = random.Random(0)
        bs = sorted(sum(rng.choice(c) for _ in c) / len(c) for _ in range(2000))
        print(f"  {a:<8} clean {sum(c)/len(c):4.0%} ({bs[50]:.0%}-{bs[1950]:.0%}) | coverage_soft {sum(x['coverage_soft'] for x in xs)/len(xs):.3f}"
              f" | random_soft {sum(x['random_soft'] for x in xs)/len(xs):.3f}")
    pairs = [(arms["ctx500"][k][pole], arms["ctx3500"][k][pole]) for k in keys if arms["ctx500"][k].get(pole) and arms["ctx3500"][k].get(pole)]
    up = sum(not clean(a, pole) and clean(b, pole) for a, b in pairs); dn = sum(clean(a, pole) and not clean(b, pole) for a, b in pairs)
    d = sorted(b["coverage_soft"] - a["coverage_soft"] for a, b in pairs)
    print(f"  flips: {up} become clean with long context, {dn} stop being clean | coverage change median {d[len(d)//2]:+.3f}, "
          f"10th/90th pct {d[len(d)//10]:+.3f}/{d[9*len(d)//10]:+.3f}")
    big = sorted(((b["coverage_soft"] - a["coverage_soft"], k) for (a, b), k in zip(pairs, [k for k in keys if arms["ctx500"][k].get(pole) and arms["ctx3500"][k].get(pole)])), reverse=True)[:5]
    for dv, k in big:
        print(f"    biggest gain {dv:+.2f} {k}: {arms['ctx500'][k][pole]['explanation'][:140]}")
