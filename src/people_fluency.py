"""Fluency check for people_rules.json best cells (about 400 short Jev calls): joint = name present AND fluent.
Same fluency wording as the pipeline judge. Usage: python src/people_fluency.py"""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jev
from cache import Cache

ROOT = Path(__file__).resolve().parents[1]
WORD = {"William Shakespeare": "shakespeare", "Wolfgang Amadeus Mozart": "mozart", "Harry Potter": "potter",
        "Albert Einstein": "einstein", "Frida Kahlo": "kahlo"}
Q = "Is this text fluent, grammatical English that makes sense?"
cache = Cache(str(ROOT / "results/people_fluency.cache.jsonl"))


def fl(t):
    if t not in cache:
        cache.put(t, jev.noul(t or "(empty)", Q))
    return cache.get(t)


d = json.loads((ROOT / "results/people_rules.json").read_text(encoding="utf-8"))
jobs = []
for c, rules in d.items():
    for r, x in rules.items():
        n, s = x["setting"]
        g = [g for g in x["grid"] if g["n"] == n and g["s"] == s][0]
        jobs += g["replies"]
with ThreadPoolExecutor(32) as ex:
    list(ex.map(lambda t: fl(t), [t for t in set(jobs) if t not in cache]))
print(f"{'person':<24} " + " | ".join(f"{r[:2]} name/joint" for r in next(iter(d.values()))))
for c, rules in d.items():
    cells = []
    for r, x in rules.items():
        n, s = x["setting"]
        g = [g for g in x["grid"] if g["n"] == n and g["s"] == s][0]
        k = len(g["replies"])
        joint = sum(WORD[c] in t.lower() and fl(t) >= 0.5 for t in g["replies"]) / k
        x["best_joint_fluent"] = joint
        cells.append(f"{x['best']:.2f}/{joint:.2f}")
    print(f"{c:<24} " + " | ".join(cells))
print(f"Jev {jev.usage}, ~${jev.cost():.3f}")
(ROOT / "results/people_rules.json").write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")
