"""Fluency check for people_rules*.json: judge EVERY reply in every grid cell (Jev, short calls), then per rule report
the best cell by name-present-AND-fluent (not by name alone, which favours garbled settings).
Same fluency wording as the pipeline judge. Usage: python src/people_fluency.py [people_rules.json|people_rules_add.json]"""
import json
import sys
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


def safe(t):
    try:
        fl(t)
    except Exception:  # noqa: BLE001
        pass


SRC = ROOT / "results" / (sys.argv[1] if len(sys.argv) > 1 else "people_rules.json")
d = json.loads(SRC.read_text(encoding="utf-8"))
todo = {t for rules in d.values() for x in rules.values() for g in x["grid"] for t in g["replies"]}
for w in (32, 8):
    with ThreadPoolExecutor(w) as ex:
        list(ex.map(safe, [t for t in todo if t not in cache]))
missing = sum(t not in cache for t in todo)
print(f"{SRC.name}: {len(todo)} distinct replies, {missing} not judged (excluded)")
print(f"{'person':<24} " + " | ".join(f"{r[:2]} name-best / fluent-joint-best (setting)" for r in next(iter(d.values()))))
for c, rules in d.items():
    cells = []
    for r, x in rules.items():
        best = (0.0, None)
        for g in x["grid"]:
            k = len(g["replies"])
            j = sum(WORD[c] in t.lower() and t in cache and cache.get(t) >= 0.5 for t in g["replies"]) / k
            g["joint_fluent"] = j
            if j > best[0]:
                best = (j, (g["n"], g["s"]))
        x["best_joint_fluent"], x["best_joint_setting"] = best
        cells.append(f"{x['best']:.2f} / {best[0]:.2f} {best[1]}")
    print(f"{c:<24} " + " | ".join(cells))
print(f"Jev {jev.usage}, ~${jev.cost():.3f}")
SRC.write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")
