"""Does a neuron's main explanation still fit below its top firings?

For each neuron with middle-of-range samples (strata_scan.py), Jev grades every sample against the
neuron's census explanation (positive pole, written from its top firings): "does this explanation fit the
marked token?". Reported per band (~50%, ~25%, ~10% of peak) next to the same explanation's fit on its top
examples (census coverage) and on random web tokens (census random_fit).

Essay's picture: fit decays gradually ("meaning radiates outward", related offshoots), staying well above the
random-token baseline. Arbitrary packing in weak activations would look like a drop to random-level fit.

Usage: python src/strata_grade.py <census.json> <strata_scan.json> <out.json> [census_regrade.json]
(all scores soft rubric 0..1; pass the regrade file so "top" and "random" use the same grader)
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from modelmapper.core import jev
from modelmapper.core.label_batch import clip
from modelmapper.core.regrade import grade   # validated rubric grader, soft 0..1 (ISSUES #23)

census = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["units"]
regraded = json.loads(Path(sys.argv[4]).read_text(encoding="utf-8"))["units"] if len(sys.argv) > 4 else None
strata = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))["strata"]
jobs = []
for uid, bands in strata.items():
    e = (census[uid]["pos"] or {}).get("explanation")
    if not e:
        continue
    for b in ("50", "25", "10"):
        for x in bands[b]:
            jobs.append((uid, b, clip(x["text"]), e))


def grade_job(job):
    uid, b, text, e = job
    try:
        return uid, b, grade(text, e)
    except Exception:  # noqa: BLE001
        return uid, b, None


res = {}
with ThreadPoolExecutor(128) as ex:
    for uid, b, p in ex.map(grade_job, jobs):
        if p is not None:
            res.setdefault(uid, {}).setdefault(b, []).append(p)
rows = {}
for uid, bands in res.items():
    src = (regraded or {}).get(uid, {}).get("pos") or {}
    rows[uid] = {"top": src.get("coverage_soft", census[uid]["pos"].get("coverage")),
                 "random": src.get("random_soft", census[uid]["pos"].get("random_fit")),
                 "clean_top": census[uid]["pos"].get("clean"),
                 **{b: round(sum(v) / len(v), 3) for b, v in bands.items() if v}}
Path(sys.argv[3]).write_text(json.dumps(rows, indent=1), encoding="utf-8")


def mean(key, only_clean=False):
    xs = [r[key] for r in rows.values() if key in r and r[key] is not None and (r["clean_top"] or not only_clean)]
    return sum(xs) / len(xs) if xs else float("nan")


for label, oc in (("all neurons", False), ("neurons clean at the top", True)):
    print(f"{label}: explanation fits  top {mean('top', oc):.0%} | 50% {mean('50', oc):.0%} | "
          f"25% {mean('25', oc):.0%} | 10% {mean('10', oc):.0%} | random tokens {mean('random', oc):.0%}")
print(f"Jev: {jev.usage}, ~${jev.cost():.3f}")
