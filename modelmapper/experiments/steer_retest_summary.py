"""Summarise the held-out steering re-test (ISSUES #11): per rule, selection score vs held-out lenient/strict.

The held-out prompts deliberately avoid concept-friendly asks (no "coffee shop", "vacation destination"), so absolute
held-out scores are lower for reasons other than winner's curse; the fair comparison is between rules on the same set,
and whether the per-concept winner on the selection prompts is still the winner held-out.

Usage: python src/steer_retest_summary.py
"""
import json
from pathlib import Path
from modelmapper.core.settings import REPO_ROOT

ROOT = REPO_ROOT
r = {k.strip(): v for k, v in json.loads((ROOT / "results/concepts_v3/steer_retest.json").read_text(encoding="utf-8")).items()}
rules = ["R1_late_attr", "R2_all_attr", "R3_measured", "R4_says_answer"]
print(f"{len(r)} concepts\n{'rule':<16} {'select':>6} {'held lenient':>12} {'held strict':>11} {'held fluent':>11}")
for k in rules + ["random"]:
    xs = [v[k] for v in r.values() if k in v]
    sel = [x.get("select_score") for x in xs if x.get("select_score") is not None]
    print(f"{k:<16} {(sum(sel)/len(sel) if sel else float('nan')):>6.2f} {sum(x['lenient'] for x in xs)/len(xs):>12.2f} "
          f"{sum(x['strict'] for x in xs)/len(xs):>11.2f} {sum(x['fluent'] for x in xs)/len(xs):>11.2f}")
keep = 0; n = 0
for c, v in r.items():
    sw = max(rules, key=lambda k: v[k]["select_score"]); hw = max(rules, key=lambda k: v[k]["lenient"])
    hs = max(v[k]["lenient"] for k in rules)
    if hs > 0:
        n += 1; keep += v[sw]["lenient"] == hs
    print(f"  {c:<24} " + " ".join(f"{k[:2]} {v[k]['select_score']:.2f}->{v[k]['lenient']:.2f}" for k in rules)
          + f" | random {v['random']['lenient']:.2f}")
print(f"\nselection-prompt winner is also (joint) best held-out in {keep}/{n} concepts with any held-out steering")
