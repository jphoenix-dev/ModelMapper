"""Summarise a coverage wave by category (prompts/concepts_wave2.json): recall, k50, steering vs random control.

Usage: python src/wave_summary.py [wave_json=prompts/concepts_wave2.json] [dir=results/concepts_w2]
"""
import json
import sys
from pathlib import Path
from modelmapper.core.settings import REPO_ROOT

ROOT = REPO_ROOT
wave = json.loads((ROOT / (sys.argv[1] if len(sys.argv) > 1 else "prompts/concepts_wave2.json")).read_text(encoding="utf-8"))
d = ROOT / (sys.argv[2] if len(sys.argv) > 2 else "results/concepts_w2")
runs = {}
for f in d.glob("*__clamp_reply.json"):
    x = json.loads(f.read_text(encoding="utf-8"))
    if isinstance(x, dict) and "concept" in x:
        runs[x["concept"].strip()] = x
allr = []
for cat, cs in wave["concepts"].items():
    print(f"\n{cat}")
    for c in cs:
        x = runs.get(c)
        if not x:
            print(f"  {c:<28} (no result)"); continue
        ok = x.get("status") == "ok"
        rnd = (x.get("steering_random_control") or {}).get("score")
        print(f"  {c:<28} {x.get('status'):<22} recall {x.get('recall_kept')}/{x.get('recall_total')} k50 {x.get('k50')} "
              f"steer {x.get('nudgeability')} random {rnd}")
        allr.append((cat, c, ok, x.get("k50"), x.get("nudgeability"), rnd))
ok = [r for r in allr if r[2]]
print(f"\nrecalled {len(ok)}/{len(allr)}")
if ok:
    print(f"k50 <= 5: {sum(r[3] is not None and r[3] <= 5 for r in ok)}/{len(ok)}; steer mean {sum(r[4] or 0 for r in ok)/len(ok):.2f}, "
          f">= 0.5: {sum((r[4] or 0) >= .5 for r in ok)}, >= 0.25: {sum((r[4] or 0) >= .25 for r in ok)}; "
          f"random mean {sum(r[5] or 0 for r in ok)/len(ok):.2f}, max {max(r[5] or 0 for r in ok):.2f}")
    print("recalled, by category:", {cat: sum(1 for r in ok if r[0] == cat) for cat in wave["concepts"]})
    print("steering by category:", {cat: round(sum(r[4] or 0 for r in ok if r[0] == cat) / max(1, sum(1 for r in ok if r[0] == cat)), 2)
                                    for cat in wave["concepts"]})
