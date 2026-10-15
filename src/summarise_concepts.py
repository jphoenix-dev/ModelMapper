"""One table from all concept-pipeline results: recall, handful (k50), specificity, legibility, nudgeability.

Usage: python src/summarise_concepts.py [tag=clamp]   (tag = steering variant, e.g. clamp, clamp_reply, add, add_reply)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
tag = sys.argv[1] if len(sys.argv) > 1 else "clamp"
rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((ROOT / "results/concepts").glob(f"*__{tag}.json"))]

print(f"{'concept':<28} {'recall':>7} {'base P':>6} {'k50':>4} {'rand@50':>7} {'unrel. drop':>11} "
      f"{'clean (either pole)':>19} {'nudge':>6} {'rand':>5}  best steer")
for r in rows:
    if r["status"] != "ok":
        print(f"{r['concept']:<28} {r['recall_kept']:>3}/{r['recall_total']:<3} {'-':>6}  -> {r['status']}")
        continue
    k50 = r["k50"] if r["k50"] is not None else ">50"
    a50 = next(x for x in r["ablation"] if x["k"] == 50)
    # unrelated controls are the last 4 Luna wrote; compare their P at k50 (or k=10) to unablated
    base_ctrl = {c["answer"]: c["p"] for c in r["controls"]}
    kk = r["k50"] or 10
    at = next(x for x in r["ablation"] if x["k"] == kk)["controls"]
    unrelated = [c["answer"] for c in r["controls"]][-4:]
    drops = [base_ctrl[a] - at[a] for a in unrelated if a in at]
    drop = f"{100*sum(drops)/len(drops):.0f}pp" if drops else "-"
    labs = r.get("labels", {})
    clean = sum(bool(v["clean_pos"] or v["clean_neg"]) for v in labs.values())
    b = r["steering_best"]
    print(f"{r['concept']:<28} {r['recall_kept']:>3}/{r['recall_total']:<3} {r['baseline_p']:>6.2f} {k50!s:>4} "
          f"{a50['p_answer_random_ablation']/max(r['baseline_p'],1e-9):>6.0%} {drop:>11} "
          f"{(f'{clean}/{len(labs)}' if labs else '-'):>19} {r['nudgeability']:>6.2f} "
          f"{r['steering_random_control']['score']:>5.2f}  n={b['n']} @{b['strength']}x "
          f"({b['on_topic']:.0%} on, {b['fluent']:.0%} fluent)")
