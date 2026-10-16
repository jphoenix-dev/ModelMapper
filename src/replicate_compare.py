"""Prompt-set noise: CR-2 vs CR-2b (same format, same pipeline, independently written prompts).

Per concept: recalled?, k50, nudgeability (joint, selection prompts), top-10 overlap (exact neurons), and overlap within
the top-50 (does each run's top 10 appear anywhere in the other's top 50?). Population numbers with bootstrap CIs over
concepts. Note: CR-2b's steering uses its own neurons and the same 20 neutral prompts.

Usage: python src/replicate_compare.py [dirA=results/concepts_v3] [dirB=results/concepts_v3b]
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "results/concepts_v3")
B = ROOT / (sys.argv[2] if len(sys.argv) > 2 else "results/concepts_v3b")


def load(d):
    out = {}
    for f in d.glob("*__clamp_reply.json"):
        x = json.loads(f.read_text(encoding="utf-8"))
        if isinstance(x, dict) and "concept" in x:
            out[x["concept"].strip()] = x
    return out


def top(x, k):
    return [(t["layer"], t["neuron"]) for t in (x.get("top_neurons") or [])[:k]]


def boot(v, n=2000):
    rng = random.Random(0)
    s = sorted(sum(rng.choice(v) for _ in v) / len(v) for _ in range(n))
    return s[int(.025 * n)], s[int(.975 * n)]


a, b = load(A), load(B)
both = sorted(set(a) & set(b))
print(f"{len(both)} concepts in both\n{'concept':<26} {'recall A/B':>11} {'k50 A/B':>9} {'nudge A/B':>11} {'top10 overlap':>13} {'top10 in other top50':>20}")
rows = []
for c in both:
    x, y = a[c], b[c]
    ra, rb = x.get("status") == "ok", y.get("status") == "ok"
    ov = len(set(top(x, 10)) & set(top(y, 10))) if ra and rb else None
    ov50 = (len(set(top(x, 10)) & set(top(y, 50))) + len(set(top(y, 10)) & set(top(x, 50)))) / 2 if ra and rb else None
    rows.append((c, ra, rb, x.get("k50"), y.get("k50"), x.get("nudgeability"), y.get("nudgeability"), ov, ov50))
    print(f"{c:<26} {str(ra)[0]}/{str(rb)[0]:<9} {str(x.get('k50')):>4}/{str(y.get('k50')):<4} "
          f"{str(x.get('nudgeability')):>5}/{str(y.get('nudgeability')):<5} {str(ov):>13} {str(ov50):>20}")
ok = [r for r in rows if r[1] and r[2]]
print(f"\nrecalled in both: {len(ok)}; recalled in A only {sum(r[1] and not r[2] for r in rows)}, B only {sum(r[2] and not r[1] for r in rows)}")
d = [abs(r[5] - r[6]) for r in ok if r[5] is not None and r[6] is not None]
print(f"nudgeability |A-B|: mean {sum(d)/len(d):.2f}, max {max(d):.2f}; means A {sum(r[5] for r in ok)/len(ok):.2f} vs B {sum(r[6] for r in ok)/len(ok):.2f}")
for name, i in (("k50 <= 5", 3), ):
    fa = [1 if r[3] is not None and r[3] <= 5 else 0 for r in ok]; fb = [1 if r[4] is not None and r[4] <= 5 else 0 for r in ok]
    print(f"{name}: A {sum(fa)}/{len(ok)} {boot(fa)} | B {sum(fb)}/{len(ok)} {boot(fb)}")
o = [r[7] for r in ok]; o50 = [r[8] for r in ok]
print(f"top-10 exact overlap: mean {sum(o)/len(o):.1f}/10 (range {min(o)}-{max(o)}); top-10 found in other's top-50: mean {sum(o50)/len(o50):.1f}/10")
