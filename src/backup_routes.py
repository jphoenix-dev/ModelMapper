"""Why are near-certain facts hard to flip? Test for backup routes (follow-up to ISSUES #27). No grader calls.

Iterative knockout: round 1 ablates the top-10 attributed neurons; then attribution is re-run WITH those neurons
ablated, so it finds what is carrying the answer now; the next 10 are added; and so on for 5 rounds (50 neurons).
After each round: mean answer log-odds drop, and share of baseline-correct prompts flipped (P < 0.5).
"Backups" predicts that for hard-to-flip facts each new round finds fresh, effective neurons (mostly NOT in the original
top-200) and the drop keeps climbing; for easy facts round 1 already flips. Control: 10 random same-layer neurons per round.

Usage: RECALL_VARIANT=V3 CONCEPT_OUT_DIR=<dir> python src/backup_routes.py <tag> "<concept>" [...]
"""
import json
import math
import random
import sys

from config import DEFAULT_MODEL
import concept_pipeline as cp
from neurons import steer
from pipeline_fast import FastPipeline

ROUNDS, PER = 5, 10


def logit(p):
    # old clamped form: caps at 20.72 (ISSUES #35). Kept only to compare with earlier runs; use pipe.batch_logodds.
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


def p_of(lo):
    """P from exact log-odds (ISSUES #35), overflow-safe."""
    if lo >= 0:
        return 1 / (1 + math.exp(-lo))
    e = math.exp(lo)
    return e / (1 + e)


def main():
    tag = sys.argv[1]
    pipe = FastPipeline(DEFAULT_MODEL)      # ablation only; reply_only is irrelevant to teacher-forced scoring
    out = {}
    for concept in [c.strip() for c in sys.argv[2:]]:
        spec, tid, recall, controls = pipe.make_prompts(concept)
        if len(recall) < 5:
            print(f"{concept}: not recalled", flush=True); continue
        rp = [q for q, _ in recall]
        pairs = [(q, tid) for q in rp]
        base = pipe.batch_logodds(pairs)
        right = [b > 0 for b in base]

        def measure(units):
            los = pipe.batch_logodds(pairs, steer(pipe.model, [(l, n, 0.0) for l, n in units]) if units else None); ps = [p_of(x) for x in los]
            drop = sum(b - x for b, x in zip(base, los)) / len(ps)
            flipped = sum(p < 0.5 for p, r in zip(ps, right) if r) / max(1, sum(right))
            return round(drop, 2), round(flipped, 2)
        first = pipe.attribute(rp, tid)
        orig200 = set(first["top"][:200])
        ablated, rows = [], []
        for r in range(ROUNDS):
            if r == 0:
                A = first
            else:
                with steer(pipe.model, [(l, n, 0.0) for l, n in ablated]):
                    A = pipe.attribute(rp, tid)
            new = [u for u in A["top"] if u not in ablated][:PER]
            ablated += new
            drop, flip = measure(ablated)
            rows.append({"round": r + 1, "k": len(ablated), "new": new, "drop": drop, "flipped": flip,
                         "new_in_orig_top200": sum(u in orig200 for u in new), "layers": sorted(l for l, _ in new)})
        rng = random.Random(3)
        rnd = [(l, rng.randrange(pipe.W)) for l, _ in ablated]
        rand_rows = [measure(rnd[:k]) for k in range(PER, len(rnd) + 1, PER)]
        out[concept] = {"baseline_mean_p": round(sum(1 / (1 + math.exp(-b)) for b in base) / len(base), 3),
                        "rounds": rows, "random": rand_rows}
        print(f"{concept:<22} P {out[concept]['baseline_mean_p']} | " + " | ".join(
            f"k{x['k']}: drop {x['drop']} flip {x['flipped']:.0%} (new in orig200 {x['new_in_orig_top200']}/10)" for x in rows)
              + f" | random k50 {rand_rows[-1]}", flush=True)
        (cp.OUT / f"backup_routes_{tag}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
