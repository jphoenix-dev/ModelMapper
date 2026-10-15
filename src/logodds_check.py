"""Re-measure "how many neurons carry recall" on a non-saturating scale (ISSUES #27).

"Halve P(answer)" is much harder when the model is near-certain (P ~ 1) than at P ~ 0.5, which biases k50 by baseline
confidence. Here the effect of ablating the top-k is the mean drop in the answer's log-odds, logit(p) = log(p/(1-p)),
computed per prompt then averaged. k_LO3 = smallest k whose ablation lowers mean log-odds by >= 3 (odds ~20x lower),
the same-sized effect at any starting confidence. Neurons are taken in measured single-ablation order (top-200 attributed,
re-ranked) and in attribution order; random neurons from the same layers as a baseline.

Usage: RECALL_VARIANT=V0|V3 CONCEPT_OUT_DIR=... python src/logodds_check.py <out_tag> "<concept>" [...]
"""
import json
import math
import random
import sys

import concept_pipeline as cp
from neurons import steer
from pipeline_fast import FastPipeline

KS = [1, 2, 3, 5, 10, 20, 50]
DELTA = 3.0


def logit(p):
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


def main():
    tag = sys.argv[1]
    pipe = FastPipeline("Qwen/Qwen3-8B")
    out = {}
    for concept in sys.argv[2:]:
        spec, tid, recall, controls = pipe.make_prompts(concept)
        if len(recall) < 5:
            continue
        rp = [q for q, _ in recall]
        pairs = [(q, tid) for q in rp]
        base = [logit(p) for p in pipe.batch_probs(pairs)]
        A = pipe.attribute(rp, tid)
        top = A["top"][:200]

        def mlo(edits):
            ps = pipe.batch_probs(pairs, steer(pipe.model, edits) if edits else None)
            return sum(b - logit(p) for b, p in zip(base, ps)) / len(ps)      # mean log-odds drop
        single = sorted(top, key=lambda u: -mlo([(u[0], u[1], 0.0)]))

        def curve(order):
            drops, k3 = [], None
            for k in KS:
                d = mlo([(l, n, 0.0) for l, n in order[:k]])
                drops.append(round(d, 2))
                if k3 is None and d >= DELTA:
                    k3 = k
            return drops, k3
        a, ka = curve(top)
        m, km = curve(single)
        rng = random.Random(2)
        r, _ = curve([(l, rng.randrange(pipe.W)) for l, _ in top])
        out[concept] = {"baseline_mean_logodds": round(sum(base) / len(base), 2),
                        "baseline_mean_p": round(sum(1 / (1 + math.exp(-b)) for b in base) / len(base), 3),
                        "drop_attr": a, "k_LO3_attr": ka, "drop_measured": m, "k_LO3_measured": km, "drop_random": r}
        print(f"{concept:<24} base log-odds {out[concept]['baseline_mean_logodds']:>6} (P {out[concept]['baseline_mean_p']}) | "
              f"k(-3 log-odds) attr {ka} / measured {km} | measured drops {m} | random {r}", flush=True)
    (cp.OUT / f"logodds_{tag}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
