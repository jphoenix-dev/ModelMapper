"""Is recall really spread out, or is attribution just ranking neurons badly? (CR-2 caveat 1)

For each concept: take the top-200 attributed neurons, ablate each ALONE and measure the real drop in P(answer),
re-rank by that measured effect, and recompute the ablation curve (k = 1..50) in the new order, next to the
attribution-order curve and a random-neuron baseline. If the measured order halves recall with far fewer neurons,
large k50 values were a ranking artefact.

Usage: RECALL_VARIANT=V3 CONCEPT_OUT_DIR=results/concepts_v3 python src/k50_check.py "<concept>" [...]
"""
import json
import random
import sys
from pathlib import Path

import torch

from modelmapper.core.config import DEFAULT_MODEL
from modelmapper.core import concept_pipeline as cp
from modelmapper.core.neurons import steer
from modelmapper.core.pipeline_fast import FastPipeline

KS = [1, 2, 3, 5, 10, 20, 50]


def main():
    pipe = FastPipeline(DEFAULT_MODEL)
    out = {}
    for concept in sys.argv[1:]:
        spec, tid, recall, controls = pipe.make_prompts(concept)
        rp = [q for q, _ in recall]
        A = pipe.attribute(rp, tid)
        top = A["top"][:200]
        base = sum(pipe.probs(rp, tid)) / len(rp)
        single = []
        for l, n in top:                                   # measured effect of each neuron alone
            p = sum(pipe.batch_probs([(q, tid) for q in rp], steer(pipe.model, [(l, n, 0.0)]))) / len(rp)
            single.append((base - p, (l, n)))
        measured = [u for _, u in sorted(single, reverse=True)]

        def curve(order):
            res, k50 = [], None
            for k in KS:
                p = sum(pipe.batch_probs([(q, tid) for q in rp], steer(pipe.model, [(l, n, 0.0) for l, n in order[:k]]))) / len(rp)
                res.append(round(p, 3))
                if k50 is None and p <= base / 2:
                    k50 = k
            return res, k50
        a_curve, a_k50 = curve(top)
        m_curve, m_k50 = curve(measured)
        rng = random.Random(1)
        r_curve, _ = curve([(l, rng.randrange(pipe.W)) for l, _ in top])
        out[concept] = {"baseline": round(base, 3), "attribution_order": a_curve, "k50_attribution": a_k50,
                        "measured_order": m_curve, "k50_measured": m_k50, "random": r_curve,
                        "top_single_effects": [(round(e, 4), u) for e, u in sorted(single, reverse=True)[:10]]}
        print(f"{concept:<24} base {base:.2f} | k50 attribution-order {a_k50} -> measured-order {m_k50} | "
              f"curves attr {a_curve} | meas {m_curve} | random {r_curve}", flush=True)
    (cp.OUT / "k50_check.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
