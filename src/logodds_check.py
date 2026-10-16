"""Re-measure "how many neurons carry recall" on a non-saturating scale (ISSUES #27).

"Halve P(answer)" is much harder when the model is near-certain (P ~ 1) than at P ~ 0.5, which biases k50 by baseline
confidence. Here the effect of ablating the top-k is the mean drop in the answer's log-odds, logit(p) = log(p/(1-p)),
computed per prompt then averaged. k_LO3 = smallest k whose ablation lowers mean log-odds by >= 3 (odds ~20x lower),
the same-sized effect at any starting confidence. Neurons are taken in measured single-ablation order (top-200 attributed,
re-ranked) and in attribution order; random neurons from the same layers as a baseline.

Also per prompt (v2): the median drop (k_LO3_median: robust to a few prompts collapsing) and k_flip = fewest neurons
that push P(answer) below 0.5 on at least half the prompts (the "answer actually changes" view).

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

        def per_prompt(edits):
            ps = pipe.batch_probs(pairs, steer(pipe.model, edits) if edits else None)
            return [b - logit(p) for b, p in zip(base, ps)], ps

        def mlo(edits):
            d, _ = per_prompt(edits)
            return sum(d) / len(d)                                            # mean log-odds drop
        single = sorted(top, key=lambda u: -mlo([(u[0], u[1], 0.0)]))
        detail = {}

        def curve(order, tag=None):
            drops, k3 = [], None
            for k in KS:
                d, ps = per_prompt([(l, n, 0.0) for l, n in order[:k]])
                mean = sum(d) / len(d)
                drops.append(round(mean, 2))
                if tag:                     # per-prompt detail: is the mean carried by a few collapsing prompts?
                    detail.setdefault(tag, []).append({"k": k, "median_drop": round(sorted(d)[len(d) // 2], 2),
                                                       "share_drop_ge3": round(sum(x >= DELTA for x in d) / len(d), 2),
                                                       "share_p_below_half": round(sum(p < 0.5 for p in ps) / len(ps), 2),
                                                       "per_prompt": [round(x, 2) for x in d]})
                if k3 is None and mean >= DELTA:
                    k3 = k
            return drops, k3
        a, ka = curve(top)
        m, km = curve(single, "measured")
        rng = random.Random(2)
        r, _ = curve([(l, rng.randrange(pipe.W)) for l, _ in top], "random")
        med = [x["median_drop"] for x in detail["measured"]]
        k_med = next((k for k, x in zip(KS, med) if x >= DELTA), None)
        k_flip = next((x["k"] for x in detail["measured"] if x["share_p_below_half"] >= 0.5), None)
        out[concept] = {"baseline_mean_logodds": round(sum(base) / len(base), 2),
                        "baseline_mean_p": round(sum(1 / (1 + math.exp(-b)) for b in base) / len(base), 3),
                        "drop_attr": a, "k_LO3_attr": ka, "drop_measured": m, "k_LO3_measured": km, "drop_random": r,
                        "k_LO3_median_measured": k_med, "k_flip_half_prompts_measured": k_flip,
                        "baseline_share_p_above_half": round(sum(b > 0 for b in base) / len(base), 2), "detail": detail}
        print(f"{concept:<24} base log-odds {out[concept]['baseline_mean_logodds']:>6} (P {out[concept]['baseline_mean_p']}) | "
              f"k(-3 log-odds) attr {ka} / measured {km} / median {k_med} | k(flip half the prompts) {k_flip} | measured drops {m} | random {r}", flush=True)
    (cp.OUT / f"logodds_{tag}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
