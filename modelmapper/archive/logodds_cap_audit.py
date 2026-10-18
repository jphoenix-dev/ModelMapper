"""Audit: did the float32 softmax + clamp (p in [1e-9, 1-1e-9]) silently cap log-odds? (no API)

Many scripts compute log-odds as log(p / (1 - p)) from a float32 softmax, clamping p to 1 - 1e-9. In float32, p
rounds to exactly 1.0 once the true log-odds exceed ~16.6, and the clamp then reports 20.72 whatever the truth.
Here: the EXACT log-odds (logit_a - logsumexp(other logits), float64) of every recall prompt's answer first token,
next to the clamped value, for every concept given. Counts prompts above 16 (at risk) and where they differ by > 0.01.
Usage: RECALL_VARIANT=V3 ANSWER_DISCOVERY=0 CONCEPT_OUT_DIR=<dir> python src/logodds_cap_audit.py <tag> "<concept>" [...]
"""
import json
import math
import sys

import torch
import torch.nn.functional as F

from config import DEFAULT_MODEL
import concept_pipeline as cp
from neurons import DEVICE
from pipeline_fast import FastPipeline


def exact_logodds(logits, a):
    x = logits.double(); others = torch.cat([x[:a], x[a + 1:]])
    return (x[a] - torch.logsumexp(others, 0)).item()


def clamped_logodds(logits, a):
    p = F.softmax(logits.float(), -1)[a].item(); p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


def main():
    tag = sys.argv[1]
    pipe = FastPipeline(DEFAULT_MODEL); m = pipe.model
    out = {}
    for concept in [c.strip() for c in sys.argv[2:]]:
        spec, tid, recall, _ = pipe.make_prompts(concept); a = tid[0]
        rows = []
        for q, _ in recall:
            ids = pipe.tok(q, return_tensors="pt").to(DEVICE).input_ids
            with torch.no_grad():
                lg = m(input_ids=ids).logits[0, -1]
            rows.append((round(exact_logodds(lg, a), 3), round(clamped_logodds(lg, a), 3)))
        out[concept] = rows
        hi = sum(e > 16 for e, _ in rows); diff = sum(abs(e - c) > 0.01 for e, c in rows)
        print(f"{concept:<28} prompts {len(rows)} | exact max {max(e for e, _ in rows):.2f} | >16: {hi} | clamped differs: {diff}", flush=True)
    (cp.OUT / f"logodds_cap_audit_{tag}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
