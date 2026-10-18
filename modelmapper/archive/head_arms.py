"""Switch off named sets of heads and measure recall + sharpness, across concepts (attention study; no API).

Generalises gain_trio.py. Arms come from the HEAD_ARMS env, e.g.
  HEAD_ARMS="writers=L35H25,L34H28,L35H22;anti=L35H27,L34H29,L35H24"
plus an automatic control arm per concept: the same number of random heads from the same layers as the FIRST arm,
excluding every named head (seeded by the concept). Per concept (all recall prompts, answer position): mean change in
the answer's first-token log-odds and in the next-token entropy (nats).
Usage: RECALL_VARIANT=V3 ANSWER_DISCOVERY=0 CONCEPT_OUT_DIR=<dir> HEAD_ARMS=... python src/head_arms.py <tag> "<concept>" [...]
"""
import json
import math
import os
import random
import sys

import torch
import torch.nn.functional as F

from config import DEFAULT_MODEL
import concept_pipeline as cp
from neurons import DEVICE
from pipeline_fast import FastPipeline


def parse_arms(s):
    arms = {}
    for part in s.split(";"):
        name, hs = part.split("=")
        arms[name.strip()] = [tuple(int(v) for v in h.strip()[1:].split("H")) for h in hs.split(",")]
    return arms


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def main():
    tag = sys.argv[1]; arms = parse_arms(os.environ["HEAD_ARMS"])
    pipe = FastPipeline(DEFAULT_MODEL); m = pipe.model; H = m.config.num_attention_heads
    named = {h for hs in arms.values() for h in hs}; first = next(iter(arms.values()))
    out_p = cp.OUT / f"head_arms_{tag}.json"
    out = json.loads(out_p.read_text(encoding="utf-8")) if out_p.exists() else {}
    for concept in [c.strip() for c in sys.argv[2:]]:
        if concept in out:
            continue
        spec, tid, recall, _ = pipe.make_prompts(concept)
        a = tid[0]
        rng = random.Random(sum(map(ord, concept)))
        ctrl = []
        for l, _ in first:
            ctrl.append((l, rng.choice([h for h in range(H) if (l, h) not in named and (l, h) not in ctrl])))
        all_arms = {"base": [], **arms, "random": ctrl}
        res = {}
        for name, heads in all_arms.items():
            lo = ent = 0.0
            for q, _ in recall:
                ids = pipe.tok(q, return_tensors="pt").to(DEVICE).input_ids
                with torch.no_grad(), (pipe.ablate_heads(heads) if heads else _null()):
                    logits = m(input_ids=ids).logits[0, -1].float()
                lp = F.log_softmax(logits, -1); p = lp.exp()
                pa = min(max(p[a].item(), 1e-9), 1 - 1e-9)
                lo += math.log(pa / (1 - pa)) / len(recall); ent += -(p * lp).sum().item() / len(recall)
            res[name] = (lo, ent)
        b = res["base"]
        out[concept] = {"random_heads": [f"L{l}H{h}" for l, h in ctrl], "base_logodds": round(b[0], 3), "base_entropy": round(b[1], 3),
                        **{k: {"d_logodds": round(v[0] - b[0], 3), "d_entropy": round(v[1] - b[1], 3)} for k, v in res.items() if k != "base"}}
        o = out[concept]
        print(f"{concept:<28} " + " | ".join(f"{k} off: {o[k]['d_logodds']:+.2f} ({o[k]['d_entropy']:+.3f})" for k in all_arms if k != "base"), flush=True)
        out_p.write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
