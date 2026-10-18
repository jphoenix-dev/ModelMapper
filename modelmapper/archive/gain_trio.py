"""Causal test of the layer-28 "gain trio" (attention study; no API calls).

Prediction (from the feeder analysis): L28H4 and L28H6 are content-blind accelerators and L28H7 a content-blind brake
on concept neurons. So switching off L28H7 should RAISE the model's confidence in its recall answers, switching off
H4+H6 should LOWER it, and a random other layer-28 head should do little. Being content-blind, the effect should show
across concepts and in the sharpness of the whole next-token distribution (entropy), not only on the right answer.

Per concept (all recall prompts, answer position): mean answer log-odds change and mean entropy change (nats) for
  brake_off (L28H7), accel_off (L28H4 + L28H6), random_off (one random layer-28 head not in the trio; fixed per concept)
Usage: RECALL_VARIANT=V3 ANSWER_DISCOVERY=0 CONCEPT_OUT_DIR=<dir> python src/gain_trio.py <tag> "<concept>" [...]
"""
import json
import math
import random
import sys

import torch
import torch.nn.functional as F

from config import DEFAULT_MODEL
import concept_pipeline as cp
from neurons import DEVICE
from pipeline_fast import FastPipeline


def main():
    tag = sys.argv[1]
    pipe = FastPipeline(DEFAULT_MODEL); m = pipe.model
    H = m.config.num_attention_heads
    out_p = cp.OUT / f"gain_trio_{tag}.json"
    out = json.loads(out_p.read_text(encoding="utf-8")) if out_p.exists() else {}
    for concept in [c.strip() for c in sys.argv[2:]]:
        if concept in out:
            continue
        spec, tid, recall, _ = pipe.make_prompts(concept)
        if len(recall) < 5:
            continue
        first = tid[0]
        rng = random.Random(hash(concept) % 1000)
        rh = rng.choice([h for h in range(H) if h not in (4, 6, 7)])
        arms = {"base": [], "brake_off": [(28, 7)], "accel_off": [(28, 4), (28, 6)], "random_off": [(28, rh)]}
        res = {}
        for name, heads in arms.items():
            lo = ent = 0.0
            for q, _ in recall:
                ids = pipe.tok(q, return_tensors="pt").to(DEVICE).input_ids
                with torch.no_grad(), (pipe.ablate_heads(heads) if heads else _null()):
                    logits = m(input_ids=ids).logits[0, -1].float()
                lp = F.log_softmax(logits, -1); p = lp.exp()
                pa = min(max(p[first].item(), 1e-9), 1 - 1e-9)
                lo += math.log(pa / (1 - pa)) / len(recall)
                ent += -(p * lp).sum().item() / len(recall)
            res[name] = {"logodds": lo, "entropy": ent}
        b = res["base"]
        out[concept] = {"random_head": f"L28H{rh}", **{k: {"d_logodds": round(v["logodds"] - b["logodds"], 3),
                                                            "d_entropy": round(v["entropy"] - b["entropy"], 3)} for k, v in res.items() if k != "base"},
                        "base_logodds": round(b["logodds"], 3), "base_entropy": round(b["entropy"], 3)}
        o = out[concept]
        print(f"{concept:<26} brake off: d log-odds {o['brake_off']['d_logodds']:+.2f}, d entropy {o['brake_off']['d_entropy']:+.3f} | "
              f"accelerators off: {o['accel_off']['d_logodds']:+.2f}, {o['accel_off']['d_entropy']:+.3f} | random L28H{rh} off: "
              f"{o['random_off']['d_logodds']:+.2f}, {o['random_off']['d_entropy']:+.3f}", flush=True)
        out_p.write_text(json.dumps(out, indent=1), encoding="utf-8")


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    main()
