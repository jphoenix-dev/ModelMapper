"""Which neurons should we clamp to steer a concept? Compare selection rules (ISSUES #11).

Rules (each picks an ordered list; the steering grid uses its top 2/4/6 at 0.5-2x own peak, reply-only clamp):
  R1_late_attr      top attributed neurons, layers >= L/2 (the frozen CR rule)
  R2_all_attr       top attributed neurons, any layer
  R3_measured       top-200 attributed, re-ranked by measured single-neuron ablation effect on P(answer)
  R4_says_answer    top-200 attributed, re-ranked by direct effect on the answer's first token
                    (output weights -> final norm -> unembedding), i.e. "say <answer>" neurons
Each rule's best joint nudgeability is compared, with a random-neuron control per concept.

Usage: RECALL_VARIANT=V3 CONCEPT_OUT_DIR=results/concepts_v3 python src/steer_select.py "<concept>" [...]
"""
import json
import random
import sys

import torch

import concept_pipeline as cp
from neurons import steer
from pipeline_fast import FastPipeline
from steer_test import PROMPTS as NEUTRAL


def grid(pipe, units, last_act, concept):
    """Same grid as pipeline_fast.steering, but for a given ordered unit list."""
    units = units[:max(cp.N_STEER)]
    peaks = torch.tensor([pipe.steer_edits([u], 1.0, last_act)[0][2] for u in units], device="cuda")
    settings = [(n, s) for n in cp.N_STEER for s in cp.STRENGTHS]
    rows, vals, act = [], [], []
    for n, s in settings:
        for p in NEUTRAL:
            rows.append(p); vals.append(peaks * s); act.append(torch.arange(len(units), device="cuda") < n)
    replies = pipe.gen_rows(rows, units, torch.stack(vals), torch.stack(act))
    on, fl = pipe.judge_detail(replies, concept)
    k = len(NEUTRAL); best = (0.0, None)
    for i, (n, s) in enumerate(settings):
        j = sum(a >= 0.5 and b >= 0.5 for a, b in zip(on[i*k:(i+1)*k], fl[i*k:(i+1)*k])) / k
        if j > best[0] or best[1] is None:
            best = (j, (n, s), replies[i*k:(i+1)*k][:3])
    return best


def main():
    pipe = FastPipeline("Qwen/Qwen3-8B")
    W_U = pipe.model.lm_head.weight.float(); g = pipe.model.model.norm.weight.float()
    out = {}
    for concept in sys.argv[1:]:
        spec, tid, recall, controls = pipe.make_prompts(concept)
        if len(recall) < 5:
            print(f"{concept}: not recalled, skipped", flush=True); continue
        rp = [q for q, _ in recall]
        A = pipe.attribute(rp, tid)
        top = A["top"][:200]; last_act = A["last_act"]
        base = sum(pipe.probs(rp, tid)) / len(rp)
        meas = sorted(top, key=lambda u: -(base - sum(pipe.batch_probs([(q, tid) for q in rp], steer(pipe.model, [(u[0], u[1], 0.0)]))) / len(rp)))
        first = tid[0]

        def says(u):
            l, n = u
            col = pipe.model.model.layers[l].mlp.down_proj.weight[:, n].float()
            sign = -1.0 if last_act[l, n] < 0 else 1.0            # the sign it takes on recall
            return sign * (W_U[first] @ (col * g)).item()
        rules = {"R1_late_attr": [u for u in top if u[0] >= pipe.L // 2],
                 "R2_all_attr": top,
                 "R3_measured": meas,
                 "R4_says_answer": sorted(top, key=lambda u: -says(u))}
        res = {}
        for name, order in rules.items():
            j, setting, sample = grid(pipe, order, last_act, concept)
            res[name] = {"joint": round(j, 3), "setting": setting, "units": order[:6], "sample": sample}
        rng = random.Random(0)
        rj, _, _ = grid(pipe, [(l, rng.randrange(pipe.W)) for l, _ in top[:6]], last_act, concept)
        res["random"] = {"joint": round(rj, 3)}
        out[concept] = res
        print(f"{concept:<24} " + " | ".join(f"{k} {v['joint']:.2f}" for k, v in res.items()), flush=True)
        (cp.OUT / "steer_select.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
