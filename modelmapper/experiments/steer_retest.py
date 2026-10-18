"""Held-out re-test of the steering-selection winners (ISSUES #11, guards against winner's curse).

steer_select.py reports each rule's BEST of 15 settings on the same 20 neutral prompts it chose the setting on, which
flatters every rule. Here each rule's chosen (units, n, strength) is re-run on 20 fresh neutral prompts
(prompts/neutral_heldout.json, no concept words) and judged with both wordings (lenient "mention/preoccupied" and strict
"central theme"), plus a random-neuron control at the strongest setting (6 neurons, 2x peak). Each rule is generated
separately: the per-row hook writes by index, so one neuron shared between two rules in one batch could be overwritten.

Usage: RECALL_VARIANT=V3 CONCEPT_OUT_DIR=results/concepts_v3 python src/steer_retest.py
"""
import json
import random
from pathlib import Path

import torch

from modelmapper.core.neurons import DEVICE
from modelmapper.core.config import DEFAULT_MODEL
from modelmapper.core import concept_pipeline as cp
from modelmapper.core.pipeline_fast import FastPipeline
from modelmapper.core.strict_judge import _safe, strict
from modelmapper.core.settings import REPO_ROOT

ROOT = REPO_ROOT
HELD = json.loads((ROOT / "prompts/neutral_heldout.json").read_text(encoding="utf-8"))


def run(pipe, units, n, s, last_act, concept):
    peaks = torch.tensor([pipe.steer_edits([u], 1.0, last_act)[0][2] for u in units], device=DEVICE)
    vals = torch.stack([peaks * s] * len(HELD))
    act = torch.stack([torch.arange(len(units), device=DEVICE) < n] * len(HELD))
    replies = pipe.gen_rows(HELD, units, vals, act)
    on, fl = pipe.judge_detail(replies, concept)
    st = [_safe(strict, r, concept) for r in replies]
    k = len(HELD)
    return {"lenient": round(sum(a >= 0.5 and b >= 0.5 for a, b in zip(on, fl)) / k, 3),
            "strict": round(sum((c or 0) >= 0.5 and b >= 0.5 for c, b in zip(st, fl)) / k, 3),
            "fluent": round(sum(b >= 0.5 for b in fl) / k, 3), "sample": replies[:2]}


def main():
    sel = json.loads((cp.OUT / "steer_select.json").read_text(encoding="utf-8"))
    out_p = cp.OUT / "steer_retest.json"
    out = json.loads(out_p.read_text(encoding="utf-8")) if out_p.exists() else {}
    pipe = FastPipeline(DEFAULT_MODEL, mode="clamp", reply_only=True)      # as in CR/CR-2 (ISSUES #28)
    print(f"steering: clamp, reply_only={pipe.reply_only}", flush=True)
    for concept, rules in sel.items():
        if concept in out:
            continue
        spec, tid, recall, controls = pipe.make_prompts(concept)
        rp = [q for q, _ in recall]
        A = pipe.attribute(rp, tid)
        last_act = A["last_act"]
        res = {}
        for name, r in rules.items():
            if name == "random" or not r.get("setting"):
                continue
            n, s = r["setting"]
            units = [tuple(u) for u in r["units"]]
            res[name] = dict(run(pipe, units, n, s, last_act, concept), setting=(n, s), select_score=r["joint"])
        rng = random.Random(0)
        rand = [(l, rng.randrange(pipe.W)) for l, _ in A["top"][:6]]
        res["random"] = run(pipe, rand, 6, 2.0, last_act, concept)
        out[concept] = res
        print(f"{concept:<26} " + " | ".join(f"{k} sel {v.get('select_score', '-')} -> held-out {v['lenient']:.2f}/{v['strict']:.2f}"
                                               for k, v in res.items()), flush=True)
        out_p.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
