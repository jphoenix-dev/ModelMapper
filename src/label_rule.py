"""R5 "label-matched" steering rule vs the frozen R1 rule, same metric (follow-up to Shakespeare's single neuron).

R5 = the concept's top-10 attributed neurons whose recall-pole label names the concept (results/label_matched_v3.json),
all of them together (1-5 neurons). R1 = top attributed neurons in layers >= L/2, first 2/4/6. Both in add mode
(nudge on natural activity), reply-only, strengths 0.5-4x peak. Prompts: original neutral prompts, or the
person-friendly set for people. On-topic = a distinctive concept word in the reply (KEY); fluency judged afterwards
by people_fluency.py-style Jev pass (label_rule_fluency.py). Every reply is saved.

Usage: RECALL_VARIANT=V3 CONCEPT_OUT_DIR=results/concepts_v3 python src/label_rule.py
"""
import json
import os
from pathlib import Path

import torch

import concept_pipeline as cp
from pipeline_fast import FastPipeline
from steer_test import PROMPTS as NEUTRAL

ROOT = Path(__file__).resolve().parents[1]
FRIENDLY = json.loads((ROOT / "prompts/neutral_person_friendly.json").read_text(encoding="utf-8"))
KEY = {"chess": "chess", "coffee": "coffee", "democracy": "democra", "Eiffel Tower": "eiffel", "Golden Gate Bridge": "golden gate",
       "Harry Potter": "potter", "Mount Everest": "everest", "Napoleon Bonaparte": "napoleon", "photosynthesis": "photosynth",
       "the French Revolution": "french revolution", "the Moon": "moon", "the Titanic": "titanic", "the violin": "violin",
       "Venice": "venice", "William Shakespeare": "shakespeare"}
PEOPLE = {"Harry Potter", "Napoleon Bonaparte", "William Shakespeare"}
STRENGTHS = [0.5, 1.0, 2.0, 3.0, 4.0]


def main():
    os.environ["ANSWER_DISCOVERY"] = "0"
    matched = json.loads((ROOT / "results/label_matched_v3.json").read_text(encoding="utf-8"))
    pipe = FastPipeline("Qwen/Qwen3-8B", mode="add", reply_only=True)
    print(f"steering: {pipe.mode}, reply_only={pipe.reply_only}", flush=True)
    out = {}
    for concept, key in KEY.items():
        r5 = [(l, n) for l, n, _ in matched.get(concept, [])]
        if not r5:
            continue
        prompts = FRIENDLY if concept in PEOPLE else NEUTRAL
        spec, tid, recall, controls = pipe.make_prompts(concept)
        A = pipe.attribute([q for q, _ in recall], tid)
        r1 = [(l, n) for l, n in A["top"] if l >= pipe.L // 2][:6]
        res = {}
        for rule, units, ns in (("R1_late_attr", r1, [2, 4, 6]), ("R5_label_matched", r5, [len(r5)])):
            peaks = torch.tensor([pipe.steer_edits([u], 1.0, A["last_act"])[0][2] for u in units], device="cuda")
            settings = [(n, s) for n in ns for s in STRENGTHS]
            rows, vals, act = [], [], []
            for n, s in settings:
                for p in prompts:
                    rows.append(p); vals.append(peaks * s); act.append(torch.arange(len(units), device="cuda") < n)
            reps = pipe.gen_rows(rows, units, torch.stack(vals), torch.stack(act))
            k = len(prompts)
            grid = [{"n": n, "s": s, "hit": sum(key in r.lower() for r in reps[i*k:(i+1)*k]) / k,
                     "replies": reps[i*k:(i+1)*k]} for i, (n, s) in enumerate(settings)]
            best = max(grid, key=lambda g: g["hit"])
            res[rule] = {"units": units, "best": best["hit"], "setting": (best["n"], best["s"]), "grid": grid}
        out[concept] = res
        print(f"{concept:<24} R1 name {res['R1_late_attr']['best']:.2f} | R5 ({len(r5)} neurons) name {res['R5_label_matched']['best']:.2f}", flush=True)
        (ROOT / "results/label_rule.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
