"""Do the hard people steer with a different neuron-picking rule? (no grader calls)

For Shakespeare, Mozart, Harry Potter, Einstein and Kahlo: each rule's top-6 neurons from the reply-only
steer_select.json (R1 late / R2 any layer / R3 measured / R4 says the answer) over the usual grid on the 20
person-friendly prompts. On-topic = name word in the reply. Saves every reply for a later fluency check.

STEER_MODE=add nudges instead of clamping (writes people_rules_add.json).

Usage: RECALL_VARIANT=V3 CONCEPT_OUT_DIR=results/concepts_v3 python src/people_rules.py
"""
import json
import os
from pathlib import Path

import torch

from config import DEFAULT_MODEL
import concept_pipeline as cp
from pipeline_fast import FastPipeline

ROOT = Path(__file__).resolve().parents[1]
FRIENDLY = json.loads((ROOT / "prompts/neutral_person_friendly.json").read_text(encoding="utf-8"))
WHO = {"William Shakespeare": "Shakespeare", "Wolfgang Amadeus Mozart": "Mozart", "Harry Potter": "Potter",
       "Albert Einstein": "Einstein", "Frida Kahlo": "Kahlo"}
RULES = ["R1_late_attr", "R2_all_attr", "R3_measured", "R4_says_answer"]


def main():
    os.environ["ANSWER_DISCOVERY"] = "0"
    sel = {k.strip(): v for k, v in json.loads((cp.OUT / "steer_select.json").read_text(encoding="utf-8")).items()}
    mode = os.environ.get("STEER_MODE", "clamp")                   # "add" = nudge on top of natural activity
    pipe = FastPipeline(DEFAULT_MODEL, mode=mode, reply_only=True)
    print(f"steering: {pipe.mode}, reply_only={pipe.reply_only}", flush=True)
    out_name = "people_rules.json" if mode == "clamp" else f"people_rules_{mode}.json"
    out = {}
    for concept, word in WHO.items():
        spec, tid, recall, controls = pipe.make_prompts(concept)
        A = pipe.attribute([q for q, _ in recall], tid)
        res = {}
        for rule in RULES:
            units = [tuple(u) for u in sel[concept][rule]["units"]]
            peaks = torch.tensor([pipe.steer_edits([u], 1.0, A["last_act"])[0][2] for u in units], device="cuda")
            settings = [(n, s) for n in cp.N_STEER for s in cp.STRENGTHS]
            rows, vals, act = [], [], []
            for n, s in settings:
                for p in FRIENDLY:
                    rows.append(p); vals.append(peaks * s); act.append(torch.arange(len(units), device="cuda") < n)
            reps = pipe.gen_rows(rows, units, torch.stack(vals), torch.stack(act))
            k = len(FRIENDLY)
            grid = [{"n": n, "s": s, "hit": sum(word.lower() in r.lower() for r in reps[i*k:(i+1)*k]) / k,
                     "replies": reps[i*k:(i+1)*k]} for i, (n, s) in enumerate(settings)]
            best = max(grid, key=lambda g: g["hit"])
            res[rule] = {"best": best["hit"], "setting": (best["n"], best["s"]), "grid": grid}
        out[concept] = res
        print(f"{concept:<24} " + " | ".join(f"{r[:2]} {res[r]['best']:.2f} {res[r]['setting']}" for r in RULES), flush=True)
        (ROOT / "results" / out_name).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
