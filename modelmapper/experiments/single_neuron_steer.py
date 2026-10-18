"""Steer with ONE named neuron (e.g. the only Shakespeare-labelled neuron in Shakespeare's top 10). No grader here;
run people_fluency-style judging after. Grid: clamp and add, strength 0.5-8x its own peak, person-friendly prompts.

Usage: RECALL_VARIANT=V3 CONCEPT_OUT_DIR=results/concepts_v3 python src/single_neuron_steer.py "<concept>" <word> <layer> <neuron> <tag>
"""
import json
import os
import sys
from pathlib import Path

import torch

from modelmapper.core.neurons import DEVICE
from modelmapper.core.config import DEFAULT_MODEL
from modelmapper.core import concept_pipeline as cp
from modelmapper.core.pipeline_fast import FastPipeline
from modelmapper.core.settings import REPO_ROOT

ROOT = REPO_ROOT
FRIENDLY = json.loads((ROOT / "prompts/neutral_person_friendly.json").read_text(encoding="utf-8"))
STRENGTHS = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]


def main():
    concept, word, layer, neuron, tag = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    os.environ["ANSWER_DISCOVERY"] = "0"
    out = {}
    pipe = None
    for mode in ("clamp", "add"):
        if pipe is None:
            pipe = FastPipeline(DEFAULT_MODEL, mode=mode, reply_only=True)
        pipe.mode = mode
        spec, tid, recall, controls = pipe.make_prompts(concept)
        A = pipe.attribute([q for q, _ in recall], tid)
        u = (layer, neuron)
        peak = pipe.steer_edits([u], 1.0, A["last_act"])[0][2]
        k = len(FRIENDLY)
        rows = [p for _ in STRENGTHS for p in FRIENDLY]
        vals = torch.tensor([[peak * s] for s in STRENGTHS for _ in FRIENDLY], device=DEVICE)
        act = torch.ones(len(rows), 1, dtype=torch.bool, device=DEVICE)
        reps = pipe.gen_rows(rows, [u], vals, act)
        grid = [{"n": 1, "s": s, "hit": sum(word.lower() in r.lower() for r in reps[i*k:(i+1)*k]) / k,
                 "replies": reps[i*k:(i+1)*k]} for i, s in enumerate(STRENGTHS)]
        best = max(grid, key=lambda g: g["hit"])
        out[mode] = {"best": best["hit"], "setting": (1, best["s"]), "grid": grid}
        print(f"{concept} L{layer}:{neuron} {mode}: name-hit by strength " +
              " ".join(f"{g['s']}x {g['hit']:.2f}" for g in grid), flush=True)
    (ROOT / f"results/single_{tag}.json").write_text(json.dumps({concept: out}, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
