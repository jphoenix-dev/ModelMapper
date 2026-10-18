"""Are people unsteerable only because the neutral prompts give them no way in? (no grader calls)

Steer each recalled person with the frozen rule's neurons (top attributed, layers >= L/2) over the usual grid
(2/4/6 neurons x 0.5-2x peak, reply-only clamp) on 20 person-friendly neutral prompts
(prompts/neutral_person_friendly.json), and also on the original neutral prompts. On-topic = the person's distinctive
name word (surname, or "Potter") appears in the reply, a reliable check for names that needs no judge. Baselines:
unsteered replies to the same prompts, and 6 random same-layer neurons at 2x.

Usage: RECALL_VARIANT=V3 python src/people_prompts.py
"""
import json
import os
import random
from pathlib import Path

import torch

from modelmapper.core.neurons import DEVICE
from modelmapper.core.config import DEFAULT_MODEL
from modelmapper.core import concept_pipeline as cp
from modelmapper.core.pipeline_fast import FastPipeline
from modelmapper.core.steer_test import PROMPTS as NEUTRAL
from modelmapper.core.settings import REPO_ROOT

ROOT = REPO_ROOT
FRIENDLY = json.loads((ROOT / "prompts/neutral_person_friendly.json").read_text(encoding="utf-8"))
PEOPLE = [("results/concepts_v3", "Albert Einstein", "Einstein"), ("results/concepts_v3", "William Shakespeare", "Shakespeare"),
          ("results/concepts_v3", "Napoleon Bonaparte", "Napoleon"), ("results/concepts_v3", "Wolfgang Amadeus Mozart", "Mozart"),
          ("results/concepts_v3", "Frida Kahlo", "Kahlo"), ("results/concepts_v3", "Harry Potter", "Potter"),
          ("results/concepts_w2", "Marie Curie", "Curie"), ("results/concepts_w2", "Charles Darwin", "Darwin")]


def hit(reply, word):
    return word.lower() in reply.lower()


def main():
    os.environ["ANSWER_DISCOVERY"] = "0"
    pipe = FastPipeline(DEFAULT_MODEL, mode="clamp", reply_only=True)
    print(f"steering: clamp, reply_only={pipe.reply_only}", flush=True)
    out = {}
    for d, concept, word in PEOPLE:
        cp.OUT = ROOT / d
        from modelmapper.core import pipeline_fast
        pipeline_fast.OUT = cp.OUT
        spec, tid, recall, controls = pipe.make_prompts(concept)
        if len(recall) < 5:
            print(f"{concept}: not recalled", flush=True); continue
        A = pipe.attribute([q for q, _ in recall], tid)
        units = [(l, n) for l, n in A["top"] if l >= pipe.L // 2][:6]
        peaks = torch.tensor([pipe.steer_edits([u], 1.0, A["last_act"])[0][2] for u in units], device=DEVICE)
        res = {}
        for pname, prompts in (("friendly", FRIENDLY), ("original", NEUTRAL)):
            base = pipe.gen_rows(prompts, units, torch.zeros(len(prompts), len(units), device=DEVICE),
                                 torch.zeros(len(prompts), len(units), dtype=torch.bool, device=DEVICE))
            rows, vals, act, settings = [], [], [], [(n, s) for n in cp.N_STEER for s in cp.STRENGTHS]
            for n, s in settings:
                for p in prompts:
                    rows.append(p); vals.append(peaks * s); act.append(torch.arange(len(units), device=DEVICE) < n)
            reps = pipe.gen_rows(rows, units, torch.stack(vals), torch.stack(act))
            k = len(prompts)
            grid = [(sum(hit(r, word) for r in reps[i*k:(i+1)*k]) / k, settings[i]) for i in range(len(settings))]
            best = max(grid)
            rng = random.Random(0)
            rand = [(l, rng.randrange(pipe.W)) for l, _ in units]
            rpk = torch.tensor([pipe.steer_edits([u], 1.0, A["last_act"])[0][2] for u in rand], device=DEVICE)
            rreps = pipe.gen_rows(prompts, rand, torch.stack([rpk * 2.0] * k), torch.ones(k, len(rand), dtype=torch.bool, device=DEVICE))
            bi = grid.index(best)
            res[pname] = {"unsteered": sum(hit(r, word) for r in base) / k, "best": best[0], "best_setting": best[1],
                          "random": sum(hit(r, word) for r in rreps) / k, "sample": reps[bi*k:bi*k+3]}
        out[concept] = res
        f, o = res["friendly"], res["original"]
        print(f"{concept:<24} friendly: unsteered {f['unsteered']:.2f} -> steered best {f['best']:.2f} {f['best_setting']} "
              f"(random {f['random']:.2f}) | original prompts: {o['unsteered']:.2f} -> {o['best']:.2f} (random {o['random']:.2f})", flush=True)
        (ROOT / "results/people_prompts.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
