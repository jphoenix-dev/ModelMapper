"""The dial's auto-selection at scale, scored without any judge API (steering "quantity" loop).

For every recalled concept (CR-2 19 + wave 2 40): candidate neuron sets R1 (late attribution), R2 (any layer),
R4 (says the answer); 2/4 neurons x 0.5/1/1.5/2 x peak; add mode, reply-only. SELECT the best setting on the 20
original neutral prompts, then score that one setting on the 20 held-out neutral prompts. Score = a distinctive answer
word appears AND the reply passes the free fluency proxy (src/fluency_proxy.py: 88.7% agreement with Jev). Also the
same held-out score for the frozen R1-4-neurons-1x default and for random neurons. Every reply saved.

Usage: python src/auto_steer_scale.py
"""
import json
import os
import random
import re
from pathlib import Path

import torch

os.environ["RECALL_VARIANT"] = "V3"
os.environ["ANSWER_DISCOVERY"] = "0"           # keep the answers the earlier runs used
from config import DEFAULT_MODEL  # noqa: E402
import concept_pipeline as cp  # noqa: E402
import pipeline_fast  # noqa: E402
from pipeline_fast import FastPipeline  # noqa: E402
from steer_test import PROMPTS as SELECT  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HELD = json.loads((ROOT / "prompts/neutral_heldout.json").read_text(encoding="utf-8"))
GENERIC = {"the", "mount", "tower", "bridge", "river", "desert", "great", "wall", "games", "table", "engine", "press"}
STRENGTHS = (0.5, 1.0, 1.5, 2.0)
RUNS = [(d, None) for d in os.environ.get("AUTO_DIRS", "results/concepts_v3,results/concepts_w2").split(",")]
OUT_NAME = os.environ.get("AUTO_OUT", "auto_steer_scale.json")


def keys(answer, concept):
    ws = [w.lower() for w in re.findall(r"[A-Za-z]{4,}", answer + " " + concept)]
    ks = [w for w in ws if w not in GENERIC]
    return sorted(set(ks or ws))


def main():
    pipe = FastPipeline(DEFAULT_MODEL, mode="add", reply_only=True)
    print(f"steering: {pipe.mode}, reply_only={pipe.reply_only}", flush=True)
    W_U = pipe.model.lm_head.weight.float(); g = pipe.model.model.norm.weight.float()

    def fluent(text):
        ids = pipe.tok(text, return_tensors="pt").input_ids.to("cuda")
        if ids.shape[1] < 3:
            return False
        with torch.no_grad():
            lp = pipe.model(ids).logits.float().log_softmax(-1)[0, :-1].gather(-1, ids[0, 1:, None]).mean().item()
        t = ids[0].tolist()
        return len(set(t)) / len(t) >= 0.64 and lp >= -2.3 and sum(ord(c) > 127 for c in text) / len(text) < 0.03

    def score(reps, ks):
        return sum(any(k in r.lower() for k in ks) and fluent(r) for r in reps) / len(reps)

    out_p = ROOT / "results" / OUT_NAME
    out = json.loads(out_p.read_text(encoding="utf-8")) if out_p.exists() else {}
    for d, _ in RUNS:
        cp.OUT = ROOT / d; pipeline_fast.OUT = cp.OUT
        for f in sorted(cp.OUT.glob("*__clamp_reply.json")):
            x = json.loads(f.read_text(encoding="utf-8"))
            if not isinstance(x, dict) or x.get("status") != "ok":
                continue
            concept = x["concept"].strip(); tag = f"{Path(d).name}/{concept}"
            if tag in out:
                continue
            spec, tid, recall, _ = pipe.make_prompts(concept)
            if len(recall) < 5:
                continue
            A = pipe.attribute([q for q, _ in recall], tid)
            top = A["top"][:200]; first = tid[0]

            def says(u):
                col = pipe.model.model.layers[u[0]].mlp.down_proj.weight[:, u[1]].float()
                sign = -1.0 if A["last_act"][u[0], u[1]] < 0 else 1.0
                return sign * (W_U[first] @ (col * g)).item()
            cands = {"R1": [u for u in top if u[0] >= pipe.L // 2][:4], "R2": top[:4], "R4": sorted(top, key=lambda u: -says(u))[:4]}
            ks = keys(spec["answer"], concept)

            def run(units, settings, prompts):
                peaks = torch.tensor([pipe.steer_edits([u], 1.0, A["last_act"])[0][2] for u in units], device="cuda")
                rows, vals, act = [], [], []
                for n, s in settings:
                    for p in prompts:
                        rows.append(p); vals.append(peaks * s); act.append(torch.arange(len(units), device="cuda") < n)
                reps = pipe.gen_rows(rows, units, torch.stack(vals), torch.stack(act))
                k = len(prompts)
                return [reps[i*k:(i+1)*k] for i in range(len(settings))]
            sel = []
            settings = [(n, s) for n in (2, 4) for s in STRENGTHS]
            for rule, units in cands.items():
                for (n, s), reps in zip(settings, run(units, settings, SELECT)):
                    sel.append((score(reps, ks), rule, n, s))
            sel.sort(key=lambda t: -t[0])
            best_score, rule, n, s = sel[0]
            held = run(cands[rule], [(n, s)], HELD)[0]
            default = run(cands["R1"], [(4, 1.0)], HELD)[0]
            rng = random.Random(0)
            rand = [(u[0], rng.randrange(pipe.W)) for u in cands["R1"]]
            rnd = run(rand, [(4, 2.0)], HELD)[0]
            unsteered = run(cands["R1"], [(4, 0.0)], HELD)[0]
            out[tag] = {"keys": ks, "selected": {"rule": rule, "n": n, "s": s, "select_score": best_score},
                        "heldout": score(held, ks), "heldout_default_R1": score(default, ks),
                        "heldout_random": score(rnd, ks), "heldout_unsteered": score(unsteered, ks),
                        "samples": held[:3], "all_heldout": held}
            v = out[tag]
            print(f"{tag:<40} pick {rule} n={n} x{s} (select {best_score:.2f}) -> held-out {v['heldout']:.2f} | default R1 "
                  f"{v['heldout_default_R1']:.2f} | random {v['heldout_random']:.2f} | unsteered {v['heldout_unsteered']:.2f}", flush=True)
            out_p.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
