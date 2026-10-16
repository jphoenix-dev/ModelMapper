"""Judge-free coverage wave: recall (+ answer discovery), log-odds concentration, auto-selected steering.

No Jev. Luna writes each concept's recall questions (+ V3 rewrite + answer-discovery check, ~3 calls). Per concept:
  1. recall: V3 question format, answer discovery on (made-up controls validated, ISSUES #31)
  2. concentration: top-200 attributed re-ranked by measured single-ablation; k_mean / k_median / k_flip on log-odds
     (same definitions as logodds_check.py v2) + random same-layer neurons
  3. steering: R1/R2/R4 x 2/4 neurons x 0.5-2x (add, reply-only) selected on the 20 original neutral prompts, scored
     on 20 held-out neutral prompts; score = distinctive answer word AND free fluency proxy. People use the
     person-friendly prompts for both. Reported with the free score's known inflation (RESULTS: vs Jev r = 0.65).

Usage: CONCEPT_OUT_DIR=results/concepts_w3 python src/wave_free.py prompts/concepts_wave3.json
"""
import json
import math
import os
import random
import re
import sys
from pathlib import Path

import torch

os.environ["RECALL_VARIANT"] = "V3"
os.environ.setdefault("ANSWER_DISCOVERY", "1")
import concept_pipeline as cp  # noqa: E402
from config import DEFAULT_MODEL  # noqa: E402
from neurons import steer  # noqa: E402
from pipeline_fast import FastPipeline  # noqa: E402
from steer_test import PROMPTS as SELECT  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HELD = json.loads((ROOT / "prompts/neutral_heldout.json").read_text(encoding="utf-8"))
FRIENDLY = json.loads((ROOT / "prompts/neutral_person_friendly.json").read_text(encoding="utf-8"))
KS = [1, 2, 3, 5, 10, 20, 50]
GENERIC = {"the", "mount", "tower", "bridge", "river", "desert", "great", "wall", "games", "table", "engine", "press",
           "city", "tree", "speed", "light", "york"}


def logit(p):
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


def main():
    wave = json.loads((ROOT / sys.argv[1]).read_text(encoding="utf-8"))
    people_cat = {c for cat, cs in wave["concepts"].items() if "people" in cat for c in cs}
    pipe = FastPipeline(os.environ.get("MODEL", DEFAULT_MODEL), mode="add", reply_only=True)
    print(f"steering: {pipe.mode}, reply_only={pipe.reply_only}; out {cp.OUT}", flush=True)
    W_U = pipe.model.lm_head.weight.float(); g = pipe.model.model.norm.weight.float()

    def fluent(text):
        ids = pipe.tok(text, return_tensors="pt").input_ids.to("cuda")
        if ids.shape[1] < 3:
            return False
        with torch.no_grad():
            lp = pipe.model(ids).logits.float().log_softmax(-1)[0, :-1].gather(-1, ids[0, 1:, None]).mean().item()
        t = ids[0].tolist()
        return len(set(t)) / len(t) >= 0.64 and lp >= -2.3 and sum(ord(c) > 127 for c in text) / len(text) < 0.03

    out_p = cp.OUT / "wave_free.json"
    out = json.loads(out_p.read_text(encoding="utf-8")) if out_p.exists() else {}
    for cat, concepts in wave["concepts"].items():
        for concept in concepts:
            if concept in out:
                continue
            spec, tid, recall, _ = pipe.make_prompts(concept)
            row = {"category": cat, "answer": spec["answer"].strip(), "answer_form": getattr(pipe, "answer_form", None),
                   "recall": len(recall), "discovered": spec.get("answer_forms_extra")}
            if len(recall) < 5:
                row["status"] = "not_recalled"; out[concept] = row
                print(f"{concept:<24} not recalled ({len(recall)}/20); discovered {row['discovered']}", flush=True)
                out_p.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8"); continue
            rp = [q for q, _ in recall]; pairs = [(q, tid) for q in rp]
            base = [logit(p) for p in pipe.batch_probs(pairs)]; right = [b > 0 for b in base]
            A = pipe.attribute(rp, tid); top = A["top"][:200]

            def per_prompt(units):
                ps = pipe.batch_probs(pairs, steer(pipe.model, [(l, n, 0.0) for l, n in units]) if units else None)
                return [b - logit(p) for b, p in zip(base, ps)], ps
            single = sorted(top, key=lambda u: -sum(per_prompt([u])[0]))

            def kvals(order):
                km = kd = kf = None; drops = []
                for k in KS:
                    d, ps = per_prompt(order[:k]); mean = sum(d) / len(d); med = sorted(d)[len(d) // 2]
                    flip = sum(p < .5 for p, r in zip(ps, right) if r) / max(1, sum(right))
                    drops.append(round(mean, 2))
                    km = km or (k if mean >= 3 else None); kd = kd or (k if med >= 3 else None); kf = kf or (k if flip >= .5 else None)
                return km, kd, kf, drops
            km, kd, kf, drops = kvals(single)
            rng = random.Random(2)
            _, _, _, rdrops = kvals([(l, rng.randrange(pipe.W)) for l, _ in top])
            row.update(status="ok", baseline_mean_logodds=round(sum(base) / len(base), 2), k_mean=km, k_median=kd, k_flip=kf,
                       drops=drops, random_drops=rdrops)
            # steering
            first = tid[0]

            def says(u):
                col = pipe.model.model.layers[u[0]].mlp.down_proj.weight[:, u[1]].float()
                sign = -1.0 if A["last_act"][u[0], u[1]] < 0 else 1.0
                return sign * (W_U[first] @ (col * g)).item()
            cands = {"R1": [u for u in top if u[0] >= pipe.L // 2][:4], "R2": top[:4], "R4": sorted(top, key=lambda u: -says(u))[:4]}
            words = [w.lower() for w in re.findall(r"[A-Za-zé]{4,}", row["answer"] + " " + concept + " " + " ".join(row["discovered"] or []))]
            keys = sorted({w for w in words if w not in GENERIC} or set(words) or {row["answer"].lower()})
            sel_p, held_p = (FRIENDLY[:10], FRIENDLY[10:]) if concept in people_cat else (SELECT, HELD)

            def run(units, settings, prompts):
                peaks = torch.tensor([pipe.steer_edits([u], 1.0, A["last_act"])[0][2] for u in units], device="cuda")
                rows, vals, act = [], [], []
                for n, s in settings:
                    for p in prompts:
                        rows.append(p); vals.append(peaks * s); act.append(torch.arange(len(units), device="cuda") < n)
                reps = pipe.gen_rows(rows, units, torch.stack(vals), torch.stack(act))
                k = len(prompts)
                return [reps[i*k:(i+1)*k] for i in range(len(settings))]

            def score(reps):
                return sum(any(k in r.lower() for k in keys) and fluent(r) for r in reps) / len(reps)
            settings = [(n, s) for n in (2, 4) for s in (0.5, 1.0, 1.5, 2.0)]
            sel = sorted(((score(reps), rule, n, s) for rule, units in cands.items()
                          for (n, s), reps in zip(settings, run(units, settings, sel_p))), key=lambda t: -t[0])
            bs, rule, n, s = sel[0]
            held = run(cands[rule], [(n, s)], held_p)[0]
            rand = [(u[0], rng.randrange(pipe.W)) for u in cands["R1"]]
            row["steer"] = {"keys": keys, "rule": rule, "n": n, "s": s, "select": bs, "heldout": score(held),
                            "heldout_default_R1": score(run(cands["R1"], [(4, 1.0)], held_p)[0]),
                            "heldout_random": score(run(rand, [(4, 2.0)], held_p)[0]), "samples": held[:3], "all_heldout": held}
            out[concept] = row
            st = row["steer"]
            print(f"{concept:<24} recall {len(recall)}/20 | k_mean {km} k_flip {kf} (base LO {row['baseline_mean_logodds']}) | "
                  f"steer {rule} n={n} x{s}: held-out {st['heldout']:.2f} (default {st['heldout_default_R1']:.2f}, random "
                  f"{st['heldout_random']:.2f})", flush=True)
            out_p.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
