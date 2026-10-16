"""Precompute every reply and number the explainer page shows. The site itself runs no model.

For every concept run that recalled (CR-2 in concepts_v3, waves 2-4), reusing its saved prompt spec (no API calls):
  1. attribution on the saved V3 recall questions -> top-200 neurons (same code as the research runs)
  2. knockout: top-200 re-ranked by measured single-ablation log-odds drop (as logodds_check.py / wave_free.py);
     mean P(answer) and mean log-odds drop after switching off the top k, against random neurons from the same
     layers; plus the model's actual reply to 3 recall questions at each k (greedy)
  3. steering: the concept's auto-selected neuron set (rule R1/R2/R4, n neurons, from auto_steer_scale.json or
     wave_free.json), add mode, reply-only, at strengths 0.25-3x own peak, on 8 fixed everyday prompts
     (6 held-out neutral + 2 person-friendly; none were used to choose the setting)
  4. what each steered neuron writes (direct output projection, top 8 tokens)
Hero extras for the Eiffel Tower: the six-neuron clamp grid James saw in CR-2 on all 20 original prompts, with the
reply-only and prompt-steering versions, and knockout replies on 6 recall questions.

Checks against saved results are printed per concept (attribution top-10 overlap; k for a 3-point log-odds drop).
Resumable: one file per concept in explainer/data/raw/. Reads results/ only; never writes there.

Usage (GPU must be free):  .venv/Scripts/python.exe explainer/gen/make_data.py [--only "Eiffel Tower,coffee"] [--limit N]
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ["RECALL_VARIANT"] = "V3"
os.environ["ANSWER_DISCOVERY"] = "0"          # use the answer wordings the saved runs found; never call Luna

import torch  # noqa: E402

import concept_pipeline as cp  # noqa: E402
import pipeline_fast  # noqa: E402
from atlas import output_words  # noqa: E402
from neurons import steer  # noqa: E402
from pipeline_fast import FastPipeline  # noqa: E402
from steer_test import PROMPTS as ORIGINAL_NEUTRAL  # noqa: E402


def _no_api(*a, **k):
    raise RuntimeError("explainer generation must not call a grader API (spec missing a saved field?)")


cp.oai.chat_json = _no_api

RAW = ROOT / "explainer" / "data" / "raw"
HELD = json.loads((ROOT / "prompts/neutral_heldout.json").read_text(encoding="utf-8"))
PERSON = json.loads((ROOT / "prompts/neutral_person_friendly.json").read_text(encoding="utf-8"))
PROMPTS = [HELD[1], HELD[4], HELD[6], HELD[8], HELD[11], HELD[13], PERSON[13], PERSON[15]]   # none used for selection (people selected on PERSON[:10])
STRENGTHS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
KS = [1, 2, 3, 5, 10, 20]
REPLY_TOKENS = 100
KNOCK_TOKENS = 24
N_KNOCK_Q = 3


def logit(p):
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


def concept_runs():
    """(results dir, concept, selected rule/n/s, saved check values) for every recalled concept run."""
    runs = []
    auto = json.loads((ROOT / "results/auto_steer_scale.json").read_text(encoding="utf-8"))
    lo = {}
    for f in ("results/concepts_v3/logodds_cr2_V3_v2.json", "results/concepts_w2/logodds_w2_V3_v2.json"):
        lo.update({(f.split("/")[1], k): v for k, v in json.loads((ROOT / f).read_text(encoding="utf-8")).items()})
    for tag, v in auto.items():
        d, concept = tag.split("/", 1)
        sel = v["selected"]
        runs.append({"dir": d, "concept": concept, "rule": sel["rule"], "n": sel["n"], "s": sel["s"],
                     "saved_heldout": v["heldout"], "grader": "free local check",
                     "saved_k3": lo.get((d, concept), {}).get("k_LO3_measured")})
    for d in ("concepts_w3", "concepts_w4"):
        for concept, v in json.loads((ROOT / "results" / d / "wave_free.json").read_text(encoding="utf-8")).items():
            if v.get("status") != "ok":
                continue
            st = v["steer"]
            runs.append({"dir": d, "concept": concept, "rule": st["rule"], "n": st["n"], "s": st["s"],
                         "saved_heldout": st["heldout"], "grader": "free local check", "saved_k3": v.get("k_mean"),
                         "category": v.get("category")})
    return runs


class Gen:
    def __init__(self):
        self.pipe = FastPipeline("Qwen/Qwen3-8B", mode="add", reply_only=True)
        p = self.pipe
        self.W_U = p.model.lm_head.weight.float()
        self.g = p.model.model.norm.weight.float()

    # ---- helpers ---------------------------------------------------------
    def use_dir(self, d):
        cp.OUT = ROOT / "results" / d
        pipeline_fast.OUT = cp.OUT

    def says(self, A, first, u):
        col = self.pipe.model.model.layers[u[0]].mlp.down_proj.weight[:, u[1]].float()
        sign = -1.0 if A["last_act"][u[0], u[1]] < 0 else 1.0
        return sign * (self.W_U[first] @ (col * self.g)).item()

    def candidates(self, A, tid, k=4):
        top = A["top"][:200]
        return {"R1": [u for u in top if u[0] >= self.pipe.L // 2][:k], "R2": top[:k],
                "R4": sorted(top, key=lambda u: -self.says(A, tid[0], u))[:k]}

    def writes(self, A, u):
        up, down = output_words(self.pipe.model, self.pipe.tok, u[0], u[1])
        return down if A["last_act"][u[0], u[1]] < 0 else up      # the direction it pushes on these prompts

    def steer_grid(self, units, last_act, prompts, strengths, tokens):
        """Replies for every (strength, prompt); strength 0 is the unsteered model (add mode)."""
        p = self.pipe
        peaks = torch.tensor([p.steer_edits([u], 1.0, last_act)[0][2] for u in units], device="cuda")
        rows, vals, act = [], [], []
        for s in strengths:
            for q in prompts:
                rows.append(q); vals.append(peaks * s); act.append(torch.ones(len(units), dtype=torch.bool, device="cuda"))
        old = pipeline_fast.GEN_TOKENS; pipeline_fast.GEN_TOKENS = tokens
        try:
            reps = p.gen_rows(rows, units, torch.stack(vals), torch.stack(act))
        finally:
            pipeline_fast.GEN_TOKENS = old
        k = len(prompts)
        return {str(s): reps[i*k:(i+1)*k] for i, s in enumerate(strengths)}, [round(x, 3) for x in peaks.tolist()]

    def raw_generate(self, texts, units, active, tokens):
        """Greedy continuation of already-templated chat strings with per-row knockouts (units set to 0)."""
        p = self.pipe
        mode, ro = p.mode, p.reply_only
        p.mode, p.reply_only = "clamp", False                      # knockout = clamp to 0 on every position
        out = []
        try:
            vals = torch.zeros(len(texts), len(units), device="cuda")
            with p.steer_rows(units, vals, active):
                p.tok.padding_side = "left"
                for s in range(0, len(texts), pipeline_fast.GEN_BATCH):
                    p._rows = torch.arange(s, min(s + pipeline_fast.GEN_BATCH, len(texts)), device="cuda")
                    ids = p.tok(texts[s:s + pipeline_fast.GEN_BATCH], return_tensors="pt", padding=True,
                                add_special_tokens=False).to("cuda")
                    with torch.no_grad():
                        g = p.model.generate(**ids, max_new_tokens=tokens, do_sample=False, pad_token_id=p.tok.pad_token_id)
                    out += [p.tok.decode(o[ids.input_ids.shape[1]:], skip_special_tokens=True).strip() for o in g]
        finally:
            p.mode, p.reply_only = mode, ro
        return out

    def knockout(self, A, tid, recall, n_q):
        p = self.pipe
        rp = [q for q, _ in recall]
        pairs = [(q, tid) for q in rp]
        base = [logit(x) for x in p.batch_probs(pairs)]

        def run(edits):
            ps = p.batch_probs(pairs, steer(p.model, edits) if edits else None)
            return ps, [b - logit(x) for b, x in zip(base, ps)]
        top = A["top"][:200]
        single = sorted(top, key=lambda u: -sum(run([(u[0], u[1], 0.0)])[1]))
        rng = random.Random(2)
        rand = [(l, rng.randrange(p.W)) for l, _ in top]
        curve = []
        for k in KS:
            ps, d = run([(l, n, 0.0) for l, n in single[:k]])
            pr, dr = run([(l, n, 0.0) for l, n in rand[:k]])
            curve.append({"k": k, "p_mean": round(sum(ps) / len(ps), 4), "logodds_drop": round(sum(d) / len(d), 2),
                          "p_mean_random": round(sum(pr) / len(pr), 4), "logodds_drop_random": round(sum(dr) / len(dr), 2),
                          "p_per_q": [round(x, 4) for x in ps[:n_q]], "p_per_q_random": [round(x, 4) for x in pr[:n_q]]})
        k3 = next((c["k"] for c in curve if c["logodds_drop"] >= 3), None)
        # replies: the first n_q recall questions (saved order, not cherry-picked) at k = 0 and each k, top vs random
        units = list(dict.fromkeys(single[:max(KS)] + rand[:max(KS)]))
        idx = {u: j for j, u in enumerate(units)}
        texts, act, keys = [], [], []
        for qi, q in enumerate(rp[:n_q]):
            for which, order in (("top", single), ("random", rand)):
                for k in [0] + KS:
                    if k == 0 and which == "random":
                        continue
                    a = torch.zeros(len(units), dtype=torch.bool, device="cuda")
                    for u in order[:k]:
                        a[idx[u]] = True
                    texts.append(q); act.append(a); keys.append((qi, which, k))
        reps = self.raw_generate(texts, units, torch.stack(act), KNOCK_TOKENS)
        replies = {}
        for (qi, which, k), r in zip(keys, reps):
            replies.setdefault(str(qi), {}).setdefault(which, {})[str(k)] = r
        p0 = [round(1 / (1 + math.exp(-b)), 4) for b in base]
        return {"baseline_p_mean": round(sum(p0) / len(p0), 4), "baseline_p_per_q": p0[:n_q], "n_questions": len(rp),
                "questions": rp[:n_q], "order_top": [list(u) for u in single[:max(KS)]],
                "order_random": [list(u) for u in rand[:max(KS)]], "curve": curve, "k_logodds3": k3, "replies": replies}

    # ---- one concept -----------------------------------------------------
    def concept(self, run, hero=False):
        p = self.pipe
        self.use_dir(run["dir"])
        spec, tid, recall, _ = p.make_prompts(run["concept"])
        if len(recall) < 5:
            return {"run": run, "status": "not_recalled_now", "recall_kept": len(recall)}
        A = p.attribute([q for q, _ in recall], tid)
        saved = cp.OUT / f"{cp.slug(run['concept'])}__clamp_reply.json"
        overlap = None
        if saved.exists():
            sj = json.loads(saved.read_text(encoding="utf-8"))
            old = {(u["layer"], u["neuron"]) for u in sj.get("top_neurons", [])[:10]}
            overlap = len(old & set(map(tuple, A["top"][:10])))
        cands = self.candidates(A, tid)
        units = cands[run["rule"]][:run["n"]]
        grid, peaks = self.steer_grid(units, A["last_act"], PROMPTS, STRENGTHS, REPLY_TOKENS)
        ko = self.knockout(A, tid, recall, 6 if hero else N_KNOCK_Q)
        row = {"run": run, "status": "ok", "answer": spec["answer"].strip(), "answer_form": p.answer_form,
               "recall_kept": len(recall), "recall_total": len(spec["recall_v3"]),
               "check": {"top10_overlap_with_saved": overlap, "k_logodds3_now": ko["k_logodds3"], "k_logodds3_saved": run.get("saved_k3")},
               "steer": {"units": [list(u) for u in units], "peaks": peaks, "mode": "add", "reply_only": True,
                         "writes": [self.writes(A, u) for u in units], "prompts": PROMPTS, "replies": grid},
               "knockout": ko}
        if hero:
            late = [u for u in A["top"][:200] if u[0] >= p.L // 2][:6]      # CR-2's grid units: top late-layer
            hs = [0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
            mode, ro = p.mode, p.reply_only
            out = {"units": [list(u) for u in late], "writes": [self.writes(A, u) for u in late], "prompts": ORIGINAL_NEUTRAL,
                   "strengths": hs, "mode": "clamp"}
            try:
                p.mode = "clamp"
                for ro_flag, key in ((True, "reply_only"), (False, "prompt_too")):
                    p.reply_only = ro_flag
                    out[key], _ = self.steer_grid(late, A["last_act"], ORIGINAL_NEUTRAL, hs, 120)
            finally:
                p.mode, p.reply_only = mode, ro
            row["hero"] = out
        return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    runs = concept_runs()
    if args.only:
        want = {x.strip() for x in args.only.split(",")}
        runs = [r for r in runs if r["concept"] in want]
    if args.limit:
        runs = runs[:args.limit]
    print(f"{len(runs)} concept runs", flush=True)
    G = Gen()
    base = RAW / "_unsteered.json"
    if not base.exists():
        G.use_dir("concepts_v3")
        old = pipeline_fast.GEN_TOKENS; pipeline_fast.GEN_TOKENS = REPLY_TOKENS
        try:
            reps = G.pipe.gen_rows(PROMPTS, [(0, 0)], torch.zeros(len(PROMPTS), 1, device="cuda"),
                                   torch.zeros(len(PROMPTS), 1, dtype=torch.bool, device="cuda"))
        finally:
            pipeline_fast.GEN_TOKENS = old
        base.write_text(json.dumps({"prompts": PROMPTS, "replies": reps}, indent=1, ensure_ascii=False), encoding="utf-8")
    for r in runs:
        out = RAW / f"{r['dir']}__{cp.slug(r['concept'])}.json"
        if out.exists():
            continue
        t0 = time.time()
        row = G.concept(r, hero=(r["dir"] == "concepts_v3" and r["concept"] == "Eiffel Tower"))
        row["seconds"] = round(time.time() - t0, 1)
        out.write_text(json.dumps(row, indent=1, ensure_ascii=False), encoding="utf-8")
        c = row.get("check", {})
        print(f"{r['dir']}/{r['concept']:<32} {row['status']} {row['seconds']:>5}s | top10 overlap {c.get('top10_overlap_with_saved')} "
              f"| k(-3 log-odds) now {c.get('k_logodds3_now')} saved {c.get('k_logodds3_saved')}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
