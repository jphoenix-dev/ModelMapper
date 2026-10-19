"""Map ONE concept from its questions: recall -> attribution -> each candidate switched off alone -> controls.

`map_concept(measurer, concept, spec, arrays_path)` returns the concept's record (FORMAT.md): the answer form that
scores best, the questions the model gets right, the 200 neurons with the largest attribution, each switched off
alone and its drop in the answer's exact log-odds measured on every question, 2 random same-layer neurons switched
off the same way (the noise floor), the k curve (how many top neurons together it takes to cut the answer), a
random-neuron k curve, the 50 most-attributed attention heads each switched off, each top-30 neuron's role read from
its weights, and full provenance. It is the code that produced run 1 of the Qwen Neuron Map (loop v2 map_core),
copied unchanged; `pipe` in its body is the Measurer. Large per-concept arrays go to an .npz next to the record.
"""
from __future__ import annotations

import math
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch

from modelmapper import guards
from modelmapper.measure import MIN_P, leaks
from modelmapper.model import DEVICE, mlps, steer

SCHEMA = "loop_v2.0"
KS = [1, 2, 3, 5, 10, 20, 50]          # as wave_free.KS
TOPN, NCAND, NHEADS, LATE_FRAC = 30, 200, 50, 20 / 36


def p_of(lo):
    """P from exact log-odds, overflow-safe (as wave_free.p_of)."""
    if lo >= 0:
        return 1 / (1 + math.exp(-lo))
    e = math.exp(lo); return e / (1 + e)


# run 1's provenance block (map_core.provenance), renamed; its GPU name now comes from guards.gpu_name(), which cannot fail
def _run1_provenance(pipe, code_commit):
    import transformers
    return {"schema": SCHEMA, "model": pipe.model_name, "dtype": str(pipe.model.dtype), "torch": torch.__version__,
            "transformers": transformers.__version__, "python": platform.python_version(),
            "gpu": guards.gpu_name(), "code_commit": code_commit,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def provenance(pipe, code_commit):
    """Run 1's provenance plus the guards' fields (CUDA, GPU memory, device, package version, ...)."""
    return {**_run1_provenance(pipe, code_commit), **guards.provenance_extra(pipe)}


def roles(pipe, units, acts, answer_id):
    """Weights-only reading of each unit's direct push (in the sign it takes at the answer position)."""
    m = pipe.model; WU = m.lm_head.weight.detach(); g = m.model.norm.weight.detach().float(); L = m.config.num_hidden_layers
    out = []
    for (l, n), a in zip(units, acts):
        v = (a * mlps(m)[l].down_proj.weight[:, n].detach().float() * g).to(WU.dtype)
        lg = (WU @ v).float()
        rank = int((lg > lg[answer_id]).sum()) + 1; z = float((lg[answer_id] - lg.mean()) / (lg.std() + 1e-6))
        role = "early" if l < round(LATE_FRAC * L) else "say" if rank <= 10 else "attribute" if z >= 2 else "late-other"
        out.append({"neuron": f"L{l}:{n}", "answer_rank": rank, "answer_z": round(z, 3), "role": role,
                    "job_tokens": [pipe.tok.decode([i]) for i in lg.topk(30).indices.tolist()]})
    return out


def map_concept(pipe, concept, spec, arrays_path, code_commit="unknown"):
    """Returns {"concept", "row" (map_loop row), "rank" (rank_save row, None unless recalled), "v2": {...}}."""
    m = pipe.model; T = {}; prov = provenance(pipe, code_commit); t = time.time()
    chat = [pipe.tok.apply_chat_template([{"role": "user", "content": q + " Answer with just the name."}],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=False) for q in spec["recall_v3"]]
    a = spec["answer"].strip(); base_ = a[4:].strip() if a.lower().startswith("the ") else a
    forms = list(dict.fromkeys([a, a[:1].upper() + a[1:], a.title(), base_, base_[:1].upper() + base_[1:]] + spec.get("answer_forms_extra", [])))
    best = None; form_scores = {}
    for f in forms:
        t_ = pipe.tok.encode(f); ps_ = pipe.batch_probs([(q, t_) for q in chat])
        form_scores[f] = [round(p, 4) for p in ps_]
        score = sum(p >= MIN_P for p in ps_)
        if best is None or score > best[0]:
            best = (score, t_, ps_, f)
    _, tid, ps, form = best
    recall = [(q, round(p, 3)) for q, p, raw in zip(chat, ps, spec["recall_v3"]) if p >= MIN_P and not leaks(raw, spec["answer"])]
    T["G2_recall"] = round(time.time() - t, 2)
    row = {"answer": a, "answer_form": form, "recall": len(recall), "discovered": spec.get("answer_forms_extra"),
           "answer_form_scores": form_scores, "recall_prompts": [q for q, _ in recall]}
    rec = {"concept": concept, "row": row, "rank": None, "v2": {"provenance": prov, "timing_s": T}}
    if len(recall) < 5:
        row["status"] = "not_recalled"; prov["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()); return rec
    rp = [q for q, _ in recall]; pairs = [(q, tid) for q in rp]
    t = time.time(); base = pipe.batch_logodds(pairs); right = [x > 0 for x in base]
    A = pipe.attribute(rp, tid); top = A["top"][:NCAND]; T["G2_attribution"] = round(time.time() - t, 2)

    def per_prompt(units):
        los = pipe.batch_logodds(pairs, steer(m, [(l, n, 0.0) for l, n in units]) if units else None)
        return [b - x for b, x in zip(base, los)], [p_of(x) for x in los]
    t = time.time()
    pp = {u: per_prompt([u])[0] for u in top}                     # unbatched: batching failed its gate and gave x1.1
    T["G2_single_knockouts"] = round(time.time() - t, 2)
    drop1 = {u: sum(pp[u]) / len(rp) for u in top}; single = sorted(top, key=lambda u: -sum(pp[u]))
    # v2: per-concept null - 2 random neurons from the layers of the top-2 candidates, zeroed alone (same procedure)
    t = time.time(); nrng = random.Random(f"null|{concept}"); nulls = []
    for l, _ in single[:2]:
        while True:
            n = nrng.randrange(pipe.W)
            if (l, n) not in drop1:
                nulls.append((l, n)); break
    null_pp = {f"L{l}:{n}": [round(x, 4) for x in per_prompt([(l, n)])[0]] for l, n in nulls}
    T["G2_null"] = round(time.time() - t, 2)

    def kvals(order):
        km = kd = kf = None; drops = []
        for k in KS:
            d, ps2 = per_prompt(order[:k]); mean = sum(d) / len(d); med = sorted(d)[len(d) // 2]
            flip = sum(p < .5 for p, r in zip(ps2, right) if r) / max(1, sum(right))
            drops.append(round(mean, 2))
            km = km or (k if mean >= 3 else None); kd = kd or (k if med >= 3 else None); kf = kf or (k if flip >= .5 else None)
        return km, kd, kf, drops
    t = time.time(); km, kd, kf, drops = kvals(single); T["G2_k_curve"] = round(time.time() - t, 2)
    rng = random.Random(2)
    t = time.time(); _, _, _, rdrops = kvals([(l, rng.randrange(pipe.W)) for l, _ in top]); T["G2_random_curve"] = round(time.time() - t, 2)
    t = time.time(); L_, W_ = pipe.L, pipe.W
    pmax = torch.zeros(L_, W_, device=DEVICE); pmin = torch.zeros(L_, W_, device=DEVICE)
    zmean = torch.zeros(L_, m.config.num_attention_heads * m.config.head_dim, device=DEVICE); att_rows = {}
    ans = torch.zeros(L_, W_, device=DEVICE); resid = []
    for qi, q in enumerate(rp):                      # default (sdpa) kernel, prompt only - exactly as cross_activation.py
        ids = pipe.tok(q, return_tensors="pt").to(DEVICE).input_ids; st = {}; zs = {}
        hk = [mlps(m)[l].down_proj.register_forward_pre_hook(lambda mod, a_, l=l: st.__setitem__(l, a_[0][0, 1:].float())) for l in range(L_)]
        hk += [m.model.layers[l].self_attn.o_proj.register_forward_pre_hook(lambda mod, a_, l=l: zs.__setitem__(l, a_[0][0, -1].float())) for l in range(L_)]
        try:
            with torch.no_grad():
                o = m(input_ids=ids, output_hidden_states=True)                 # v2: residuals (same computation)
        finally:
            for h in hk:
                h.remove()
        resid.append(torch.stack([h_[0, -1] for h_ in o.hidden_states]).half().cpu().numpy())    # [L+1, d_model]
        for l in range(L_):
            ans[l] += st[l][-1] / len(rp); pmax[l] += st[l].max(0).values / len(rp); pmin[l] += st[l].min(0).values / len(rp)
            zmean[l] += zs[l] / len(rp)
    m.set_attn_implementation("eager")               # attention patterns need the eager kernel: separate pass
    try:
        for qi, q in enumerate(rp):
            ids = pipe.tok(q, return_tensors="pt").to(DEVICE).input_ids
            with torch.no_grad():
                att = m(input_ids=ids, output_attentions=True).attentions
            att_rows[f"att{qi}"] = torch.stack([a_[0, :, -1, :] for a_ in att]).half().cpu().numpy()   # [L, H, T]
    finally:
        m.set_attn_implementation("sdpa")
    T["G2_maps_heads"] = round(time.time() - t, 2)
    t = time.time(); ha = A["head_attr"].detach(); Hh = m.config.num_attention_heads
    top_heads = [divmod(i.item(), Hh) for i in ha.flatten().argsort(descending=True)[:NHEADS]]
    head_pp = {}
    for (l, h) in top_heads:
        los = pipe.batch_logodds(pairs, pipe.ablate_heads([(l, h)]))
        head_pp[f"L{l}H{h}"] = [round(b - x, 3) for b, x in zip(base, los)]
    T["G2_head_knockouts"] = round(time.time() - t, 2)
    t = time.time()
    rl = roles(pipe, single[:TOPN], [A["last_act"][l, n].item() for l, n in single[:TOPN]], tid[0])
    T["G2_roles"] = round(time.time() - t, 2)
    Path(arrays_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_path, attr=A["attr"].detach().half().cpu().numpy(), last_act=ans.half().cpu().numpy(),
                        last_act_attr_pass=A["last_act"].detach().half().cpu().numpy(),
                        pmax=pmax.half().cpu().numpy(), pmin=pmin.half().cpu().numpy(), top200=np.array(top, dtype=np.int32),
                        head_attr=ha.half().cpu().numpy(), head_out_answer_pos=zmean.half().cpu().numpy(),
                        resid_answer_pos=np.stack(resid), **att_rows)
    row.update(status="ok", baseline_mean_logodds=round(sum(base) / len(base), 2), base_per_prompt=[round(x, 3) for x in base],
               k_mean=km, k_median=kd, k_flip=kf, drops=drops, random_drops=rdrops)
    cand = [{"neuron": f"L{l}:{n}", "single_drop": round(drop1[(l, n)], 3), "attr_rank": top.index((l, n)),
             "act_answer_pos": round(A["last_act"][l, n].item(), 3)} for l, n in single]
    rec["rank"] = {"answer_token": pipe.tok.decode([tid[0]]), "answer_ids": tid, "recall": len(rp), "k_mean": km, "k_flip": kf,
                   "baseline_mean_logodds": round(sum(base) / len(base), 3), "base_per_prompt": [round(x, 3) for x in base],
                   "prompts": rp, "candidates_all200": cand, "top": cand[:TOPN],
                   "per_prompt_drop_all200": {f"L{l}:{n}": [round(x, 3) for x in pp[(l, n)]] for l, n in single},
                   "matched_original": False, "mismatch": None,
                   "heads_top50_by_attr": [f"L{l}H{h}" for l, h in top_heads], "head_knockout_per_prompt": head_pp,
                   "head_knockout_mean": {k: round(sum(v) / len(v), 3) for k, v in head_pp.items()}}
    rec["v2"].update({"null_single_drops": null_pp, "roles_top30": rl, "arrays": str(Path(arrays_path).name)})
    prov["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return rec
