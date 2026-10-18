"""links:<batch>: links between each concept's top-30 neurons (method and pass rule: map session's LINKS.md, 18 Oct 2026,
fixed before any data). Generalises research/attention's src/indirect_path.py from the top 5 to the top 30.

Inputs (written by make_run --kind links from loop v2 run-1 records): inputs/<batch>/<slug>.json with
  concept, units (top 30 "L<layer>:<index>", run-1 rank order), types (run-1 roles), prompts (chat-templated recalled
  prompts), answer_id (first answer token), base (saved whole-answer log-odds per prompt).
Per prompt (answer position = last prompt token):
  pass 1, batch 1: answer-position activations of every layer holding a top-30 neuron; for each top-30 neuron the
    activation-matched null = the same-layer neuron outside the top 30 whose |activation| is closest to it.
  pass 2, one batch of 61 identical rows: row 0 as is; rows 1-30 one top-30 neuron zeroed at EVERY position; rows 31-60
    its matched null zeroed the same way. Per row: the 30 top neurons' answer-position activations (after the
    knockouts) and the answer's first-token log-odds (float64, as pipeline_fast.batch_logodds).
  gate: a zeroed neuron reads exactly 0 in its own row; row 0's first-token log-odds >= saved whole-answer baseline - 0.5.
Links, for layer(i) < layer(j), per prompt (prompts with |a_j(base)| < 1e-3 skipped for that pair):
  drop_ij = (a_j(base) - a_j(i zeroed)) * sign(a_j(base)) / |a_j(base)|;  null_ij = the same with i's null zeroed.
  passes = |mean drop| >= 0.05 and |mean drop| >= 3 |mean null| and the sign of drop agrees with the mean in >= 2/3 of
  the prompts used and in >= 3 of them. Every pair is stored, passing or not.
Outputs in results/links/<batch>/: records.jsonl (one per concept), arrays.npz (per concept "<k>|acts" [P, 61, 30],
"<k>|logodds" [P, 61]), links.jsonl (one per pair).
"""
from __future__ import annotations

import io
import json
import math

import numpy as np
import torch

from neurons import DEVICE, mlps

N_TOP = 30
MIN_ABS, MIN_DROP, NULL_RATIO, AGREE_FRAC, AGREE_MIN = 1e-3, 0.05, 3.0, 2 / 3, 3


def _unit(u):
    L, n = u.lstrip("L").split(":")
    return int(L), int(n)


def measure_concept(model, tok, spec):
    units = [_unit(u) for u in spec["units"][:N_TOP]]
    top = set(units)
    layers = sorted({L for L, _ in units})
    aid = int(spec["answer_id"])
    P_out, gate_zero, gate_lo = [], 0, 0
    acts_all, lo_all = [], []
    for pi, prompt in enumerate(spec["prompts"]):
        ids = torch.tensor(tok.encode(prompt), device=DEVICE)[None]
        # pass 1: base activations at the answer position, to pick the nulls
        base = {}
        hk = [mlps(model)[L].down_proj.register_forward_pre_hook(lambda m, a, L=L: base.__setitem__(L, a[0][0, -1].float().clone()))
              for L in layers]
        try:
            with torch.no_grad():
                model.model(input_ids=ids)
        finally:
            for h in hk:
                h.remove()
        nulls = []
        for L, n in units:
            d = (base[L].abs() - base[L][n].abs()).abs()
            for L2, n2 in top:
                if L2 == L:
                    d[n2] = float("inf")
            nulls.append((L, int(d.argmin())))
        kills = units + nulls
        R = 1 + len(kills)
        rec = {}

        def mk(L):
            rows = torch.tensor([r for r, (Lk, _) in enumerate(kills, start=1) if Lk == L], device=DEVICE, dtype=torch.long)
            cols = torch.tensor([nk for (Lk, nk) in kills if Lk == L], device=DEVICE, dtype=torch.long)

            def h(m, a):
                x = a[0].clone()
                if len(rows):
                    x[rows, :, cols] = 0.0                                   # zero at every position, one neuron per row
                rec[L] = x[:, -1].float().clone()
                return (x,)
            return h
        hk = [mlps(model)[L].down_proj.register_forward_pre_hook(mk(L)) for L in layers]
        try:
            with torch.no_grad():
                hs = model.model(input_ids=ids.repeat(R, 1)).last_hidden_state[:, -1]
                lp = model.lm_head(hs).double().log_softmax(-1)[:, aid]
        finally:
            for h in hk:
                h.remove()
        logodds = (lp - torch.log(-torch.expm1(lp))).cpu().numpy()
        acts = np.array([[rec[L][r, n].item() for L, n in units] for r in range(R)], dtype=np.float32)   # [61, 30]
        zero_bad = any(acts[1 + k, k] != 0.0 for k in range(len(units)))
        lo_bad = bool(logodds[0] < spec["base"][pi] - 0.5)
        gate_zero += zero_bad; gate_lo += lo_bad
        acts_all.append(acts); lo_all.append(logodds.astype(np.float64))
        P_out.append({"prompt_i": pi, "nulls": [f"L{L}:{n}" for L, n in nulls],
                      "null_base_act": [round(base[L][n].item(), 6) for L, n in nulls],
                      "logodds_base": round(float(logodds[0]), 6), "gate_zero_fail": zero_bad, "gate_logodds_fail": lo_bad})
    return P_out, np.stack(acts_all), np.stack(lo_all), gate_zero, gate_lo


def links_for(spec, acts):
    """acts [P, 61, 30] -> one dict per pair (i earlier layer than j)."""
    units = spec["units"][:N_TOP]; L = [_unit(u)[0] for u in units]; n = len(units)
    out = []
    a0 = acts[:, 0, :]                                                    # [P, 30]
    for i in range(n):
        for j in range(n):
            if not L[i] < L[j]:
                continue
            base = a0[:, j]; ok = np.abs(base) >= MIN_ABS
            if not ok.any():
                out.append({"i": i, "j": j, "unit_i": units[i], "unit_j": units[j], "n_prompts": 0, "passes": False}); continue
            s = np.sign(base[ok]) / np.abs(base[ok])
            drop = (base[ok] - acts[ok, 1 + i, j]) * s
            null = (base[ok] - acts[ok, 1 + n + i, j]) * s
            md, mn = float(drop.mean()), float(null.mean())
            agree = int((np.sign(drop) == np.sign(md)).sum()) if md != 0 else 0
            frac = agree / len(drop)
            passes = abs(md) >= MIN_DROP and abs(md) >= NULL_RATIO * abs(mn) and frac >= AGREE_FRAC and agree >= AGREE_MIN
            out.append({"i": i, "j": j, "unit_i": units[i], "unit_j": units[j], "layer_i": L[i], "layer_j": L[j],
                        "type_i": spec["types"][i] if spec.get("types") else None,
                        "type_j": spec["types"][j] if spec.get("types") else None,
                        "mean_drop": round(md, 6), "mean_null": round(mn, 6), "n_prompts": int(len(drop)),
                        "sign_agree": round(frac, 4), "n_agree": agree, "passes": bool(passes),
                        "direction": ("supports" if md > 0 else "suppresses") if passes else None})
    return out


def links_task(store, task_id, ctx, key):
    batch = task_id.split(":")[1]
    specs = [store.get_json(k) for k in sorted(store.list(f"inputs/{batch}/"))]
    if not specs:
        raise RuntimeError(f"no inputs under inputs/{batch}/")
    model, tok = ctx.model, ctx.tok
    recs, link_lines, arrays = [], [], {}
    gz = gl = nprompts = npass = npairs = 0
    import time
    t0 = time.time()
    for k, sp in enumerate(specs):
        P, acts, lo, z, l_ = measure_concept(model, tok, sp)
        gz += z; gl += l_; nprompts += len(P)
        L = links_for(sp, acts)
        for x in L:
            x["concept"] = sp["concept"]
        npairs += len(L); npass += sum(x["passes"] for x in L)
        link_lines += L
        arrays[f"{k}|acts"] = acts; arrays[f"{k}|logodds"] = lo
        recs.append({"concept": sp["concept"], "array_index": k, "units": sp["units"][:N_TOP], "types": sp.get("types"),
                     "answer_id": sp["answer_id"], "n_prompts": len(P), "gate_zero_fail": z, "gate_logodds_fail": l_,
                     "n_pairs": len(L), "n_links_passing": sum(x["passes"] for x in L), "prompts": P,
                     "source": sp.get("source")})
    store.put(f"{key}/records.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs).encode("utf-8"))
    store.put(f"{key}/links.jsonl", "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in link_lines).encode("utf-8"))
    buf = io.BytesIO(); np.savez_compressed(buf, **arrays); store.put(f"{key}/arrays.npz", buf.getvalue())
    return {"tokens": 0, "concepts": len(specs), "prompts": nprompts, "gate_zero_fail": gz, "gate_logodds_fail": gl,
            "pairs": npairs, "links_passing": npass,
            "sec_per_concept": round((time.time() - t0) / len(specs), 2), "sec_per_prompt": round((time.time() - t0) / max(nprompts, 1), 3)}
