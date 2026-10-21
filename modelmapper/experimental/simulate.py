"""EXPERIMENTAL. Graded simulation: from the label alone, how well does the judge predict how strongly a neuron fires?

Ported from the research run (sim_pass.py and sim_run.py, 21 Oct 2026; as in Bills et al. 2023 and Transluce, built from
this package's pieces). The research run scored every labelled Qwen3-8B neuron's gate and side labels (12,244 gates,
12,018 per side) and 200 fakes per part on web text and met its pre-registered bars (median rank correlation: verified
gates 0.65, sides 0.55; control labels ~0.1; fakes 0.13-0.17). Paths, model and judge are arguments; the numbers are the research run's. Not re-run or calibrated in this package.

  sim_pass  GPU for big models. Stores z_g and z_u of EVERY neuron (and the labelled fakes) at N_POS shared random
            positions in N_DOCS random documents of a readings pass (length >= 64, position >= 8, ordinary tokens;
            seed 2617), so any neuron's activation range can be sampled without another pass.
  sim_run   Judge. Per label (gate, + side, - side): 60 positions = N_TOP of the neuron's top contexts (readings boards;
            fakes: their full profiles from the corpus exam's gather, at most 2 per document) + 4 x N_BIN shared positions
            stratified by the neuron's own value (< p50, p50-p90, p90-p99, >= p99). Value: z_g (gate), a = SiLU(z_g) z_u
            (+ side), -a (- side). The judge rates each passage (ending at the token) from the label alone, 0-100; also
            with 5 control labels (rotation within the board). Score: Spearman(rating, value); also without the top
            contexts (the everyday-firing gap).
Outputs (out_dir): sim_pass_<tag>.npz, sim_pass_<tag>_{g,u}.npy, sim_run_<tag>_<board>.json.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from modelmapper.experimental.realtext import BOARDS, CTX, Docs, band, special_ids

N_TOP, N_BIN, LEFT = 12, 12, 32
FRAME = ("You will see short passages of text. In each, one token is marked [[like this]]. A unit (neuron) inside a language "
         "model is described below. For each passage, judge only from the description how strongly the unit switches on at the "
         "marked token, given the text around it (0 = not at all, 100 = as strongly as it ever does).")
silu = lambda x: x / (1 + np.exp(-x))   # noqa: E731


def spearman(a, b):
    a = np.argsort(np.argsort(a)); b = np.argsort(np.argsort(b))
    return float(np.corrcoef(a, b)[0, 1]) if a.std() and b.std() else 0.0


def sim_pass(model, tok, readings_dir, corpus: str, tag: str, pseudo_set=None, pseudo: dict | None = None, out_dir=None,
             n_docs: int = 400, n_pos: int = 4000, B: int = 8, min_len: int = 64, log=print) -> Path:
    """pseudo: realtext.load_pseudo(...)["gate"] (the fakes whose profiles to store), with pseudo_set their .npz."""
    import torch
    R_dir = Path(readings_dir); od = Path(out_dir or R_dir); od.mkdir(parents=True, exist_ok=True)
    layers = model.model.layers; NL, W = len(layers), layers[0].mlp.gate_proj.out_features
    dev = next(model.parameters()).device; dt = next(model.parameters()).dtype
    special = np.array(sorted(special_ids(tok)))
    D = Docs(R_dir / f"readings_{corpus}_docs.npz"); toks, ln, st = D.toks, D.ln, D.st
    g = np.random.default_rng(2617)
    docs = np.sort(g.choice(np.where(ln >= min_len)[0], min(n_docs, int((ln >= min_len).sum())), replace=False))
    cand = np.concatenate([st[d] + np.arange(8, ln[d]) for d in docs]); cand = cand[~np.isin(toks[cand], special)]
    flat = np.sort(g.choice(cand, min(n_pos, len(cand)), replace=False)); doc_of = np.searchsorted(st, flat, side="right") - 1
    n_pos = len(flat)
    pn = sorted(pseudo or {}) if pseudo_set else []
    PW = {}
    if pn:
        pz = np.load(pseudo_set); pl = {}
        for col, k in enumerate(pn):
            pl.setdefault(pseudo[k][4], []).append((col, pseudo[k][5]))
        PW = {L: (np.array([c for c, _ in v]), torch.tensor(pz["gate"][[r for _, r in v]], device=dev, dtype=dt),
                  torch.tensor(pz["up"][[r for _, r in v]], device=dev, dtype=dt)) for L, v in pl.items()}
    mg = np.lib.format.open_memmap(od / f"sim_pass_{tag}_g.npy", mode="w+", dtype=np.float16, shape=(n_pos, NL, W))
    mu = np.lib.format.open_memmap(od / f"sim_pass_{tag}_u.npy", mode="w+", dtype=np.float16, shape=(n_pos, NL, W))
    PG = np.zeros((n_pos, len(pn)), np.float32); PU = np.zeros_like(PG)
    S = {}

    def on_gate(mod, inp, out, L):
        S[("g", L)] = out[S["rows"], S["cols"]].float().cpu().numpy()
        if L in PW:
            cs, Gw, Uw = PW[L]; x = inp[0][S["rows"], S["cols"]]
            S[("pg", L)] = ((x @ Gw.T).float().cpu().numpy(), (x @ Uw.T).float().cpu().numpy(), cs)

    def on_up(mod, inp, out, L):
        S[("u", L)] = out[S["rows"], S["cols"]].float().cpu().numpy()
    hs = [layers[L].mlp.gate_proj.register_forward_hook(lambda mod, i, o, L=L: on_gate(mod, i, o, L)) for L in range(NL)]
    hs += [layers[L].mlp.up_proj.register_forward_hook(lambda mod, i, o, L=L: on_up(mod, i, o, L)) for L in range(NL)]
    t0 = time.time()
    try:
        with torch.inference_mode():
            for b0 in range(0, len(docs), B):
                bd = docs[b0:b0 + B]; sel = np.where(np.isin(doc_of, bd))[0]
                if not len(sel):
                    continue
                T = int(ln[bd].max()); ids = torch.full((len(bd), T), tok.pad_token_id or 0, device=dev); att = torch.zeros_like(ids)
                for j, d in enumerate(bd):
                    ids[j, :ln[d]] = torch.as_tensor(toks[st[d]:st[d] + ln[d]].astype(np.int64), device=dev); att[j, :ln[d]] = 1
                rmap = {int(d): j for j, d in enumerate(bd)}
                S.clear(); S["rows"] = torch.as_tensor([rmap[int(doc_of[k])] for k in sel], device=dev)
                S["cols"] = torch.as_tensor(flat[sel] - st[doc_of[sel]], device=dev)
                model.model(input_ids=ids, attention_mask=att)
                for L in range(NL):
                    mg[sel, L] = S[("g", L)].astype(np.float16); mu[sel, L] = S[("u", L)].astype(np.float16)
                    if ("pg", L) in S:
                        a_, b_, cs = S[("pg", L)]; PG[np.ix_(sel, cs)] = a_; PU[np.ix_(sel, cs)] = b_
    finally:
        for h in hs:
            h.remove()
    mg.flush(); mu.flush()
    np.savez_compressed(od / f"sim_pass_{tag}.npz", flat=flat, doc=doc_of, docs=docs, corpus=corpus, pseudo_names=np.array(pn),
                        pseudo_g=PG, pseudo_u=PU, seconds=time.time() - t0)
    log(f"sim_pass {tag}: {len(docs)} docs, {n_pos} positions, {time.time() - t0:.0f} s")
    return od / f"sim_pass_{tag}.npz"


def sim_run(tok, readings_dir, tag: str, ptag: str, labels: dict, pseudo: dict | None = None, corpus_tag: str | None = None,
            out_dir=None, n: int = 0, workers: int = 128, log=print) -> dict:
    """labels: realtext.load_labels(...); pseudo: realtext.load_pseudo(...) (needs corpus_tag: the corpus exam gather that
    holds the fakes' full profiles on this corpus). n: a seeded subset of n real labels per board (0 = all)."""
    import torch

    from modelmapper import judge
    R_dir = Path(readings_dir); od = Path(out_dir or R_dir)
    special = special_ids(tok)
    P = np.load(od / f"sim_pass_{ptag}.npz"); pflat = P["flat"]; corpus = str(P["corpus"])
    PG = np.load(od / f"sim_pass_{ptag}_g.npy", mmap_mode="r"); PU = np.load(od / f"sim_pass_{ptag}_u.npy", mmap_mode="r")
    n_layers = PG.shape[1]
    pn = [str(x) for x in P["pseudo_names"]]
    D = Docs(R_dir / f"readings_{corpus}_docs.npz"); toks, st = D.toks, D.st
    R = torch.load(R_dir / f"readings_{corpus}.pt", weights_only=False, map_location="cpu")
    boards = {b: (R[f"{k}_v"].float().numpy(), R[f"{k}_c"].numpy()) for b, k in (("gate", "gate"), ("apos", "act_pos"), ("aneg", "act_neg"))}
    del R
    plab = {b: {k: v for k, v in (pseudo or {}).get(b, {}).items() if k in pn} for b in BOARDS}
    if any(plab.values()):
        cu = json.loads((od / f"corpus_units_{corpus_tag}.json").read_text(encoding="utf-8"))["pseudo"]
        assert pn == sorted(cu), "the simulation pass and the corpus gather hold different fakes"
        pfull = {c: np.load(od / f"corpus_pseudo_{corpus}_{corpus_tag}_{c}.npy") for c in ("g", "u")}
    allf = np.arange(len(toks)); okpos = ((allf - st[np.searchsorted(st, allf, side="right") - 1]) >= 1) & ~np.isin(toks, list(special))
    out_all = {}
    for b in BOARDS:
        keys = [("r", u) for u in sorted(labels.get(b, {}), key=lambda u: tuple(map(int, u[1:].split(":"))))] + [("p", k) for k in sorted(plab[b])]
        if not keys:
            continue
        if n:
            g0 = np.random.default_rng(2618); nr = len(keys) - len(plab[b])
            keys = [keys[i] for i in sorted(g0.choice(nr, min(n, nr), replace=False))] + [k for k in keys if k[0] == "p"][:max(5, n // 6)]
        bv, bc = boards[b]; items = {}
        for kk, k in enumerate(keys):
            g = np.random.default_rng([kk, 2618, len(b)])
            tv, tf = [], []
            if k[0] == "r":
                L, i = map(int, k[1][1:].split(":"))
                zg, zu = np.asarray(PG[:, L, i], np.float32), np.asarray(PU[:, L, i], np.float32)
                per = {}
                for r in range(bv.shape[2]):
                    if len(tf) == N_TOP or bv[L, i, r] < -1e4:
                        break
                    d_, p_ = divmod(int(bc[L, i, r]), CTX); f = int(st[d_] + p_)
                    if int(toks[f]) in special or per.get(d_, 0) >= 2:
                        continue
                    per[d_] = per.get(d_, 0) + 1; tf.append(f); tv.append(float(bv[L, i, r]))
            else:
                j = pn.index(k[1]); zg, zu = P["pseudo_g"][:, j], P["pseudo_u"][:, j]
                fg = np.asarray(pfull["g"][:, j], np.float32); fu = np.asarray(pfull["u"][:, j], np.float32)
                fv = fg if b == "gate" else silu(fg) * fu if b == "apos" else -silu(fg) * fu
                fv = np.where(okpos[:len(fv)], fv, -1e9); per = {}
                for f in np.argsort(-fv)[:2000]:
                    d_ = D.doc(f)
                    if per.get(d_, 0) < 2:
                        per[d_] = per.get(d_, 0) + 1; tf.append(int(f)); tv.append(float(fv[f]))
                    if len(tf) == N_TOP:
                        break
            val = zg if b == "gate" else silu(zg) * zu if b == "apos" else -silu(zg) * zu
            q = np.quantile(val, [0.5, 0.9, 0.99]); bins = [val < q[0], (val >= q[0]) & (val < q[1]), (val >= q[1]) & (val < q[2]), val >= q[2]]
            sel = np.concatenate([g.choice(np.where(m_)[0], min(N_BIN, int(m_.sum())), replace=False) for m_ in bins])
            fl = tf + [int(pflat[s]) for s in sel]; vv = tv + [float(val[s]) for s in sel]
            perm = g.permutation(len(fl))
            items[k] = {"flat": [fl[p] for p in perm], "value": [round(vv[p], 4) for p in perm], "top": [int(p < len(tf)) for p in perm],
                        "text": [D.snip(tok, fl[p], LEFT) for p in perm]}
        L_ = lambda k: (labels[b][k[1]] if k[0] == "r" else plab[b][k[1]])      # noqa: E731
        ctrl = {k: [keys[(j + o) % len(keys)] for o in (2, 3, 5, 17, 19)] for j, k in enumerate(keys)}

        def rate(label, its):
            stt = f"{FRAME}\n\nUnit description: {label}"
            qs = {f"t{q}": {"type": "noul", "instructions": f"Passage: «{it}» How strongly does the unit switch on at the marked token?"}
                  for q, it in enumerate(its)}
            try:
                a = judge.ask(stt, qs); return [100.0 * float(a[f"t{q}"]["noul"]) for q in range(len(its))]
            except Exception:                   # noqa: BLE001  (a failed unit is reported, the rest go on)
                return None
        c0 = judge.cost()
        tasks = [(k, "own", L_(k)[0]) for k in keys] + [(k, f"c{q}", L_(c)[0]) for k in keys for q, c in enumerate(ctrl[k])]
        with ThreadPoolExecutor(workers) as ex:
            res = list(ex.map(lambda t: rate(t[2], items[t[0]]["text"]), tasks))
        Rt = {}
        for (k, kind, _), r in zip(tasks, res):
            Rt.setdefault(k, {})[kind] = r
        out = {}
        for k in keys:
            it = items[k]; v = np.array(it["value"]); o = Rt[k].get("own"); top = np.array(it["top"]) == 1
            cs = [spearman(Rt[k][f"c{q}"], v) for q in range(5) if Rt[k].get(f"c{q}")]
            name = k[1]; lb = L_(k)
            out[name] = {"kind": "real" if k[0] == "r" else "pseudo", "verified": lb[1], "verified_strict": lb[3], "label_run": lb[2],
                         "layer": int(name[1:].split(":")[0]) if k[0] == "r" else lb[4],
                         "rho_own": None if o is None else round(spearman(o, v), 3), "rho_control": round(float(np.median(cs)), 3) if cs else None,
                         "rho_own_not_top": None if o is None else round(spearman(np.array(o)[~top], v[~top]), 3), **it, "ratings": Rt[k]}
        md = lambda x: round(float(np.median(x)), 3) if len(x) else None     # noqa: E731
        summ = {"board": b, "units": len(out), "judge_usd": round(judge.cost() - c0, 3), "judge_failed": sum(e["rho_own"] is None for e in out.values())}
        for g_, f_ in (("verified", lambda e: e["kind"] == "real" and e["verified"]), ("not_verified", lambda e: e["kind"] == "real" and not e["verified"]),
                       ("pseudo", lambda e: e["kind"] == "pseudo")):
            v = [e for e in out.values() if f_(e) and e["rho_own"] is not None]
            summ[g_] = {"n": len(v), "rho_own": md([e["rho_own"] for e in v]), "rho_control": md([e["rho_control"] for e in v if e["rho_control"] is not None]),
                        "rho_not_top": md([e["rho_own_not_top"] for e in v])}
            if g_ != "pseudo":
                summ[g_]["by_band"] = {bb: md([e["rho_own"] for e in v if band(e["layer"], n_layers) == bb]) for bb in ("early", "middle", "late")}
        (od / f"sim_run_{tag}_{b}.json").write_text(json.dumps({"_summary": summ, **out}, ensure_ascii=False), encoding="utf-8")
        log(json.dumps(summ)); out_all[b] = summ
    return out_all
