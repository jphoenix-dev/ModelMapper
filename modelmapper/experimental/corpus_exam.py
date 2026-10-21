"""EXPERIMENTAL. The corpus exam: does a gate label pick out where the neuron fires in ordinary text it never saw?

Ported from the research run's corpus exam v2 (corpus2_pass.py, corpus2_exam.py; 21 Oct 2026, run on 12,244 labelled
Qwen3-8B neurons and 200 fakes over three corpora, ~9.3M tokens; all eight pre-registered predictions met). Paths, model
and judge are arguments; the numbers (caps) are the research run's. Not re-run or calibrated in this package.

Steps, per corpus (each needs a readings pass first: realtext.readings):
  plan     CPU. Units = every neuron with a gate label. Positives = the neuron's top gate contexts in the readings (at most
           N_POS, at most PER_DOC per document, ordinary tokens); candidates = up to N_CAND other occurrences of each
           positive's token (seeded; position >= MIN_POS; not in the neuron's top-K board).
  gather   GPU for big models. Reads z_g and z_u at every planned position, in the readings' documents, and the full z_g /
           z_u profiles of the labelled fake neurons.
  exam     Judge. Per unit: the positives, against three kinds of decoy:
             A  the SAME token in another passage where the gate is closed (z_g <= 0)   (primary: holds the word fixed)
             B  top contexts of the 5 labelled neurons with the nearest labels (Qwen3-Embedding-0.6B)
             C  top contexts of 5 random other labelled neurons
           Passages end AT the marked token (LEFT tokens of left context). The judge rates every item with the unit's label
           and 5 control labels (fixed rotation). Pass per contrast: AUC >= 0.75 and >= 0.20 above the control median.
           A needs >= MIN_A decoys; with fewer the neuron fires on its token almost everywhere: "token-locked", reported,
           not failed. Also a judge-free keyword baseline on A (label word stems found in the passage).
Outputs (out_dir): corpus_plan_<corpus>_<tag>.npz, corpus_units_<tag>.json, corpus_vals_<corpus>_<tag>.npz,
corpus_pseudo_<corpus>_<tag>_{g,u}.npy, corpus_exam_<out>.json.
"""
from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from modelmapper.experimental.realtext import CTX, Docs, band, special_ids

N_POS, PER_DOC, N_CAND, MIN_POS = 10, 2, 6, 8                     # the research run's caps (its CAPS.md #42)
LEFT, N_TOPIC, N_RAND, MIN_A = 32, 5, 5, 10
AUC_MIN, AUC_GAP = 0.75, 0.20
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
FRAME = ("You will see short passages of text. In each, one token is marked [[like this]]. A unit (neuron) inside a language "
         "model is described below. For each passage, judge only from the description whether the unit is likely to switch on "
         "strongly at the marked token, given the text around it.")
STOP = set("that this with from words word tokens token about other their which there these those such into also when where "
           "text terms term like used uses using mostly mainly often especially including related forms form usually".split())


def auc(p, n):
    if not len(p) or not len(n):
        return None
    p, n = np.array(p, float)[:, None], np.array(n, float)[None]
    return float((p > n).mean() + 0.5 * (p == n).mean())


def _layer_of(u):
    return int(u[1:].split(":")[0])


# ------------------------------------------------------------------------------------------------ plan (CPU)
def plan(readings_dir, corpus: str, tag: str, labels: dict, pseudo: dict, tok, out_dir=None, log=print) -> Path:
    """labels: realtext.load_labels(...)["gate"]; pseudo: realtext.load_pseudo(...)["gate"] (or {})."""
    import torch
    R_dir = Path(readings_dir); out = Path(out_dir or R_dir); out.mkdir(parents=True, exist_ok=True)
    special = special_ids(tok)
    units = sorted(labels, key=lambda u: tuple(map(int, u[1:].split(":"))))
    D = Docs(R_dir / f"readings_{corpus}_docs.npz"); toks, ln, st = D.toks, D.ln, D.st
    R = torch.load(R_dir / f"readings_{corpus}.pt", weights_only=False, map_location="cpu")
    gv, gc = R["gate_v"].float().numpy(), R["gate_c"].numpy(); del R
    order = np.argsort(toks, kind="stable"); stoks = toks[order]                   # token id -> its flat positions
    doc_of = np.repeat(np.arange(len(ln)), ln); pos_of = np.arange(len(toks)) - st[doc_of]
    rows = []                                                                         # flat, layer, neuron, role, unit, pidx
    for ui, u in enumerate(units):
        L, i = map(int, u[1:].split(":")); board = set()
        for r in range(gv.shape[2]):
            if gv[L, i, r] < -1e4:
                break
            d_, p_ = divmod(int(gc[L, i, r]), CTX); board.add(int(st[d_] + p_))
        npos, per = 0, {}
        rng = np.random.default_rng([ui, 4242, len(corpus)])
        for r in range(gv.shape[2]):
            if npos == N_POS or gv[L, i, r] < -1e4:
                break
            d_, p_ = divmod(int(gc[L, i, r]), CTX); f = int(st[d_] + p_); t = int(toks[f])
            if t in special or per.get(d_, 0) >= PER_DOC:
                continue
            per[d_] = per.get(d_, 0) + 1
            rows.append((f, L, i, 0, ui, npos))
            a, b = np.searchsorted(stoks, [t, t + 1]); occ = order[a:b]
            if len(occ) > 16 * N_CAND:                                   # seeded subsample first (common tokens), then filter
                occ = rng.choice(occ, 16 * N_CAND, replace=False)
            else:
                occ = rng.permutation(occ)
            occ = [int(o) for o in occ if pos_of[o] >= MIN_POS and int(o) not in board][:N_CAND]
            for o in occ:
                rows.append((o, L, i, 1, ui, npos))
            npos += 1
    a = np.array(rows, dtype=np.int64).reshape(-1, 6); a = a[np.argsort(a[:, 0], kind="stable")]
    np.savez_compressed(out / f"corpus_plan_{corpus}_{tag}.npz", flat=a[:, 0], layer=a[:, 1].astype(np.int16), neuron=a[:, 2].astype(np.int16),
                        role=a[:, 3].astype(np.int8), unit=a[:, 4].astype(np.int32), pidx=a[:, 5].astype(np.int16))
    ps = {k: [v[0], v[1], v[4], v[5]] for k, v in (pseudo or {}).items()}       # label, verified, layer, row in the set
    (out / f"corpus_units_{tag}.json").write_text(json.dumps({"units": units, "labels": {u: list(labels[u]) for u in units}, "pseudo": ps},
                                                            ensure_ascii=False), encoding="utf-8")
    log(f"plan {corpus}: {len(units)} units, {int((a[:, 3] == 0).sum())} positives, {int((a[:, 3] == 1).sum())} candidates")
    return out / f"corpus_plan_{corpus}_{tag}.npz"


# ------------------------------------------------------------------------------------------------ gather (GPU)
def gather(model, tok, readings_dir, corpus: str, tag: str, pseudo_set=None, out_dir=None, B: int = 8, max_docs: int = 0, log=print) -> Path:
    """z_g and z_u at every planned position, in the readings' documents and batches; the fakes' full profiles."""
    import torch
    R_dir = Path(readings_dir); out = Path(out_dir or R_dir)
    layers = model.model.layers; NL = len(layers); dev = next(model.parameters()).device
    P = np.load(out / f"corpus_plan_{corpus}_{tag}.npz"); flat, lay, neu = P["flat"], P["layer"].astype(np.int64), P["neuron"].astype(np.int64)
    D = Docs(R_dir / f"readings_{corpus}_docs.npz"); toks, ln, st = D.toks, D.ln, D.st
    nd = len(ln) if not max_docs else min(max_docs, len(ln))
    doc_of = np.searchsorted(st, flat, side="right") - 1; batch_of = doc_of // B
    key = batch_of * NL + lay; srt = np.argsort(key, kind="stable"); kk = key[srt]
    ps = json.loads((out / f"corpus_units_{tag}.json").read_text(encoding="utf-8"))["pseudo"]
    pnames = sorted(ps) if pseudo_set else []
    pl = {}
    for col, k in enumerate(pnames):
        pl.setdefault(ps[k][2], []).append((col, ps[k][3]))
    dt = next(model.parameters()).dtype
    if pnames:
        pz = np.load(pseudo_set)
        PG = {l: (torch.tensor([c for c, _ in v], device=dev), torch.tensor(pz["gate"][[j for _, j in v]], device=dev, dtype=dt),
                  torch.tensor(pz["up"][[j for _, j in v]], device=dev, dtype=dt)) for l, v in pl.items()}
    else:
        PG = {}
    ntok = int(st[nd - 1] + ln[nd - 1]) if nd else 0
    mm = {k: np.lib.format.open_memmap(out / f"corpus_pseudo_{corpus}_{tag}_{k}.npy", mode="w+", dtype=np.float16, shape=(ntok, len(pnames)))
          for k in ("g", "u")}
    zg = np.full(len(flat), np.nan, np.float32); zu = np.full(len(flat), np.nan, np.float32)
    S = {}
    R_row = torch.as_tensor(doc_of % B, device=dev); R_pos = torch.as_tensor(flat - st[doc_of], device=dev)
    R_neu = torch.as_tensor(neu, device=dev); srt_t = torch.as_tensor(srt, device=dev)

    def gate_hook(mod, inp, out_, l):
        k = S["b"] * NL + l; a_, b_ = np.searchsorted(kk, [k, k + 1])
        if b_ > a_:
            it = srt_t[a_:b_]; S[("g", l)] = (srt[a_:b_], it, out_[R_row[it], R_pos[it], R_neu[it]].float())
        if l in PG:
            cols, G, U = PG[l]; x = inp[0]
            S["pg"][:, :, cols] = (x @ G.T).float(); S["pu"][:, :, cols] = (x @ U.T).float()

    def up_hook(mod, inp, out_, l):
        if ("g", l) in S:
            idx, it, g = S.pop(("g", l)); u = out_[R_row[it], R_pos[it], R_neu[it]].float()
            S.setdefault("got", []).append((idx, g, u))
    hs = [layers[l].mlp.gate_proj.register_forward_hook(lambda mod, i, o, l=l: gate_hook(mod, i, o, l)) for l in range(NL)]
    hs += [layers[l].mlp.up_proj.register_forward_hook(lambda mod, i, o, l=l: up_hook(mod, i, o, l)) for l in range(NL)]
    t0 = time.time(); done = 0
    try:
        with torch.inference_mode():
            for b in range((nd + B - 1) // B):
                d0, d1 = b * B, min(nd, (b + 1) * B); docs = [toks[st[d]:st[d] + ln[d]] for d in range(d0, d1)]
                T = max(map(len, docs)); ids = torch.full((len(docs), T), tok.pad_token_id or 0, device=dev); att = torch.zeros_like(ids)
                for j, dd in enumerate(docs):
                    ids[j, :len(dd)] = torch.as_tensor(dd.astype(np.int64), device=dev); att[j, :len(dd)] = 1
                S.clear(); S["b"] = b
                S["pg"] = torch.zeros((len(docs), T, len(pnames)), device=dev); S["pu"] = torch.zeros_like(S["pg"])
                model.model(input_ids=ids, attention_mask=att)
                for idx, g, u in S.get("got", []):
                    zg[idx] = g.cpu().numpy(); zu[idx] = u.cpu().numpy()
                for j, dd in enumerate(docs):
                    f0 = int(st[d0 + j]); mm["g"][f0:f0 + len(dd)] = S["pg"][j, :len(dd)].half().cpu().numpy(); mm["u"][f0:f0 + len(dd)] = S["pu"][j, :len(dd)].half().cpu().numpy()
                done += sum(map(len, docs))
                if dev.type == "cuda" and b % 25 == 0:
                    torch.cuda.empty_cache()
                if b % 100 == 0:
                    log(f"{done:,} tokens, {time.time() - t0:.0f} s")
    finally:
        for h in hs:
            h.remove()
    for k in mm:
        mm[k].flush()
    np.savez_compressed(out / f"corpus_vals_{corpus}_{tag}.npz", zg=zg, zu=zu, n_docs=nd, n_tokens=done, seconds=time.time() - t0,
                        pseudo_names=np.array(pnames))
    log(f"gather {corpus}: {nd:,} docs, {done:,} tokens, {time.time() - t0:.0f} s, gathered {int(np.isfinite(zg).sum()):,} / {len(zg):,}")
    return out / f"corpus_vals_{corpus}_{tag}.npz"


# ------------------------------------------------------------------------------------------------ exam (judge)
def embed(texts, model_name: str = EMBED_MODEL):
    """Unit-length label embeddings (last-token pooling), for the same-topic decoys."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    qt = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    qm = AutoModel.from_pretrained(model_name, dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
    out = []
    with torch.inference_mode():
        for k in range(0, len(texts), 128):
            b = qt(texts[k:k + 128], padding=True, truncation=True, max_length=160, return_tensors="pt").to(dev)
            out.append(torch.nn.functional.normalize(qm(**b).last_hidden_state[:, -1].float(), dim=-1).cpu().numpy())
    del qm
    return np.concatenate(out)


def exam(tok, readings_dir, tag: str, out: str, corpora, out_dir=None, which: str = "all", n_real: int = 100, n_pseudo: int = 30,
         workers: int = 128, judge_items: bool = True, log=print) -> dict:
    """Builds every unit's items over the corpora and has the judge rate them (see the module docstring).
    which: "all", or "smoke" (n_real real units stratified by band x verified, n_pseudo fakes)."""
    from modelmapper import judge
    R_dir = Path(readings_dir); od = Path(out_dir or R_dir)
    special = special_ids(tok)
    U = json.loads((od / f"corpus_units_{tag}.json").read_text(encoding="utf-8"))
    units, labs, ps = U["units"], U["labels"], U["pseudo"]
    pn = sorted(ps)
    n_layers = max([_layer_of(u) for u in units] + [0]) + 1
    ef = od / f"corpus_label_emb_{tag}.npz"                                         # label embeddings, cached
    if ef.exists():
        E = np.load(ef)["e"]
    else:
        E = embed([labs[u][0] for u in units] + [ps[k][0] for k in pn]); np.savez_compressed(ef, e=E)
    ER, EP = E[:len(units)], E[len(units):]
    rng0 = np.random.default_rng(2614)
    if which == "smoke":
        strata = {}
        for ui, u in enumerate(units):
            strata.setdefault((band(_layer_of(u), n_layers), labs[u][1]), []).append(ui)
        per = max(1, n_real // len(strata))
        real_ids = sorted(int(x) for s in sorted(strata) for x in rng0.choice(strata[s], min(per, len(strata[s])), replace=False))
        p_ids = sorted(int(x) for x in rng0.choice(len(pn), min(n_pseudo, len(pn)), replace=False)) if pn else []
    else:
        real_ids, p_ids = list(range(len(units))), list(range(len(pn)))
    keys = [("r", i) for i in real_ids] + [("p", j) for j in p_ids]
    lab_of = lambda k: labs[units[k[1]]][0] if k[0] == "r" else ps[pn[k[1]]][0]      # noqa: E731
    ctrl = {k: [keys[(n + o) % len(keys)] for o in (2, 3, 5, 17, 19)] for n, k in enumerate(keys)}
    items = {k: {"text": [], "role": [], "corpus": [], "flat": []} for k in keys}
    stats = {k: {"positives": 0, "A": 0} for k in keys}
    for corpus in corpora:
        D = Docs(R_dir / f"readings_{corpus}_docs.npz"); toks, ln, st = D.toks, D.ln, D.st
        P = np.load(od / f"corpus_plan_{corpus}_{tag}.npz"); V = np.load(od / f"corpus_vals_{corpus}_{tag}.npz")
        flat, role, unit, pidx, zg = P["flat"], P["role"], P["unit"], P["pidx"], V["zg"]
        o = np.lexsort((pidx, unit)); flat, role, unit, pidx, zg = flat[o], role[o], unit[o], pidx[o], zg[o]
        bounds = np.searchsorted(unit, np.arange(len(units) + 1))
        pos_of, A_of = {}, {}
        for ui in range(len(units)):
            a, b = bounds[ui], bounds[ui + 1]
            if a == b:
                continue
            pos_of[ui] = [int(f) for f, r in zip(flat[a:b], role[a:b]) if r == 0]
            if ("r", ui) in stats:
                g = np.random.default_rng([ui, 77, len(corpus)]); A_of[ui] = []
                for q in sorted(set(pidx[a:b].tolist())):
                    m = (pidx[a:b] == q) & (role[a:b] == 1) & (zg[a:b] <= 0)
                    c_ = flat[a:b][m]
                    A_of[ui].append(int(c_[g.integers(len(c_))]) if len(c_) else None)
        if p_ids:                                                   # fakes: positives and A decoys from their full profiles
            pg = np.load(od / f"corpus_pseudo_{corpus}_{tag}_g.npy")
            doc_of = np.repeat(np.arange(len(ln)), ln); pin = np.arange(len(toks)) - st[doc_of]
            order = np.argsort(toks, kind="stable"); stoks = toks[order]
            okpos = (pin >= 1) & ~np.isin(toks, list(special))
            for j in p_ids:
                col = np.asarray(pg[:, j], dtype=np.float32); cand = np.where(okpos[:len(col)])[0]
                top = cand[np.argsort(-col[cand])[:2000]]; picks, per = [], {}
                for f in top:
                    d = int(doc_of[f])
                    if per.get(d, 0) < PER_DOC:
                        per[d] = per.get(d, 0) + 1; picks.append(int(f))
                    if len(picks) == N_POS:
                        break
                g = np.random.default_rng([j, 78, len(corpus)]); A = []
                for f in picks:
                    t = toks[f]; a, b = np.searchsorted(stoks, [t, t + 1]); occ = order[a:b]
                    occ = occ[(pin[occ] >= MIN_POS) & (col[occ] <= 0) & (occ != f)]
                    A.append(int(occ[g.integers(len(occ))]) if len(occ) else None)
                pos_of[("p", j)] = picks; A_of[("p", j)] = A
        real_with = [ui for ui in range(len(units)) if pos_of.get(ui)]
        for k in keys:
            e = ER[k[1]] if k[0] == "r" else EP[k[1]]
            P_ = pos_of.get(k[1] if k[0] == "r" else k, []); A_ = A_of.get(k[1] if k[0] == "r" else k, [])
            if not P_:
                continue
            g = np.random.default_rng([(2 * k[1] + (k[0] == "p")), len(corpus), 79])
            banned = {c[1] for c in ctrl[k] if c[0] == "r"} | ({k[1]} if k[0] == "r" else set())
            sims = ER[real_with] @ e; nb = [real_with[x] for x in np.argsort(-sims) if real_with[x] not in banned][:N_TOPIC]
            rest = [u for u in real_with if u not in banned and u not in nb]
            rd = [rest[x] for x in g.choice(len(rest), min(N_RAND, len(rest)), replace=False)]
            add = [(f, "P") for f in P_] + [(f, "A") for f in A_ if f is not None] + \
                  [(pos_of[u][g.integers(len(pos_of[u]))], "B") for u in nb] + [(pos_of[u][g.integers(len(pos_of[u]))], "C") for u in rd]
            for f, r in add:
                items[k]["text"].append(D.snip(tok, f, LEFT)); items[k]["role"].append(r); items[k]["corpus"].append(corpus); items[k]["flat"].append(int(f))
            stats[k]["positives"] += len(P_); stats[k]["A"] += sum(f is not None for f in A_)
    jobs = []
    for k in keys:
        n = len(items[k]["text"])
        if n:
            perm = np.random.default_rng([(2 * k[1] + (k[0] == "p")), 80]).permutation(n)
            for f in ("text", "role", "corpus", "flat"):
                items[k][f] = [items[k][f][q] for q in perm]
            jobs.append(k)
    name_of = lambda k: units[k[1]] if k[0] == "r" else pn[k[1]]                   # noqa: E731
    if not judge_items:                                                              # items only (positions saved), no judge calls
        (od / f"corpus_items_{out}.json").write_text(json.dumps({name_of(k): items[k] for k in jobs}, ensure_ascii=False), encoding="utf-8")
        log(f"items only: {len(jobs)} units"); return {"items_only": len(jobs)}

    def rate(label, its):
        stt = f"{FRAME}\n\nUnit description: {label}"
        qs = {f"t{q}": {"type": "noul", "instructions": f"Passage: «{it}» Would the unit switch on strongly at the marked token?"}
              for q, it in enumerate(its)}
        try:
            a = judge.ask(stt, qs); return [100.0 * float(a[f"t{q}"]["noul"]) for q in range(len(its))]
        except Exception:                       # noqa: BLE001  (a failed unit is reported, the rest go on)
            return None
    t0 = time.time(); c0 = judge.cost()
    tasks = [(k, "own", lab_of(k)) for k in jobs] + [(k, f"c{q}", lab_of(c)) for k in jobs for q, c in enumerate(ctrl[k])]
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(lambda t: rate(t[2], items[t[0]]["text"]), tasks))
    R = {}
    for (k, kind, _), r in zip(tasks, res):
        R.setdefault(k, {})[kind] = r

    def stems(label):
        return {w[:5] for w in re.findall(r"[a-z]+", label.lower()) if len(w) >= 4 and w not in STOP}
    outd = {}
    for k in jobs:
        rl = np.array(items[k]["role"]); r_ = R[k]; name = name_of(k)
        if not r_.get("own"):
            outd[name] = {"error": "judge failed"}; continue
        own = np.array(r_["own"]); ctl = [np.array(r_[f"c{q}"]) for q in range(5) if r_.get(f"c{q}")]
        sm = stems(lab_of(k)); kw = np.array([sum(s in t.lower() for s in sm) for t in items[k]["text"]], float)
        e = {"kind": "pseudo" if k[0] == "p" else "real", "label": lab_of(k), "positives": stats[k]["positives"], "n_A": stats[k]["A"]}
        if k[0] == "r":
            L = _layer_of(name)
            e.update({"layer": L, "band": band(L, n_layers), "verified": labs[name][1], "label_run": labs[name][2], "verified_strict": labs[name][3]})
        else:
            e.update({"layer": ps[name][2], "band": band(ps[name][2], n_layers), "verified": ps[name][1]})
        for X in ("A", "B", "C"):
            a_o = auc(own[rl == "P"], own[rl == X]); cs = [auc(c[rl == "P"], c[rl == X]) for c in ctl]
            cs = [c for c in cs if c is not None]; a_c = float(np.median(cs)) if cs else None
            ok = a_o is not None and a_c is not None and a_o >= AUC_MIN and a_o - a_c >= AUC_GAP
            e[X] = {"auc_own": None if a_o is None else round(a_o, 3), "auc_control": None if a_c is None else round(a_c, 3), "pass": bool(ok)}
        e["token_locked"] = stats[k]["A"] < MIN_A
        if e["token_locked"]:
            e["A"]["pass"] = None
        e["keyword_auc_A"] = None if e["token_locked"] else (lambda v: None if v is None else round(v, 3))(auc(kw[rl == "P"], kw[rl == "A"]))
        e.update({"items": items[k]["text"], "roles": items[k]["role"], "corpora": items[k]["corpus"], "flats": items[k]["flat"], "ratings": r_,
                  "control_units": [name_of(c) for c in ctrl[k]]})
        outd[name] = e
    V_ = [e for e in outd.values() if "error" not in e]
    rate_ = lambda g, X: (lambda v: f"{sum(v)}/{len(v)} = {np.mean(v):.3f}" if v else "0/0")([e[X]["pass"] for e in g if e[X]["pass"] is not None])   # noqa: E731
    med = lambda xs: round(float(np.median(xs)), 3) if len(xs) else None                                                                        # noqa: E731
    summ = {"tag": tag, "out": out, "corpora": list(corpora), "units_judged": len(V_), "judge_failed": [n for n, e in outd.items() if "error" in e],
            "judge_usd": round(judge.cost() - c0, 4), "seconds": round(time.time() - t0),
            "control_median": {X: med([e[X]["auc_control"] for e in V_ if e[X]["auc_control"] is not None]) for X in "ABC"},
            "keyword_auc_A_median": med([e["keyword_auc_A"] for e in V_ if e["keyword_auc_A"] is not None])}
    for nm, f in (("real_verified", lambda e: e["kind"] == "real" and e["verified"]), ("real_not_verified", lambda e: e["kind"] == "real" and not e["verified"]),
                  ("pseudo", lambda e: e["kind"] == "pseudo")):
        g = [e for e in V_ if f(e)]
        summ[nm] = {"n": len(g), "token_locked": sum(e["token_locked"] for e in g), **{f"pass_{X}": rate_(g, X) for X in "ABC"},
                    **{f"median_auc_{X}": med([e[X]["auc_own"] for e in g if e[X]["auc_own"] is not None]) for X in "ABC"}}
        if nm != "pseudo":
            summ[nm]["by_band_pass_A"] = {b: rate_([e for e in g if e["band"] == b], "A") for b in ("early", "middle", "late")}
    (od / f"corpus_exam_{out}.json").write_text(json.dumps({"_summary": summ, **outd}, ensure_ascii=False, indent=1), encoding="utf-8")
    log(json.dumps(summ, indent=1, ensure_ascii=False))
    return summ
