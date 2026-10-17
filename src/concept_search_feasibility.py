"""Can cheap, model-internal profiles tell concept neurons apart from other neurons? (GPU for the weights; no API)

James (17 Oct): search for neurons that look like the concept neurons we've verified, instead of grinding every neuron
- if concept-ness is predictable from cheap profiles. No labels imposed: only numbers measured from Qwen.
Profile of every one of the 442,368 neurons (both signs - a neuron's 'firing' can be its positive or negative side):
  atlas (10M web tokens, top-40 / bottom-40 firings): peak |activation|, spikiness (peak / mean of the 40), decay (40th
    / peak), distinct firing tokens / 40, distinct documents / 40, share of firing tokens that start a word, share
    that start a capitalised word, share with the gate open; stats: corpus max and min
  weights: its push through the final norm weight and the unembedding, each sign: max z over the vocabulary, share of
    the 50 most-pushed tokens that start a word / start a capitalised word / are Latin letters; norms of its
    down / gate / up vectors, cosine(gate, up)
Positives: every neuron in a mapped concept's top-5 (causally verified). Negatives: 5 random neurons per positive from
the same layer, in no concept's top-30 (some may be concept neurons for concepts never mapped - that only lowers the
scores). Model: logistic regression (standardised features, L2), fitted in torch.
Tests (criteria stated before running):
  (1) 5-fold cross-validated AUC on the pre-coverage-4 positives: >= 0.85
  (2) out of sample: fit on pre-coverage-4 positives only; the NEW coverage-4 top-5 neurons (in no earlier map's
      top-30) ranked among ALL 12,288 neurons of their layer: median percentile >= 90
  also reported: of the top 1% of all 442k neurons by score, how many are known concept neurons (enrichment), and the
  most predictive features.
Output (D:/already-clean-data/concept_search/): features.npz (all neurons, every feature), scores.npz, summary.txt,
summary.json.  Usage: python src/concept_search_feasibility.py
"""
import glob
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch

from config import DEFAULT_MODEL
from neurons import DEVICE, load, mlps

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("AC_DATA_DIR", "D:/already-clean-data")) / "concept_search"
CTX, NEG_PER_POS, TOPW = 1024, 5, 50


def maps():
    """top-5 and top-30 neurons per concept, with the set (directory) it came from; first occurrence wins."""
    out = {}
    for f in sorted(glob.glob(str(ROOT / "results/concepts_*/rank_save_*.json"))):
        if "rank_save_test" in f or f.endswith("_exact.json"):
            continue
        for c, v in json.loads(Path(f).read_text(encoding="utf-8")).items():
            if "top" in v and c not in out:
                ns = [e["neuron"] for e in v["top"][:30]]
                out[c] = {"set": Path(f).parent.name, "top5": ns[:5], "top30": ns}
    return out


def uid(u):
    L, n = u[1:].split(":"); return int(L) * 12288 + int(n)


def token_props(tok, V):
    word, cap, latin = np.zeros(V, bool), np.zeros(V, bool), np.zeros(V, bool)
    for i in range(V):
        s = tok.decode([i])
        st = s.lstrip()
        word[i] = s.startswith(" ") and len(st) >= 2 and st[:2].isalpha()
        cap[i] = word[i] and st[0].isupper()
        latin[i] = len(st) >= 1 and all(("a" <= ch.lower() <= "z") for ch in st)
    return word, cap, latin


def atlas_features(A, toks, starts, word, cap):
    feats, names = [], []
    for side in ("top", "bot"):
        v = A[f"{side}_v"].float().abs().numpy().reshape(-1, 40)            # [442k, 40], strongest first
        c = A[f"{side}_c"].numpy().reshape(-1, 40).astype(np.int64)
        g = A[f"{side}_g"].numpy().reshape(-1, 40)
        doc, pos = c // CTX, c % CTX
        t = toks[np.minimum(starts[doc] + pos, len(toks) - 1)]
        peak = v[:, 0]; mean = v.mean(1) + 1e-6
        ts = np.sort(t, 1); nt = 1 + (np.diff(ts, axis=1) != 0).sum(1)
        ds = np.sort(doc, 1); nd = 1 + (np.diff(ds, axis=1) != 0).sum(1)
        feats += [np.log1p(peak), peak / mean, v[:, -1] / (peak + 1e-6), nt / 40, nd / 40, word[t].mean(1), cap[t].mean(1), g.mean(1)]
        names += [f"{side}_{x}" for x in ("log_peak", "spikiness", "decay", "distinct_tokens", "distinct_docs", "word_start", "cap_word", "gate_open")]
    return feats, names


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    M = maps()
    model, tok = load(DEFAULT_MODEL)
    V = model.lm_head.weight.shape[0]
    word, cap, latin = token_props(tok, V)
    A = torch.load(ROOT / f"data/atlas_{DEFAULT_MODEL.split('/')[-1]}.pt", map_location="cpu", weights_only=False)
    S = torch.load(ROOT / f"data/stats_{DEFAULT_MODEL.split('/')[-1]}.pt", map_location="cpu", weights_only=False)
    d = np.load(ROOT / f"data/atlas_{DEFAULT_MODEL.split('/')[-1]}_docs.npz")
    toks, lens = d["tokens"].astype(np.int64), d["lengths"]; starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    feats, names = atlas_features(A, toks, starts, word, cap)
    feats += [np.log1p(S["max"].numpy().reshape(-1).clip(min=0)), np.log1p((-S["min"].numpy().reshape(-1)).clip(min=0))]
    names += ["stats_log_max", "stats_log_neg_min"]
    # weights
    WU = model.lm_head.weight.detach(); g = model.model.norm.weight.detach().float()
    wd, wc, wl = (torch.tensor(x, device=DEVICE) for x in (word, cap, latin))
    wf = {k: [] for k in ("pos_maxz", "pos_word", "pos_cap", "pos_latin", "neg_maxz", "neg_word", "neg_cap", "neg_latin", "down_norm", "gate_norm", "up_norm", "gate_up_cos")}
    for L, m in enumerate(mlps(model)):
        D = m.down_proj.weight.detach(); G = m.gate_proj.weight.detach().float(); U = m.up_proj.weight.detach().float()
        wf["down_norm"].append(D.float().norm(dim=0).cpu()); wf["gate_norm"].append(G.norm(dim=1).cpu()); wf["up_norm"].append(U.norm(dim=1).cpu())
        wf["gate_up_cos"].append(torch.nn.functional.cosine_similarity(G, U, dim=1).cpu())
        for i in range(0, D.shape[1], 1024):
            cols = (D[:, i:i + 1024].float().T * g).to(WU.dtype)            # [n, 4096]
            lg = (cols @ WU.T).float()                                     # [n, V]
            mu, sd = lg.mean(1, keepdim=True), lg.std(1, keepdim=True) + 1e-6
            for sgn, key in ((1, "pos"), (-1, "neg")):
                x = sgn * lg
                top = x.topk(TOPW, dim=1).indices
                wf[f"{key}_maxz"].append(((x.max(1).values - sgn * mu[:, 0]) / sd[:, 0]).cpu())
                wf[f"{key}_word"].append(wd[top].float().mean(1).cpu()); wf[f"{key}_cap"].append(wc[top].float().mean(1).cpu())
                wf[f"{key}_latin"].append(wl[top].float().mean(1).cpu())
        print(f"weights: layer {L} done", flush=True)
    for k, v in wf.items():
        feats.append(torch.cat(v).numpy()); names.append(f"w_{k}")
    X = np.stack(feats, 1).astype(np.float32)                              # [442368, F]
    layer = np.repeat(np.arange(36), 12288)
    np.savez_compressed(OUT / "features.npz", X=X, names=np.array(names), layer=layer)
    # labels
    pre = {c: v for c, v in M.items() if v["set"] != "concepts_fame5"}
    pos_old = {uid(u) for v in pre.values() for u in v["top5"]}
    top30_old = {uid(u) for v in pre.values() for u in v["top30"]}
    top30_all = {uid(u) for v in M.values() for u in v["top30"]}
    pos_new = {uid(u) for c, v in M.items() if v["set"] == "concepts_fame5" for u in v["top5"]} - top30_old
    rng = random.Random(31)

    def negatives(pos):
        neg = []
        for p in sorted(pos):
            L = p // 12288
            while True:
                cands = [L * 12288 + rng.randrange(12288) for _ in range(NEG_PER_POS * 3)]
                cands = [x for x in cands if x not in top30_all][:NEG_PER_POS]
                if len(cands) == NEG_PER_POS:
                    neg += cands; break
        return neg
    neg_old = negatives(pos_old)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd

    def fit(idx, y):
        Xt = torch.tensor(Z[idx], device=DEVICE); yt = torch.tensor(y, device=DEVICE, dtype=torch.float32)
        w = torch.zeros(Z.shape[1], device=DEVICE, requires_grad=True); b = torch.zeros(1, device=DEVICE, requires_grad=True)
        opt = torch.optim.LBFGS([w, b], max_iter=500, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(Xt @ w + b, yt) + 1e-3 * (w ** 2).sum()
            loss.backward(); return loss
        with torch.enable_grad():
            opt.step(closure)
        return w.detach().cpu().numpy(), float(b.detach())

    def auc(s, y):
        o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
        npos = y.sum(); nneg = len(y) - npos
        return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)
    idx = np.array(sorted(pos_old) + neg_old); y = np.array([1] * len(pos_old) + [0] * len(neg_old))
    perm = np.random.RandomState(31).permutation(len(idx)); folds = np.array_split(perm, 5); aucs = []
    for k in range(5):
        te = folds[k]; tr = np.concatenate([folds[j] for j in range(5) if j != k])
        w, b = fit(idx[tr], y[tr]); aucs.append(auc(Z[idx[te]] @ w + b, y[te]))
    w, b = fit(idx, y)
    score = Z @ w + b
    pct = np.empty_like(score)
    for L in range(36):
        sl = slice(L * 12288, (L + 1) * 12288); o = np.argsort(score[sl]); r = np.empty(12288); r[o] = np.arange(12288)
        pct[sl] = 100 * r / 12287
    new_pct = np.array([pct[p] for p in sorted(pos_new)])
    top1 = np.argsort(-score)[: len(score) // 100]
    known = pos_old | pos_new
    hit = len(set(top1.tolist()) & known); base = len(known) / len(score)
    # the same out-of-sample test with layer-matched RANDOM neurons in place of the new positives (sanity: ~50)
    rnd = [L * 12288 + rng.randrange(12288) for L in (p // 12288 for p in sorted(pos_new))]
    rnd_pct = np.array([pct[x] for x in rnd])
    order = np.argsort(-np.abs(w))
    np.savez_compressed(OUT / "scores.npz", score=score, pct=pct, w=w, b=b, names=np.array(names))
    res = {"n_features": len(names), "pos_old": len(pos_old), "neg_old": len(neg_old), "pos_new": len(pos_new),
           "cv_auc": [float(a) for a in aucs], "new_median_pct": float(np.median(new_pct)), "new_share_top10pct": float((new_pct >= 90).mean()),
           "random_median_pct": float(np.median(rnd_pct)), "top1pct_known": hit, "top1pct_size": len(top1), "top1pct_expected": base * len(top1),
           "weights": {names[i]: float(w[i]) for i in order}}
    crit = np.mean(aucs) >= 0.85 and np.median(new_pct) >= 90
    L_ = [f"Concept-search feasibility: {len(names)} cheap features for all 442,368 neurons; positives (pre-coverage-4 top-5) {len(pos_old)}, "
          f"layer-matched negatives {len(neg_old)}; new coverage-4 top-5 neurons (in no earlier top-30) {len(pos_new)}",
          f"(1) 5-fold CV AUC: {np.mean(aucs):.3f} (folds {', '.join(f'{a:.3f}' for a in aucs)}) - needs >= 0.85",
          f"(2) out of sample, new coverage-4 neurons ranked among all 12,288 of their layer: median percentile {np.median(new_pct):.1f} "
          f"(needs >= 90); {100 * (new_pct >= 90).mean():.0f}% in their layer's top 10%; layer-matched random neurons: median {np.median(rnd_pct):.1f}",
          f"enrichment: top 1% of all neurons by score ({len(top1)}) holds {hit} known concept neurons vs {base * len(top1):.0f} expected at random "
          f"(x{hit / max(1e-9, base * len(top1)):.1f})",
          f"CRITERIA {'MET' if crit else 'NOT MET'}",
          "most predictive features (standardised weight; + = more concept-like):"]
    L_ += [f"  {names[i]:<24} {w[i]:+.2f}" for i in order[:12]]
    txt = "\n".join(L_)
    (OUT / "summary.txt").write_text(txt + "\n", encoding="utf-8"); (OUT / "summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
