"""Artefact check for concept_search_feasibility.py: is 'concept-like' just 'big neuron'? (no model, no API; seconds)

The strongest feature was the gate vector's norm, and concept neurons were found by attribution, which favours neurons
that fire strongly. Three re-runs of both tests on the saved features:
  A  the same, for reference
  B  SIZE-MATCHED negatives: for each positive, 5 neurons of the same layer, in no concept's top-30, with the closest
     corpus peak (max of stats max and -min) - so size can't separate the classes
  C  size features removed (gate / up / down norms, peak, corpus max and min) and size-matched negatives
Output: D:/already-clean-data/concept_search/size_check.txt.  Usage: python src/concept_search_size_check.py
"""
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from concept_search_feasibility import maps, uid

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("AC_DATA_DIR", "D:/already-clean-data")) / "concept_search"
SIZE = {"w_gate_norm", "w_up_norm", "w_down_norm", "top_log_peak", "bot_log_peak", "stats_log_max", "stats_log_neg_min"}


def main():
    F = np.load(OUT / "features.npz"); X, names = F["X"], list(F["names"])
    M = maps()
    pre = {c: v for c, v in M.items() if v["set"] != "concepts_fame5"}
    pos_old = sorted({uid(u) for v in pre.values() for u in v["top5"]})
    top30_old = {uid(u) for v in pre.values() for u in v["top30"]}
    top30_all = {uid(u) for v in M.values() for u in v["top30"]}
    pos_new = sorted({uid(u) for c, v in M.items() if v["set"] == "concepts_fame5" for u in v["top5"]} - top30_old)
    size = np.maximum(X[:, names.index("stats_log_max")], X[:, names.index("stats_log_neg_min")])
    rng = random.Random(41)

    def neg_random(pos):
        out = []
        for p in pos:
            L = p // 12288; got = []
            while len(got) < 5:
                x = L * 12288 + rng.randrange(12288)
                if x not in top30_all and x not in got:
                    got.append(x)
            out += got
        return out

    by_layer = {}
    for L in range(36):
        ids = np.arange(L * 12288, (L + 1) * 12288); ids = np.array([i for i in ids if i not in top30_all])
        by_layer[L] = (ids, size[ids])

    def neg_matched(pos):
        out, used = [], set()
        for p in pos:
            ids, sz = by_layer[p // 12288]
            o = np.argsort(np.abs(sz - size[p]))
            k = 0; got = 0
            while got < 5:
                x = int(ids[o[k]]); k += 1
                if x not in used:
                    used.add(x); out.append(x); got += 1
        return out

    def run(cols, neg):
        Z = X[:, cols]; Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-6)
        idx = np.array(pos_old + neg); y = np.array([1] * len(pos_old) + [0] * len(neg))

        def fit(ii, yy):
            Xt = torch.tensor(Z[ii]); yt = torch.tensor(yy, dtype=torch.float32)
            w = torch.zeros(Z.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
            opt = torch.optim.LBFGS([w, b], max_iter=500, line_search_fn="strong_wolfe")

            def cl():
                opt.zero_grad(); l = torch.nn.functional.binary_cross_entropy_with_logits(Xt @ w + b, yt) + 1e-3 * (w ** 2).sum(); l.backward(); return l
            opt.step(cl); return w.detach().numpy(), float(b.detach())

        def auc(s, yy):
            o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1); n1 = yy.sum(); n0 = len(yy) - n1
            return (r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
        perm = np.random.RandomState(41).permutation(len(idx)); folds = np.array_split(perm, 5); aucs = []
        for k in range(5):
            te = folds[k]; tr = np.concatenate([folds[j] for j in range(5) if j != k]); w, b = fit(idx[tr], y[tr]); aucs.append(auc(Z[idx[te]] @ w + b, y[te]))
        w, b = fit(idx, y); s = Z @ w + b
        pct = np.empty_like(s)
        for L in range(36):
            sl = slice(L * 12288, (L + 1) * 12288); o = np.argsort(s[sl]); r = np.empty(12288); r[o] = np.arange(12288); pct[sl] = 100 * r / 12287
        newp = pct[pos_new]
        # size-matched out-of-sample reference: for each new positive, its layer's neurons of similar size
        mref = neg_matched(pos_new); mrefp = pct[mref]
        order = np.argsort(-np.abs(w))
        return np.mean(aucs), np.median(newp), (newp >= 90).mean(), np.median(mrefp), [(cols_names[i], w[i]) for i in order[:6]] if False else [(names[cols[i]], w[i]) for i in order[:6]]
    all_cols = list(range(len(names))); nosize = [i for i, n in enumerate(names) if n not in SIZE]
    L = ["Is 'concept-like' just 'big neuron'? (5-fold CV AUC on pre-coverage-4 positives; out-of-sample percentile of the 414 new coverage-4 "
         "neurons among all neurons of their layer; reference = size-matched non-concept neurons of the same layers)"]
    for tag, cols, neg in (("A all features, random negatives", all_cols, neg_random(pos_old)),
                           ("B all features, SIZE-MATCHED negatives", all_cols, neg_matched(pos_old)),
                           ("C no size features, size-matched negatives", nosize, neg_matched(pos_old))):
        a, m, t10, ref, top = run(cols, neg)
        L.append(f"{tag}: AUC {a:.3f}; new neurons median percentile {m:.1f} ({100 * t10:.0f}% in top 10%); size-matched reference median {ref:.1f}")
        L.append("    top features: " + ", ".join(f"{n} {w:+.2f}" for n, w in top))
    txt = "\n".join(L)
    (OUT / "size_check.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
