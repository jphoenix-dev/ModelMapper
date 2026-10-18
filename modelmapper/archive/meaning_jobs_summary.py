"""Stage D summary (Luna-free): one job or several? Rules fixed 17 Oct before the stage D data existed. (no model, no API)

Per unit (with saved context vectors from stage C and effects from stage D):
  Qwen's own meaning groups: k-means (k = 2..4, 20 restarts, seed 61) on the unit's 30 whitened context vectors (read
    directions removed); the k with the best silhouette is kept; if that silhouette < 0.10 the unit has ONE meaning.
  Job comparison: for each pair of contexts, cosine between their INDIRECT effects (final-state change minus the
    neuron's own write) and Jaccard overlap of their 20 most-raised next tokens. job_gap = mean within-cluster minus
    mean between-cluster similarity (0 = the neuron does the same thing in every cluster = one job).
  Calibration: spliced units (two neurons' firings mixed; clusters = the true sources) give the 'different jobs' scale;
    split halves of single-meaning units give 'same job'.
  Agreement with Luna: adjusted Rand index between Qwen's clusters and Luna's groups (misc snippets left out).
Usage: python src/meaning_jobs_summary.py  -> results/meaning_jobs_summary.txt
"""
import json
import random
import statistics as stx
from collections import Counter
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SIL_MIN = 0.10


def kmeans(X, k, rng, iters=50):
    best = None
    for _ in range(20):
        C = X[rng.sample(range(len(X)), k)]
        for _ in range(iters):
            lab = np.argmax(X @ C.T, 1)
            newC = np.stack([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
            newC /= np.linalg.norm(newC, axis=1, keepdims=True) + 1e-9
            if np.allclose(newC, C):
                break
            C = newC
        inertia = -(X * C[lab]).sum()
        if best is None or inertia < best[0]:
            best = (inertia, lab)
    return best[1]


def silhouette(X, lab):
    D = 1 - X @ X.T; s = []
    for i in range(len(X)):
        same = lab == lab[i]
        if same.sum() <= 1:
            s.append(0.0); continue
        a = D[i, same].sum() / (same.sum() - 1)
        b = min(D[i, lab == j].mean() for j in set(lab) if j != lab[i])
        s.append((b - a) / max(a, b))
    return float(np.mean(s))


def ari(a, b):
    n = len(a); ct = Counter(zip(a, b)); ra = Counter(a); rb = Counter(b)
    idx = sum(comb(v, 2) for v in ct.values()); ea = sum(comb(v, 2) for v in ra.values()); eb = sum(comb(v, 2) for v in rb.values())
    exp = ea * eb / comb(n, 2) if n > 1 else 0; mx = (ea + eb) / 2
    return (idx - exp) / (mx - exp) if mx != exp else 1.0


def gaps(E, T, lab):
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9); S = En @ En.T
    w, bt, jw, jb = [], [], [], []
    for i in range(len(lab)):
        for j in range(i + 1, len(lab)):
            jac = len(T[i] & T[j]) / max(1, len(T[i] | T[j]))
            (w if lab[i] == lab[j] else bt).append(S[i, j]); (jw if lab[i] == lab[j] else jb).append(jac)
    if not w or not bt:
        return None
    return float(np.mean(w) - np.mean(bt)), float(np.mean(jw) - np.mean(jb))


def main():
    V = np.load(ROOT / "results/meaning_space_vectors.npz"); vkeys = list(V["keys"])
    vec = {k: V[f"v{i}"].astype(np.float32) for i, k in enumerate(vkeys)}
    J = json.loads((ROOT / "results/meaning_jobs.json").read_text(encoding="utf-8")); A = np.load(ROOT / "results/meaning_jobs.npz")
    MJ = json.loads((ROOT / "results/meaning_judge.json").read_text(encoding="utf-8"))["jobs"]
    rng = random.Random(61); res = {}
    for key, j in J.items():
        if key not in vec:
            continue
        X = vec[key]; X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        E = A[f"{j['array_index']}|indirect"].astype(np.float32)
        T = [set(t for t, _ in c["up"]) for c in j["contexts"]]
        if j["kind"] == "spliced":
            lab = np.array(MJ[key]["source"]); k = 2; sil = silhouette(X, lab)
            qlab = None; bestk = None
            for kk in (2, 3, 4):
                l_ = kmeans(X, kk, rng); s_ = silhouette(X, l_)
                if bestk is None or s_ > bestk[0]:
                    bestk = (s_, l_)
            qlab = bestk[1]
            res[key] = {"kind": "spliced", "gap": gaps(E, T, lab), "ari_qwen_vs_sources": ari(list(qlab), list(lab)), "sil_sources": sil}
            continue
        best = None
        for kk in (2, 3, 4):
            l_ = kmeans(X, kk, rng); s_ = silhouette(X, l_)
            if best is None or s_ > best[0]:
                best = (s_, kk, l_)
        one = best[0] < SIL_MIN
        lab = np.zeros(len(X), int) if one else best[2]
        g = None if one else gaps(E, T, lab)
        half = np.array([i % 2 for i in range(len(X))])            # split-half reference for 'same job'
        gh = gaps(E, T, half)
        groups = MJ[key]["result"]["groups"]; luna = {}
        for gi, gr in enumerate(groups):
            for mm in gr["members"]:
                luna[int(mm) - 1] = gi
        keep = [i for i in range(len(X)) if i in luna]
        res[key] = {"kind": j["kind"], "one_meaning": one, "k": 1 if one else best[1], "silhouette": round(best[0], 3), "gap": g, "split_half_gap": gh,
                    "ari_qwen_vs_luna": ari([int(lab[i]) for i in keep], [luna[i] for i in keep]) if len(keep) > 2 else None,
                    "luna_groups": len(groups)}
    md = lambda xs: stx.median(xs) if xs else float("nan")
    L = [f"Stage D (Luna-free): {len(res)} units with context vectors and effects"]
    sp = [r for r in res.values() if r["kind"] == "spliced" and r["gap"]]
    L.append(f"calibration - spliced (true sources): job_gap indirect-effect cosine median {md([r['gap'][0] for r in sp]):+.3f}, next-token Jaccard "
             f"{md([r['gap'][1] for r in sp]):+.3f}; Qwen's clusters recover the sources: ARI median {md([r['ari_qwen_vs_sources'] for r in sp]):.2f}")
    sh = [r["split_half_gap"] for r in res.values() if r["kind"] != "spliced" and r.get("split_half_gap")]
    L.append(f"reference - random split halves (same job by construction): job_gap {md([g[0] for g in sh]):+.3f} / {md([g[1] for g in sh]):+.3f}")
    for kind in ("concept", "control"):
        rs = [r for r in res.values() if r["kind"] == kind]
        if not rs:
            continue
        one = sum(r["one_meaning"] for r in rs); multi = [r for r in rs if r["gap"]]
        L.append(f"\n{kind}: {len(rs)} units; ONE meaning in Qwen's own space (best silhouette < {SIL_MIN}): {one} ({one / len(rs):.0%}); "
                 f"k = {dict(Counter(r['k'] for r in rs))}")
        if multi:
            L.append(f"  multi-meaning units ({len(multi)}): job_gap indirect-effect cosine median {md([r['gap'][0] for r in multi]):+.3f}, next-token "
                     f"Jaccard {md([r['gap'][1] for r in multi]):+.3f}")
            sp_g = md([r['gap'][0] for r in sp]) if sp else float('nan'); sh_g = md([g[0] for g in sh])
            L.append(f"  -> position between 'same job' ({sh_g:+.3f}) and 'different neurons' ({sp_g:+.3f}): "
                     f"{(md([r['gap'][0] for r in multi]) - sh_g) / (sp_g - sh_g) if sp_g != sh_g else float('nan'):.2f}")
        L.append(f"  agreement with Luna's groups: ARI median {md([r['ari_qwen_vs_luna'] for r in rs if r['ari_qwen_vs_luna'] is not None]):.2f} "
                 f"(0 = chance, 1 = identical)")
    txt = "\n".join(L)
    (ROOT / "results/meaning_jobs_summary.txt").write_text(txt + "\n", encoding="utf-8")
    (ROOT / "results/meaning_jobs_summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
