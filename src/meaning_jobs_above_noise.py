"""Stage D robustness (17 Oct, prompted by the gpu-fan-out session's bf16 noise floor): does the 'one job' reading hold
on contexts whose knockout is clearly above bf16 noise? Same analysis as meaning_jobs_luna_groups.py, keeping only
contexts with KL >= threshold. Thresholds from the noise measurement (8B, L18, batch-shape noise): p90 1.9e-3,
10x median 3.2e-3; plus 1e-2 and 3e-2. A unit needs >= 2 kept contexts in >= 2 groups. (no model, no API)
Usage: python src/meaning_jobs_above_noise.py -> results/meaning_jobs_above_noise.txt
"""
import json
import statistics as stx
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def run(J, A, MJ, thr):
    gaps = defaultdict(list); half = []; by_rating = defaultdict(lambda: defaultdict(list)); kept = defaultdict(lambda: [0, 0])
    for key, j in J.items():
        E = A[f"{j['array_index']}|indirect"].astype(np.float32); En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9); S = En @ En.T
        ok = [c["kl"] >= thr for c in j["contexts"]]
        kept[j["kind"]][0] += sum(ok); kept[j["kind"]][1] += len(ok)
        if j["kind"] == "spliced":
            lab = list(MJ[key]["source"]); rating = {}
        else:
            lab = [None] * len(ok)
            for gi, g in enumerate(MJ[key]["result"]["groups"]):
                for mm in g["members"]:
                    lab[int(mm) - 1] = gi
            rating = {(int(p["a"]), int(p["b"])): int(p["rating"]) for p in MJ[key]["result"]["relatedness"]}
        idx = [i for i in range(len(ok)) if ok[i]]
        lab = [lab[i] if ok[i] else None for i in range(len(ok))]
        cnt = defaultdict(int)
        for i in idx:
            if lab[i] is not None:
                cnt[lab[i]] += 1
        if sum(v >= 2 for v in cnt.values()) < 2:
            continue
        w, b = [], []
        for x in idx:
            for y in idx:
                if y <= x or lab[x] is None or lab[y] is None:
                    continue
                (w if lab[x] == lab[y] else b).append(S[x, y])
                if j["kind"] != "spliced" and lab[x] != lab[y]:
                    r = rating.get((lab[x], lab[y]), rating.get((lab[y], lab[x])))
                    if r is not None:
                        by_rating[j["kind"]][r].append(S[x, y])
        gaps[j["kind"]].append(np.mean(w) - np.mean(b))
        if len(idx) >= 4:
            h = {i: k % 2 for k, i in enumerate(idx)}
            hw = [S[x, y] for x in idx for y in idx if y > x and h[x] == h[y]]
            hb = [S[x, y] for x in idx for y in idx if y > x and h[x] != h[y]]
            half.append(np.mean(hw) - np.mean(hb))
    md = stx.median
    sp = md(gaps["spliced"]) if gaps["spliced"] else float("nan"); sh = md(half)
    L = [f"\nKL >= {thr:g}: contexts kept " + ", ".join(f"{k} {a}/{n} ({a / n:.0%})" for k, (a, n) in kept.items())]
    L.append(f"  calibration: same job (random halves) {sh:+.3f}; different neurons (spliced) {sp:+.3f} (n={len(gaps['spliced'])})")
    for kind in ("concept", "control"):
        g = gaps[kind]
        if g:
            L.append(f"  {kind}: n={len(g)} units; gap {md(g):+.3f} -> position {(md(g) - sh) / (sp - sh):.2f}; between-group cosine by Luna rating: "
                     + ", ".join(f"{r}: {np.mean(v):+.3f} (n={len(v)})" for r, v in sorted(by_rating[kind].items())))
    return L


def main():
    J = json.loads((ROOT / "results/meaning_jobs.json").read_text(encoding="utf-8")); A = np.load(ROOT / "results/meaning_jobs.npz")
    MJ = json.loads((ROOT / "results/meaning_judge.json").read_text(encoding="utf-8"))["jobs"]
    kl = {k: [c["kl"] for j in J.values() if j["kind"] == k for c in j["contexts"]] for k in ("concept", "control", "spliced")}
    L = ["Stage D on above-noise contexts only (indirect-effect cosine gap within - between Luna groups; position 0 = same job, 1 = different neurons)",
         "stage D knockout KL quantiles (p10/p50/p90): " + "; ".join(f"{k} {np.quantile(v, .1):.1e}/{np.median(v):.1e}/{np.quantile(v, .9):.1e}" for k, v in kl.items())]
    for thr in (0.0, 1.9e-3, 3.2e-3, 1e-2, 3e-2):
        L += run(J, A, MJ, thr)
    txt = "\n".join(L)
    (ROOT / "results/meaning_jobs_above_noise.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
