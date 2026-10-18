"""Stage D, second reading (added after the pre-set Luna-free clustering failed its own calibration: the silhouette rule
called 40/52 known two-neuron mixtures 'one meaning'). Contexts grouped by LUNA's meanings, jobs measured in QWEN:
indirect-effect cosine and next-token Jaccard, within vs between Luna groups (misc excluded); also split by Luna's
relatedness rating of each group pair. Calibration as before: spliced (true sources) vs random split halves.
Usage: python src/meaning_jobs_luna_groups.py -> results/meaning_jobs_luna_groups.txt
"""
import json
import statistics as stx
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    J = json.loads((ROOT / "results/meaning_jobs.json").read_text(encoding="utf-8")); A = np.load(ROOT / "results/meaning_jobs.npz")
    MJ = json.loads((ROOT / "results/meaning_judge.json").read_text(encoding="utf-8"))["jobs"]
    by_rating = defaultdict(lambda: defaultdict(list)); gaps = defaultdict(list); half = []
    for key, j in J.items():
        E = A[f"{j['array_index']}|indirect"].astype(np.float32); En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9); S = En @ En.T
        T = [set(t for t, _ in c["up"]) for c in j["contexts"]]
        jac = lambda a, b: len(T[a] & T[b]) / max(1, len(T[a] | T[b]))
        if j["kind"] == "spliced":
            lab = MJ[key]["source"]; rating = None
        else:
            lab = [None] * len(T); groups = MJ[key]["result"]["groups"]
            for gi, g in enumerate(groups):
                for mm in g["members"]:
                    lab[int(mm) - 1] = gi
            rating = {(int(p["a"]), int(p["b"])): int(p["rating"]) for p in MJ[key]["result"]["relatedness"]}
        w, b = [], []
        for x in range(len(T)):
            for y in range(x + 1, len(T)):
                if lab[x] is None or lab[y] is None:
                    continue
                (w if lab[x] == lab[y] else b).append((S[x, y], jac(x, y)))
                if j["kind"] != "spliced" and lab[x] != lab[y]:
                    r = rating.get((lab[x], lab[y]), rating.get((lab[y], lab[x])))
                    if r is not None:
                        by_rating[j["kind"]][r].append(S[x, y])
        if w and b:
            gaps[j["kind"]].append((np.mean([u for u, _ in w]) - np.mean([u for u, _ in b]), np.mean([v for _, v in w]) - np.mean([v for _, v in b])))
        h = [i % 2 for i in range(len(T))]
        hw = [S[x, y] for x in range(len(T)) for y in range(x + 1, len(T)) if h[x] == h[y]]
        hb = [S[x, y] for x in range(len(T)) for y in range(x + 1, len(T)) if h[x] != h[y]]
        half.append(np.mean(hw) - np.mean(hb))
    md = stx.median
    sp = md([g[0] for g in gaps["spliced"]]); sh = md(half)
    L = [f"Jobs measured in Qwen, contexts grouped by Luna's meanings (indirect-effect cosine gap: within - between).",
         f"calibration: same job (random halves) {sh:+.3f}; different neurons (spliced sources) {sp:+.3f}"]
    for kind in ("concept", "control"):
        g = gaps[kind]
        if g:
            L.append(f"{kind}: n={len(g)} units with >= 2 Luna groups; gap {md([x[0] for x in g]):+.3f} (position {(md([x[0] for x in g]) - sh) / (sp - sh):.2f} "
                     f"between same-job 0 and different-neurons 1); next-token Jaccard gap {md([x[1] for x in g]):+.3f}")
            L.append("  mean indirect-effect cosine between contexts of different Luna groups, by Luna's rating: " + ", ".join(
                f"{r}: {np.mean(v):+.3f} (n={len(v)})" for r, v in sorted(by_rating[kind].items())))
    txt = "\n".join(L)
    (ROOT / "results/meaning_jobs_luna_groups.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
