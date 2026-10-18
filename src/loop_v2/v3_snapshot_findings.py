"""Findings from the loop v3 enrichment snapshot (CPU; no model). Recomputes run-1 statistics with E0's missed candidates
added, in log-odds AND in probability terms (LOOP_V3 method note, 19 Oct).

For each snapshot concept: run 1's 200 candidates (single knockout, per-prompt log-odds drops) + E0's log-odds-only
candidates (same procedure). Base P(answer) per prompt from E0's record.
  S1 recomputed  share of each concept's top-5 neurons in layers 28-35 (and < 20): run 1 alone vs run 1 + E0, ranked by
                 mean log-odds drop and by mean probability drop
  E2             format results (activation and short-format knockout) - as in RESULTS
  E3             layer-matched null floor
Output: D:/already-clean-data/loop_v3/enrich_run1/snapshot_findings.txt.  Usage: python src/loop_v2/v3_snapshot_findings.py
"""
import glob
import json
import math
import statistics as stx
from pathlib import Path

D = Path("D:/already-clean-data")
SNAP = sorted((D / "loop_v3/enrich_run1").glob("records_snapshot_*.jsonl"))[-1]


def pdrop(p, d):
    p = min(max(p, 1e-12), 1 - 1e-12); lo = math.log(p / (1 - p))
    return p - 1 / (1 + math.exp(-(lo - d)))


def main():
    R = {}
    for l in open(SNAP, encoding="utf-8"):
        r = json.loads(l); R[r["concept"]] = r
    run1 = {}
    for f in sorted(glob.glob(str(D / "loop_v2/run1/results/**/records.jsonl"), recursive=True)):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            if r["concept"] in R and (r.get("row") or {}).get("status") == "ok":
                run1[r["concept"]] = r["rank"]
    lay = lambda u: int(u.split(":")[0][1:])
    acc = {k: [] for k in ("lo_run1", "lo_both", "p_run1", "p_both")}
    early = {k: [] for k in acc}
    n = 0
    for c, r in R.items():
        e0 = r.get("E0"); rk = run1.get(c)
        if not e0 or not rk:
            continue
        P = e0["p"]; ppd = rk["per_prompt_drop_all200"]
        cand = {}
        for x in rk["candidates_all200"]:
            pp = ppd.get(x["neuron"])
            if pp:
                cand[x["neuron"]] = (stx.mean(pp), stx.mean(pdrop(p, d) for p, d in zip(P, pp)), "run1")
        for x in e0["new"]:
            cand[x["neuron"]] = (x["mean_drop"], stx.mean(pdrop(p, d) for p, d in zip(P, x["per_prompt"])), "e0")
        n += 1
        for key, idx, pool in (("lo_run1", 0, "run1"), ("lo_both", 0, None), ("p_run1", 1, "run1"), ("p_both", 1, None)):
            us = [u for u, v in cand.items() if pool is None or v[2] == pool]
            top = sorted(us, key=lambda u: -cand[u][idx])[:5]
            acc[key].append(sum(28 <= lay(u) <= 35 for u in top) / 5); early[key].append(sum(lay(u) < 20 for u in top) / 5)
    L = [f"Loop v3 snapshot findings ({SNAP.name}; {n} concepts with run-1 maps and E0)"]
    L.append("S1 recomputed - share of each concept's top-5 neurons by layer band (mean over concepts):")
    for key, label in (("lo_run1", "log-odds, run 1's candidates only (as published)"), ("lo_both", "log-odds, run 1 + E0 missed candidates"),
                       ("p_run1", "probability drop, run 1's candidates only"), ("p_both", "probability drop, run 1 + E0")):
        L.append(f"  {label:<52} layers 28-35: {stx.mean(acc[key]):.0%} | layers < 20: {stx.mean(early[key]):.0%}")
    # E0 value, in log-odds and probability terms (entrants = E0 neurons entering the top 30)
    ent_p, n_ch, n_big, n_e0 = [], 0, 0, 0
    for c, r in R.items():
        e0 = r.get("E0")
        if not e0:
            continue
        n_e0 += 1; n_ch += bool(e0["entering_top30"])
        by = {x["neuron"]: x for x in e0["new"]}
        ps = [100 * stx.mean(pdrop(p, d) for p, d in zip(e0["p"], by[u]["per_prompt"])) for u in e0["entering_top30"] if u in by]
        ent_p += ps; n_big += any(v >= 5 for v in ps)
    L.append(f"E0: concepts whose top 30 changes {n_ch}/{n_e0} ({n_ch / n_e0:.0%}); entrants {len(ent_p)}; median entrant moves "
             f"P(answer) {stx.median(ent_p):.2f} pts; entrants >= 5 pts {sum(v >= 5 for v in ent_p) / len(ent_p):.0%}; "
             f"concepts with >= 1 such entrant {n_big}/{n_e0} ({n_big / n_e0:.0%})")
    # E2 format-bound: short-format (F2) knockout < half of the original (F1) where F1 > 0.1; activation kept (F2 >= half F1 act)
    fb, ak = [], []
    for r in R.values():
        x = r.get("E2")
        if not x:
            continue
        for u, k in x["knockouts_top30"].items():
            if k.get("F1") and k.get("F2") and stx.mean(k["F1"]) > 0.1:
                d1, d2 = stx.mean(k["F1"]), stx.mean(k["F2"]); fb.append(d2 < 0.5 * d1)
                a1, a2 = x["acts"].get("F1", {}).get(u), x["acts"].get("F2", {}).get(u)
                if fb[-1] and a1 and a2 and abs(stx.mean(a1)) > 0:
                    ak.append(abs(stx.mean(a2)) >= 0.5 * abs(stx.mean(a1)))
    L.append(f"E2: top-30 neurons whose short-format knockout drops by more than half: {sum(fb)}/{len(fb)} ({sum(fb) / len(fb):.0%}); "
             f"of those, activation stays (>= half): {sum(ak)}/{len(ak)} ({sum(ak) / max(1, len(ak)):.0%})")
    nul = sorted(abs(stx.mean(v)) for r in R.values() if r.get("E3") for v in r["E3"]["nulls"].values())
    L.append(f"E3: {len(nul)} layer-matched nulls; |mean drop| median {stx.median(nul):.3f}, p99 {nul[int(0.99 * (len(nul) - 1))]:.3f}")
    txt = "\n".join(L)
    (SNAP.parent / "snapshot_findings.txt").write_text(txt + "\n", encoding="utf-8"); print(txt)


if __name__ == "__main__":
    main()
