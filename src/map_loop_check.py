"""Validate map_loop.py (no model, no API).

1. Reproduction (concepts_fame2, tag 'repro', cached specs): status, recall count, answer form, k_mean / k_median /
   k_flip and the drop curves must equal wave_free.json; the top-30 neurons (in order) must equal rank_save_fame2.json.
2. Maps vs cross-activation: for those concepts, last_act (answer-position mean) must match cross_activation_fame2.json
   'ans', and pmax / pmin (by each neuron's sign) its 'peak', within bf16 noise (|diff| <= 0.02 + 2% of |value|).
3. Fresh run (concepts_fame4, tag 'fresh'): GPU idle share, minutes per concept by status, median per-step times,
   Luna cost per concept - against the old loop (~1.28 min per recalled concept + ~0.37 min ranking; ~0.64 min not recalled).
Usage: python src/map_loop_check.py  -> results/map_loop_check.txt
"""
import json
import statistics as stx
from collections import defaultdict
from pathlib import Path

import numpy as np

import concept_pipeline as cp

ROOT = Path(__file__).resolve().parents[1]


def jl(p):
    return json.loads((ROOT / p).read_text(encoding="utf-8"))


def main():
    L = []
    new = jl("results/maploop_repro/map_loop_repro.json"); wf = jl("results/concepts_fame2/wave_free.json")
    rk = jl("results/concepts_fame2/rank_save_fame2.json"); nrk = jl("results/maploop_repro/rank_save_repro.json")
    bad = []
    for c, r in new.items():
        o = wf[c]
        for k in ("status", "recall", "answer_form", "k_mean", "k_median", "k_flip", "drops", "random_drops"):
            if k in o and r.get(k) != o.get(k):
                bad.append(f"{c}: {k} {o.get(k)} -> {r.get(k)}")
        if c in rk and c in nrk and [e["neuron"] for e in rk[c]["top"]] != [e["neuron"] for e in nrk[c]["top"]]:
            bad.append(f"{c}: top-30 differs")
    L.append(f"1. REPRODUCTION ({len(new)} concepts): {'PASS - identical' if not bad else 'FAIL'}")
    L += [f"   {b}" for b in bad]
    xa = jl("results/cross_activation_fame2.json"); ui = {u: i for i, u in enumerate(xa["units"])}
    worst = 0.0; n = 0; over = 0
    for c in new:
        if c not in xa["concepts"] or new[c]["status"] != "ok":
            continue
        z = np.load(ROOT / "results/maploop_repro/rank_save_arrays" / f"{cp.slug(c)}.npz")
        la, pmax, pmin = z["last_act"].astype(np.float32), z["pmax"].astype(np.float32), z["pmin"].astype(np.float32)
        for u, i in ui.items():
            l, k = (int(x) for x in u[1:].split(":"))
            for mine, ref in ((la[l, k], xa["concepts"][c]["ans"][i]),):
                d = abs(mine - ref); n += 1; worst = max(worst, d); over += d > 0.02 + 0.02 * abs(ref)
            pk = xa["concepts"][c]["peak"][i]
            cand = min(abs(pmax[l, k] - pk), abs(pmin[l, k] - pk)); n += 1; worst = max(worst, cand); over += cand > 0.02 + 0.02 * abs(pk)
    L.append(f"\n2. MAPS vs CROSS-ACTIVATION: {n} values compared; outside bf16 tolerance: {over}; largest |diff| {worst:.4f} "
             f"-> {'PASS' if over == 0 else 'CHECK'}")
    meta = jl("results/concepts_fame4/map_loop_fresh_meta.json"); fr = jl("results/concepts_fame4/map_loop_fresh.json")
    L.append(f"\n3. FRESH RUN: {meta['concepts']} concepts in {meta['elapsed_s'] / 60:.1f} min; GPU idle waiting on Luna "
             f"{meta['gpu_idle_s'] / 60:.1f} min ({meta['gpu_idle_s'] / max(meta['elapsed_s'], 1):.0%}); Luna {meta['luna_calls']} calls, ${meta['luna_usd']:.3f}")
    by = defaultdict(list); steps = defaultdict(list)
    for c, r in fr.items():
        g = sum(v for k, v in r["timing_s"].items() if k.startswith("G")); by[r["status"]].append(g)
        for k, v in r["timing_s"].items():
            steps[k].append(v)
    for s, v in by.items():
        L.append(f"   {s:<14} n={len(v):<3} GPU minutes per concept: median {stx.median(v) / 60:.2f}")
    L.append("   median seconds per step: " + ", ".join(f"{k} {stx.median(v):.1f}" for k, v in sorted(steps.items())))
    L.append(f"   wall-clock minutes per concept overall: {meta['elapsed_s'] / 60 / max(meta['concepts'], 1):.2f} "
             f"(old loop: ~1.65 per recalled incl. ranking, ~0.64 per not recalled)")
    usd = [sum(x["usd"] for x in r.get("luna", {}).values()) for r in fr.values()]
    L.append(f"   Luna per concept: median ${stx.median(usd):.4f}")
    txt = "\n".join(L)
    (ROOT / "results/map_loop_check.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
