"""ISSUES #35 end-to-end check: does exact log-odds change any ranking result? (no model, no API)

Compares rank_save_<tag>.json (old clamped log-odds, wave_free.py's logit) with rank_save_<tag>_exact.json (exact,
pipe.batch_logodds) for every concept: k_mean, k_flip, the top-10 neurons, the baseline mean log-odds - and, where a
published original exists (wave_free.json of waves 3/4 and CR-3), whether the exact run still reproduces it.
Usage: python src/cap_fix_compare.py  -> results/cap_fix_compare.txt
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETS = [("concepts_v3", "cr2"), ("concepts_w2", "w2"), ("concepts_w3", "w3"), ("concepts_w4", "w4"), ("concepts_cr3", "cr3")]


def main():
    L = []; tot = same_k = same_top = 0; changed = []; orig_mis = []
    for d, t in SETS:
        a_p, b_p = ROOT / f"results/{d}/rank_save_{t}.json", ROOT / f"results/{d}/rank_save_{t}_exact.json"
        if not b_p.exists():
            L.append(f"{t}: no exact run yet"); continue
        a = json.loads(a_p.read_text(encoding="utf-8")); b = json.loads(b_p.read_text(encoding="utf-8"))
        wf_p = ROOT / f"results/{d}/wave_free.json"; wf = json.loads(wf_p.read_text(encoding="utf-8")) if wf_p.exists() else {}
        for c, x in b.items():
            if "top" not in x or "top" not in a.get(c, {}):
                continue
            y = a[c]; tot += 1
            k_ok = (x["k_mean"], x["k_flip"]) == (y["k_mean"], y["k_flip"]); same_k += k_ok
            t10 = [e["neuron"] for e in x["top"][:10]] == [e["neuron"] for e in y["top"][:10]]; same_top += t10
            if not k_ok or not t10:
                changed.append(f"  {t:<4} {c:<30} k_mean/k_flip {y['k_mean']}/{y['k_flip']} -> {x['k_mean']}/{x['k_flip']} | top-10 "
                               f"{'same' if t10 else 'order/set differs (overlap %d)' % len({e['neuron'] for e in x['top'][:10]} & {e['neuron'] for e in y['top'][:10]})}"
                               f" | baseline {y['baseline_mean_logodds']} -> {x['baseline_mean_logodds']}")
            if c in wf and "k_mean" in wf[c] and (x["k_mean"], x["k_flip"]) != (wf[c]["k_mean"], wf[c]["k_flip"]):
                orig_mis.append(f"  {t:<4} {c:<30} published {wf[c]['k_mean']}/{wf[c]['k_flip']} -> exact {x['k_mean']}/{x['k_flip']}")
    L.insert(0, f"Cap fix (ISSUES #35), {tot} concepts: k_mean and k_flip identical in {same_k}/{tot}; top-10 neurons identical (same order) "
                f"in {same_top}/{tot}. Published wave results (waves 3/4, CR-3) that the exact run no longer reproduces: {len(orig_mis)}\n")
    L += ["concepts where anything changed:"] + (changed or ["  none"]) + ["published values that change:"] + (orig_mis or ["  none"])
    txt = "\n".join(L)
    (ROOT / "results/cap_fix_compare.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
