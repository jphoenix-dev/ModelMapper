"""Indirect path summary (no model, no API). Reads results/indirect_path.jsonl (see indirect_path.py).
Rows per prompt: 0 as is; 1-5 the concept's top-5 neuron i zeroed (every position); 6-10 its activation-matched null.
  total_i = score0 - score_{1+i}; null_i = score0 - score_{6+i}; direct_i = its write at the answer position read into
  the answer score; indirect_i = total_i - direct_i. (score = answer logit - mean logit at the answer position)
  composition: for an earlier unit i and a later unit j of the same concept, relative drop of j's activation at the
  answer position when i is zeroed = sign_j * (act_j(row 0) - act_j(row 1+i)) / |act_j(row 0)|; null: the same with
  row 6+i. Averaged over prompts per (concept, i, j) pair.
Predictions (stated in indirect_path.py before running): (a) 'say' neurons mostly direct (median direct/total >= 0.7)
and more direct than attribute neurons; (b) attribute -> later 'say' relative drop >= 2x null, sign test p < 0.05.
Usage: python src/indirect_path_summary.py -> results/indirect_path_summary.txt
"""
import json
import math
import statistics as stx
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sign_p(pos, neg):
    n = pos + neg
    return 1.0 if not n else min(1.0, 2 * sum(math.comb(n, i) for i in range(max(pos, neg), n + 1)) / 2 ** n)


def lay(u):
    return int(u.split(":")[0][1:])


def main():
    R = [json.loads(l) for l in (ROOT / "results/indirect_path.jsonl").read_text(encoding="utf-8").splitlines()]
    per = defaultdict(list)          # type -> list of (total, direct, null) averaged over prompts per (concept, unit)
    comp = defaultdict(list)         # (type_i, type_j) -> list of (rel drop, null rel drop) per pair
    for r in R:
        U, TY, P = r["units"], r["types"], r["prompts"]
        for i, u in enumerate(U):
            tot = stx.mean(p["score"][0] - p["score"][1 + i] for p in P)
            nul = stx.mean(p["score"][0] - p["score"][6 + i] for p in P)
            dr = stx.mean(p["direct"][i] for p in P)
            lo = stx.mean(p["logodds"][0] - p["logodds"][1 + i] for p in P)
            per[TY[i]].append((tot, dr, nul, lo))
        for i, ui in enumerate(U):
            for j, uj in enumerate(U):
                if lay(ui) >= lay(uj):
                    continue
                rel, reln = [], []
                for p in P:
                    a0 = p["unit_acts"][0][j]
                    if abs(a0) < 1e-3:
                        continue
                    s = 1.0 if a0 > 0 else -1.0
                    rel.append(s * (a0 - p["unit_acts"][1 + i][j]) / abs(a0)); reln.append(s * (a0 - p["unit_acts"][6 + i][j]) / abs(a0))
                if rel:
                    comp[(TY[i], TY[j])].append((stx.mean(rel), stx.mean(reln)))
    md = stx.median
    L = [f"Indirect path: {len(R)} concepts, {sum(len(r['prompts']) for r in R)} prompts; per (concept, neuron), averaged over prompts",
         "score = answer logit - mean logit; total = drop when the neuron is zeroed; direct = its own write read into the score;"
         " indirect = total - direct; null = activation-matched same-layer neuron zeroed",
         f"\n{'type':<12}{'n':>6}{'total':>9}{'direct':>9}{'indirect':>10}{'direct/total*':>15}{'null total':>12}{'log-odds drop':>15}"]
    for t in ("say", "attribute", "late-other", "early"):
        xs = per.get(t, [])
        if not xs:
            continue
        ratio = [d / tt for tt, d, _, _ in xs if tt > 0.5]
        L.append(f"{t:<12}{len(xs):>6}{md(x[0] for x in xs):>9.2f}{md(x[1] for x in xs):>9.2f}{md(x[0] - x[1] for x in xs):>10.2f}"
                 f"{(md(ratio) if ratio else float('nan')):>15.2f}{md(x[2] for x in xs):>12.3f}{md(x[3] for x in xs):>15.2f}")
    L.append("*median over neurons whose total > 0.5 (ratio unstable near 0)")
    sr = [d / tt for tt, d, _, _ in per.get("say", []) if tt > 0.5]; ar = [d / tt for tt, d, _, _ in per.get("attribute", []) if tt > 0.5]
    if sr and ar:
        L.append(f"prediction (a): 'say' direct/total median {md(sr):.2f} (needs >= 0.7) vs attribute {md(ar):.2f} -> "
                 f"{'HOLDS' if md(sr) >= 0.7 and md(sr) > md(ar) else 'FAILS'}")
    L.append(f"\ncomposition (earlier unit zeroed -> relative drop in a later unit's activation at the answer position; per pair, mean over prompts):")
    L.append(f"{'earlier -> later':<26}{'pairs':>7}{'median drop':>13}{'null':>9}{'real>null':>11}{'p':>9}")
    for key in sorted(comp, key=lambda k: -len(comp[k])):
        xs = comp[key]; w = sum(a > b for a, b in xs); l_ = sum(a < b for a, b in xs)
        L.append(f"{key[0] + ' -> ' + key[1]:<26}{len(xs):>7}{md(a for a, _ in xs):>13.3f}{md(b for _, b in xs):>9.3f}{f'{w}/{l_}':>11}{sign_p(w, l_):>9.2g}")
    xs = comp.get(("attribute", "say"), [])
    if xs:
        w = sum(a > b for a, b in xs); l_ = sum(a < b for a, b in xs); m, mn = md(a for a, _ in xs), md(b for _, b in xs)
        L.append(f"prediction (b): attribute -> later say: median drop {m:.3f} vs null {mn:.3f}; sign p {sign_p(w, l_):.2g} -> "
                 f"{'HOLDS' if m >= 2 * max(mn, 1e-9) and m > 0 and sign_p(w, l_) < 0.05 and w > l_ else 'FAILS'}")
    txt = "\n".join(L)
    (ROOT / "results/indirect_path_summary.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
