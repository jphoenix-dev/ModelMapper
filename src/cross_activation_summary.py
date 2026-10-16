"""Summarise the cross-activation test: dedicated, family-shared or broadly shared concept neurons? (no model, no API)

Rules (fixed before the results existed):
  ratio      a neuron's activation on concept B, in its own direction, divided by its activation on its OWN concept A
             (both at the answer position, mean over prompts; 'peak' = the same with B's strongest position)
  fires      ratio >= 0.5
  excluded   neuron-owner pairs whose own activation |a_A| < 1 (barely active at the answer position; counted)
  families   families.py (fixed beforehand)
  sanity     ratio of A's neurons on A itself must be ~1 (median within 0.9-1.1) - else the measurement is broken
Also: how often two concepts' top-10 lists share a neuron, within vs across families (the shared-parts measure at scale).
Usage: python src/cross_activation_summary.py   -> results/cross_activation_summary.txt / .json
"""
import glob
import json
import statistics as stx
from collections import Counter, defaultdict
from pathlib import Path

from families import family_map

ROOT = Path(__file__).resolve().parents[1]
TOP, FIRE, MIN_OWN = 10, 0.5, 1.0


def main():
    xs = {}
    units = None
    for f in sorted(glob.glob(str(ROOT / "results/cross_activation_*.json"))):
        if f.endswith("summary.json"):
            continue
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        assert units is None or d["units"] == units, f"{f}: different neuron union"
        units = d["units"]; xs.update(d["concepts"])
    ui = {u: i for i, u in enumerate(units)}
    ranks = {}
    for f in sorted(glob.glob(str(ROOT / "results/concepts_*/rank_save_*.json"))):
        if "rank_save_test" not in f and not f.endswith("_exact.json"):
            ranks.update({c: v for c, v in json.loads(Path(f).read_text(encoding="utf-8")).items() if "top" in v})
    fam = family_map()
    concepts = [c for c in ranks if c in xs]
    missing_fam = [c for c in concepts if c not in fam]
    L = [f"Cross-activation: {len(concepts)} concepts, {len(units)} distinct top-{TOP} neurons. 'fires' = >= {FIRE} of its own-concept "
         f"activation (answer position). Families: {len(set(fam[c] for c in concepts if c in fam))}; concepts without a family: {missing_fam}\n"]
    pairs, excluded, self_r = [], 0, []
    for a in concepts:
        for e in ranks[a]["top"][:TOP]:
            u = e["neuron"]; own = e["act_answer_pos"]
            if abs(own) < MIN_OWN:
                excluded += 1; continue
            sg = 1.0 if own >= 0 else -1.0; i = ui[u]
            self_r.append(xs[a]["ans"][i] * sg / abs(own))
            row = {"neuron": u, "owner": a, "layer": int(u[1:].split(":")[0]), "own": own, "same": [], "diff": [], "peak_same": [], "peak_diff": []}
            for b in concepts:
                if b == a:
                    continue
                r = xs[b]["ans"][i] * sg / abs(own); rp = xs[b]["peak"][i] * sg / abs(own)
                same = fam.get(a) is not None and fam.get(a) == fam.get(b)
                row["same" if same else "diff"].append((b, r)); row["peak_same" if same else "peak_diff"].append((b, rp))
            pairs.append(row)
    ms = stx.median(self_r)
    L.append(f"sanity: own-concept ratio median {ms:.3f} (should be ~1) over {len(self_r)} neuron-owner pairs; excluded (|own act| < {MIN_OWN}): {excluded}")
    assert 0.9 <= ms <= 1.1, "own-concept ratio is not ~1: the measurement is broken"
    def rate(rows, key):
        hits = sum(sum(r >= FIRE for _, r in row[key]) for row in rows); n = sum(len(row[key]) for row in rows)
        return hits / n if n else float("nan"), hits, n
    for lab, key_s, key_d in (("answer position", "same", "diff"), ("strongest position", "peak_same", "peak_diff")):
        s, hs, ns = rate(pairs, key_s); d, hd, nd = rate(pairs, key_d)
        L.append(f"\n[{lab}] P(fires for another concept): same family {s:.3%} ({hs}/{ns}) vs different family {d:.3%} ({hd}/{nd}); "
                 f"ratio {s / d if d else float('inf'):.1f}x")
    n_other = [sum(r >= FIRE for _, r in row["same"] + row["diff"]) for row in pairs]
    n_diff = [sum(r >= FIRE for _, r in row["diff"]) for row in pairs]
    tot_other = len(concepts) - 1
    L.append(f"\nper neuron-owner pair (answer position): fires for 0 other concepts in {sum(n == 0 for n in n_other)}/{len(pairs)}; "
             f"1-2 in {sum(1 <= n <= 2 for n in n_other)}; 3-9 in {sum(3 <= n <= 9 for n in n_other)}; >= 10 in {sum(n >= 10 for n in n_other)} "
             f"(of {tot_other} others)")
    cls = Counter()
    for row, n, nd in zip(pairs, n_other, n_diff):
        nsm = n - nd
        cls["dedicated (no other concept)" if n == 0 else "family-shared (others mostly same family)" if nsm >= nd else
            "broad (>= 10% of other-family concepts)" if nd >= 0.1 * len(row["diff"]) else "a few unrelated"] += 1
    L.append("classes: " + ", ".join(f"{k} {v}" for k, v in cls.most_common()))
    # POST HOC (chosen after the first look showed 'always on' machinery firing for every concept): split off neurons
    # that fire for >= 90% of the other concepts, then recompute the family comparison on the rest
    always = {row["neuron"] for row, n in zip(pairs, n_other) if n >= 0.9 * tot_other}
    rest = [(row, n) for row, n in zip(pairs, n_other) if row["neuron"] not in always]
    L.append(f"\nPOST HOC split: 'always on' = fires for >= 90% of the other concepts: {len(always)} neurons "
             f"({sum(row['neuron'] in always for row in pairs)} neuron-owner pairs): {sorted(always)}")
    s, hs, ns = rate([r for r, _ in rest], "same"); d, hd, nd = rate([r for r, _ in rest], "diff")
    L.append(f"  the rest ({len(rest)} pairs): P(fires for another concept) same family {s:.2%} ({hs}/{ns}) vs different family "
             f"{d:.2%} ({hd}/{nd}); ratio {s / d if d else float('inf'):.1f}x")
    nr = [n for _, n in rest]
    L.append(f"  the rest: fires for 0 others {sum(n == 0 for n in nr)}/{len(nr)}; 1-2 {sum(1 <= n <= 2 for n in nr)}; 3-9 "
             f"{sum(3 <= n <= 9 for n in nr)}; >= 10 {sum(n >= 10 for n in nr)}")
    cls2 = Counter()
    for row, n in rest:
        nd_ = sum(r >= FIRE for _, r in row["diff"]); nsm = n - nd_
        cls2["dedicated" if n == 0 else "family-shared" if nsm >= nd_ else "broad" if nd_ >= 0.1 * len(row["diff"]) else "a few unrelated"] += 1
    L.append("  the rest, classes: " + ", ".join(f"{k} {v}" for k, v in cls2.most_common()))
    fams = sorted({fam[c] for c in concepts})
    L.append("  the rest, per family: pairs | P(fires) same family | different family")
    for f_ in fams:
        rr = [r for r, _ in rest if fam.get(r["owner"]) == f_]
        if rr:
            s_, _, ns_ = rate(rr, "same"); d_, _, _ = rate(rr, "diff")
            L.append(f"    {f_:<12} {len(rr):>4} | {s_:6.1%} (n={ns_}) | {d_:6.1%}")
    L.append("\nby layer band (answer position): pairs | fires for 0 others | median ratio on different-family concepts")
    for lo, hi in ((0, 16), (16, 24), (24, 30), (30, 36)):
        rr = [(row, n) for row, n in zip(pairs, n_other) if lo <= row["layer"] < hi]
        if rr:
            L.append(f"  L{lo}-{hi - 1}: {len(rr):>4} | {sum(n == 0 for _, n in rr) / len(rr):.0%} | "
                     f"{stx.median(stx.median(r for _, r in row['diff']) for row, _ in rr):+.3f}")
    # overlap of the top-10 lists themselves
    sets = {c: {e["neuron"] for e in ranks[c]["top"][:TOP]} for c in concepts}
    sp = Counter(); tp = Counter()
    for i, a in enumerate(concepts):
        for b in concepts[i + 1:]:
            k = "same" if fam.get(a) == fam.get(b) and fam.get(a) else "diff"
            tp[k] += 1; sp[k] += bool(sets[a] & sets[b])
    L.append(f"\ntop-10 lists sharing >= 1 neuron: same family {sp['same']}/{tp['same']} ({sp['same'] / tp['same']:.1%}) vs different "
             f"{sp['diff']}/{tp['diff']} ({sp['diff'] / tp['diff']:.1%})")
    multi = Counter(u for c in concepts for u in sets[c])
    L.append("neurons in >= 3 concepts' top-10: " + ", ".join(
        f"{u} x{k} ({', '.join(sorted({fam.get(c, '?') for c in concepts if u in sets[c]}))})" for u, k in multi.most_common() if k >= 3))
    L.append("\nmost broadly firing neurons (answer position; other concepts it fires for, by family):")
    for row, n in sorted(zip(pairs, n_other), key=lambda t: -t[1])[:12]:
        fs = Counter(fam.get(b, "?") for b, r in row["same"] + row["diff"] if r >= FIRE)
        L.append(f"  {row['neuron']:<10} (owner {row['owner']}, act {row['own']:+.1f}) fires for {n}: {dict(fs.most_common(5))}")
    L.append("\nexamples of family-shared firing (owner -> other same-family concepts it fires for):")
    ex = [(row, [b for b, r in row["same"] if r >= FIRE]) for row in pairs]
    for row, bs in sorted(ex, key=lambda t: -len(t[1]))[:12]:
        if bs:
            L.append(f"  {row['neuron']:<10} {row['owner']} -> {bs[:6]}")
    txt = "\n".join(L)
    (ROOT / "results/cross_activation_summary.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
