"""Summarise meaning_judge.json (stage A): how many meanings, how related - calibrated on spliced units. (no model, no API)

Calibration (spliced = 15 firings from each of two concept neurons of different families): 'recovered' if Luna's
groups separate the sources - at least two groups with different majority sources, each >= 80% pure, together covering
>= 60% of the 30 snippets - and then the rating between such groups (should be 2-3).
For concept neurons and controls: number of groups, share in 'misc', the largest relatedness rating among the groups
(3 = has a pair of unrelated meanings), and the share with exactly one meaning.
Usage: python src/meaning_judge_summary.py  -> results/meaning_judge_summary.txt
"""
import json
import statistics as stx
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    J = json.loads((ROOT / "results/meaning_judge.json").read_text(encoding="utf-8"))["jobs"]
    by = {"concept": [], "control": [], "spliced": []}
    bad = Counter()
    for k, v in J.items():
        (by[v["kind"]].append(v) if v.get("valid") else bad.update([v["kind"]]))
    L = [f"Blind meaning grouping (Luna, no concept named), 30 random strong firings per unit. Valid: "
         + ", ".join(f"{k} {len(v)}" for k, v in by.items()) + f"; invalid: {dict(bad)}\n"]
    rec = 0; rec_ratings = []; sp = by["spliced"]
    for v in sp:
        src = v["source"]; groups = v["result"]["groups"]
        good = []
        for gi, g in enumerate(groups):
            mem = [int(m) - 1 for m in g["members"]]
            if not mem:
                continue
            c = Counter(src[m] for m in mem); s, top = c.most_common(1)[0]
            if top / len(mem) >= 0.8:
                good.append((gi, s, len(mem)))
        srcs = {s for _, s, _ in good}
        ok = len(srcs) >= 2 and sum(n for _, _, n in good) >= 18
        rec += ok
        if ok:
            rel = {(int(p["a"]), int(p["b"])): int(p["rating"]) for p in v["result"]["relatedness"]}
            rr = [rel.get((a, b), rel.get((b, a))) for a, sa, _ in good for b, sb, _ in good if a < b and sa != sb]
            rec_ratings += [x for x in rr if x is not None]
    if sp:
        L.append(f"CALIBRATION - spliced two-neuron units: Luna separated the two sources in {rec}/{len(sp)} ({rec / len(sp):.0%}); "
                 f"ratings between correctly separated groups: {dict(sorted(Counter(rec_ratings).items()))} (3 = unrelated)")
        L.append(f"  spliced units given ONE group: {sum(len(v['result']['groups']) == 1 for v in sp)}/{len(sp)}\n")
    L.append(f"{'kind':<9} {'n':>4} | groups: 1 / 2 / 3 / 4+ | median misc | max pair rating 0 / 1 / 2 / 3 (among units with >= 2 groups)")
    for kind in ("concept", "control", "spliced"):
        vs = by[kind]
        if not vs:
            continue
        ng = [len(v["result"]["groups"]) for v in vs]
        mx = []
        for v in vs:
            rs = [int(p["rating"]) for p in v["result"]["relatedness"]]
            if rs:
                mx.append(max(rs))
        c = Counter(mx)
        L.append(f"{kind:<9} {len(vs):>4} | {sum(x == 1 for x in ng):>3} / {sum(x == 2 for x in ng):>3} / {sum(x == 3 for x in ng):>3} / "
                 f"{sum(x >= 4 for x in ng):>3} | {stx.median(len(v['result'].get('misc', [])) for v in vs):>4} | "
                 f"{c[0]:>3} / {c[1]:>3} / {c[2]:>3} / {c[3]:>3}")
    con = by["concept"]
    if con:
        L.append("\nconcept neurons by layer band: n | one meaning | has an unrelated pair (rating 3)")
        for lo, hi in ((0, 16), (16, 24), (24, 30), (30, 36)):
            vs = [v for v in con if lo <= v["layer"] < hi]
            if vs:
                one = sum(len(v["result"]["groups"]) == 1 for v in vs)
                unrel = sum(any(int(p["rating"]) == 3 for p in v["result"]["relatedness"]) for v in vs)
                L.append(f"  L{lo}-{hi - 1}: {len(vs):>3} | {one / len(vs):.0%} | {unrel / len(vs):.0%}")
        L.append("\nexamples (concept neuron: owner concept -> Luna's meaning groups, sizes; ratings):")
        for v in con[:25]:
            gs = v["result"]["groups"]
            L.append(f"  {v['units'][0]:<10} {', '.join(v['concepts'][0])[:28]:<28} -> " + " | ".join(
                f"{g.get('label')} ({len(g['members'])})" for g in gs) + "  " + str([int(p['rating']) for p in v['result']['relatedness']]))
    txt = "\n".join(L)
    (ROOT / "results/meaning_judge_summary.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
