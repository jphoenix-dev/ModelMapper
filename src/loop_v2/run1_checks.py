"""Run 1 checks that need more than report.py (no model, no API; seconds).

(a) P4 proper: families = first Wikidata P31 type (as p4_family.py), not Vital Articles sections (too broad). Family
    neurons from the first half of finished batches (top-30 of >= 20% of a family's >= 5 members, minus neurons in
    >= 15% of all maps); scored on the second half: own-family vs other-family hit rate (median ratio, sign test).
(b) Is the common-noun 'say neuron' advantage an artefact of answer tokenisation? A 'say' neuron = the answer's FIRST
    token in its top-10 pushed tokens; a one-token answer (Bicycle) is fully pushed by such a neuron, a multi-token
    name (Bossa nova) only starts. Split named / common by whether the answer is one token.
Usage: python src/loop_v2/run1_checks.py <run_dir> -> <run_dir>/run1_checks.txt
"""
import glob
import json
import math
import statistics as stx
import sys
from collections import Counter, defaultdict
from pathlib import Path

run = Path(sys.argv[1])
FR = Path("D:/already-clean-data/loop_v2/frame")


def sign_p(w, l_):
    n = w + l_
    return 1.0 if not n else min(1.0, 2 * sum(math.comb(n, i) for i in range(max(w, l_), n + 1)) / 2 ** n)


def main():
    F = {x["concept"]: x for x in json.loads((FR / "frame.json").read_text(encoding="utf-8"))["draw"]}
    V = json.loads((FR / "vital.json").read_text(encoding="utf-8"))
    recs = {}
    for b in sorted(glob.glob(str(run / "results/**/map/*/records.jsonl"), recursive=True)):
        for line in Path(b).read_text(encoding="utf-8").splitlines():
            r = json.loads(line); r["_batch"] = Path(b).parent.name; recs[r["concept"]] = r
    ok = {c: r for c, r in recs.items() if r["row"]["status"] == "ok"}
    top30 = {c: [e["neuron"] for e in r["rank"]["top"]] for c, r in ok.items()}
    p31 = {c: (V["wd"].get(V["qid"].get(c) or "", {}).get("p31") or [None])[0] for c in ok}
    L = []
    # (a)
    batches = sorted({r["_batch"] for r in ok.values()}); half = set(batches[: len(batches) // 2])
    first = [c for c in ok if ok[c]["_batch"] in half]; second = [c for c in ok if ok[c]["_batch"] not in half]
    allc = Counter(u for c in first for u in set(top30[c])); gen = {u for u, k in allc.items() if k >= 0.15 * len(first)}
    mem = defaultdict(list)
    for c in first:
        mem[p31[c]].append(c)
    fam = {}
    for f, cs in mem.items():
        if f and len(cs) >= 5:
            fc = Counter(u for c in cs for u in set(top30[c]))
            s = {u for u, k in fc.items() if k >= 0.2 * len(cs) and u not in gen}
            if s:
                fam[f] = s
    rows = []
    for c in second:
        f = p31[c]
        if f in fam and len(fam) > 1:
            own = len(set(top30[c]) & fam[f]) / len(fam[f])
            oth = stx.mean(len(set(top30[c]) & fam[g]) / len(fam[g]) for g in fam if g != f)
            rows.append((f, own, oth))
    if rows:
        ratios = [(o + 1e-9) / (x + 1e-9) if x > 0 else (float("inf") if o > 0 else 1.0) for _, o, x in rows]
        w = sum(o > x for _, o, x in rows); l_ = sum(o < x for _, o, x in rows)
        L.append(f"(a) P4 with Wikidata P31 families: {len(fam)} families from {len(first)} first-half maps (generic neurons removed: {len(gen)}); "
                 f"scored on {len(rows)} second-half concepts in those families")
        L.append(f"    median own/other ratio {stx.median(ratios):.2f}; own > other {w}, < {l_} (sign p {sign_p(w, l_):.2g}); "
                 f"median own {stx.median(o for _, o, _ in rows):.3f} vs other {stx.median(x for _, _, x in rows):.3f}")
        L.append(f"    (P4 on the earlier maps: ratio 9.18, 118/185, p = 0.0001; pre-registered criterion: ratio >= 3 and p < 0.05)")
        by = defaultdict(list)
        for f, o, x in rows:
            by[f].append((o, x))
        big = sorted(by, key=lambda f: -len(by[f]))[:8]
        L.append("    largest families (Wikidata id: n, median own vs other): " + "; ".join(
            f"{f}: {len(by[f])}, {stx.median(o for o, _ in by[f]):.3f} vs {stx.median(x for _, x in by[f]):.3f}" for f in big))
    # (b)
    L.append("\n(b) 'say' neuron in the top-5, by named/common x one-token answer:")
    for kind in ("named", "common"):
        for one in (True, False):
            rs = [r for c, r in ok.items() if F[c].get("kind") == kind and (len(r["rank"].get("answer_ids", [0, 0])) == 1) == one]
            if rs:
                say = sum(any(x["role"] == "say" for x in r["v2"]["roles_top30"][:5]) for r in rs)
                km = stx.median(r["row"]["k_mean"] for r in rs if r["row"].get("k_mean"))
                L.append(f"    {kind:<7} {'one-token' if one else 'multi-token':<12} n={len(rs):<5} say in top-5 {say / len(rs):.0%}  k_mean median {km}")
    txt = "\n".join(L)
    (run / "run1_checks.txt").write_text(txt + "\n", encoding="utf-8"); print(txt)


if __name__ == "__main__":
    main()
