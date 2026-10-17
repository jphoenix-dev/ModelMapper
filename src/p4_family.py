"""PREREG_MAP P4: does the existing map predict which neurons NEW concepts use? (no model, no API)

predict (run BEFORE coverage-4 exists): from every mapped concept so far (rank_save_*.json, excluding coverage-4 =
concepts_fame5), with families = first Wikidata P31 label (wiki_families.json):
  generic neurons  = in the top-30 of >= 15% of all mapped concepts ('always on' machinery; detail fixed here, before data)
  family neurons   = for families with >= 5 mapped members: in the top-30 of >= 20% of members, minus generic neurons
  -> results/p4_predictions.json (committed before coverage-4 runs)
score (after coverage-4): for each coverage-4 concept in a predicted family, own-family hit rate = |top30 & F_own| /
  |F_own|; other-family rate = mean over the other predicted families of |top30 & F| / |F|. Prediction (PREREG_MAP):
  median own/other >= 3x, and own > other in significantly more concepts than chance (one-sided sign test p < 0.05).
Usage: python src/p4_family.py predict | score
"""
import glob
import json
import math
import statistics as stx
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERIC_SHARE, FAMILY_SHARE, MIN_MEMBERS, TOP = 0.15, 0.20, 5, 30


def tops(exclude_dirs=(), only_dirs=None):
    out = {}
    for f in sorted(glob.glob(str(ROOT / "results/concepts_*/rank_save_*.json"))):
        if "rank_save_test" in f or f.endswith("_exact.json"):
            continue
        d = Path(f).parent.name
        if d in exclude_dirs or (only_dirs and d not in only_dirs):
            continue
        for c, v in json.loads(Path(f).read_text(encoding="utf-8")).items():
            if "top" in v and c not in out:
                out[c] = [e["neuron"] for e in v["top"][:TOP]]
    return out


def main():
    mode = sys.argv[1]
    fam = json.loads((ROOT / "results/wiki_families.json").read_text(encoding="utf-8"))
    family = {c: v["p31_labels"][0] for c, v in fam.items() if c != "_unresolved" and v["p31_labels"]}
    if mode == "predict":
        assert not (ROOT / "results/concepts_fame5/map_loop_fame5.json").exists(), "coverage-4 already ran - predictions must come first"
        t = tops(exclude_dirs=("concepts_fame5",))
        n = len(t); cnt = Counter(u for ns in t.values() for u in set(ns))
        generic = {u for u, k in cnt.items() if k >= GENERIC_SHARE * n}
        members = defaultdict(list)
        for c in t:
            if c in family:
                members[family[c]].append(c)
        pred = {}
        for f, cs in members.items():
            if len(cs) < MIN_MEMBERS:
                continue
            fc = Counter(u for c in cs for u in set(t[c]))
            pred[f] = sorted(u for u, k in fc.items() if k >= FAMILY_SHARE * len(cs) and u not in generic)
        out = {"mapped_concepts": n, "generic_neurons": sorted(generic), "families": {f: {"members": len(members[f]), "neurons": v}
               for f, v in pred.items()}, "rules": {"generic_share": GENERIC_SHARE, "family_share": FAMILY_SHARE, "min_members": MIN_MEMBERS, "top": TOP}}
        (ROOT / "results/p4_predictions.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"{n} mapped concepts; {len(generic)} generic neurons; families predicted: "
              + ", ".join(f"{f} ({out['families'][f]['members']} members, {len(v)} neurons)" for f, v in pred.items()))
    else:
        P = json.loads((ROOT / "results/p4_predictions.json").read_text(encoding="utf-8"))
        F = {f: set(v["neurons"]) for f, v in P["families"].items() if v["neurons"]}
        new = tops(only_dirs=("concepts_fame5",))
        rows = []
        for c, ns in new.items():
            f = family.get(c)
            if f not in F:
                continue
            s = set(ns)
            own = len(s & F[f]) / len(F[f])
            other = stx.mean(len(s & F[g]) / len(F[g]) for g in F if g != f)
            rows.append((c, f, own, other))
        L = [f"P4 score: {len(rows)} coverage-4 concepts in predicted families ({dict(Counter(r[1] for r in rows))})"]
        if rows:
            ratios = [(o + 1e-9) / (x + 1e-9) if x > 0 else (float('inf') if o > 0 else 1.0) for _, _, o, x in rows]
            wins = sum(o > x for _, _, o, x in rows); ties = sum(o == x for _, _, o, x in rows); k = len(rows) - ties
            p = sum(math.comb(k, i) for i in range(wins, k + 1)) / 2 ** k if k else 1.0
            med = stx.median(ratios)
            L.append(f"median own/other hit-rate ratio {med:.2f}; own > other in {wins}/{k} (ties {ties}); one-sided sign test p = {p:.2g}")
            L.append(f"median own-family hit rate {stx.median(r[2] for r in rows):.3f} vs other-family {stx.median(r[3] for r in rows):.3f}")
            L.append(f"PREDICTION {'HOLDS' if med >= 3 and p < 0.05 else 'FAILS'} (needs median ratio >= 3 and p < 0.05)")
            for f in F:
                rr = [r for r in rows if r[1] == f]
                if rr:
                    L.append(f"  {f:<28} n={len(rr):<4} own {stx.median(r[2] for r in rr):.3f} other {stx.median(r[3] for r in rr):.3f}")
        txt = "\n".join(L)
        (ROOT / "results/p4_score.txt").write_text(txt + "\n", encoding="utf-8")
        print(txt)


if __name__ == "__main__":
    main()
