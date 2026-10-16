"""Summarise luna_firing_judge.json: dedicated, ingredient, unrelated or mixed concept neurons? (no model, no API)

Per neuron, 12 random firings labelled A (about the concept) / B (related: same topic or broader category) /
C (unrelated). Classes (fixed before the results were read):
  dedicated   A >= 6
  ingredient  A < 6 and A + B >= 9
  unrelated   C >= 6
  mixed       anything else
Controls (random same-layer neurons judged against the same concept name) go through the same rules: their rates
are Luna's leniency baseline. Usage: python src/luna_judge_summary.py  -> results/luna_judge_summary.txt
"""
import json
import statistics as stx
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cls(labels):
    a, b, c = labels.count("A"), labels.count("B"), labels.count("C")
    return "dedicated" if a >= 6 else "ingredient" if a + b >= 9 else "unrelated" if c >= 6 else "mixed"


def main():
    d = json.loads((ROOT / "results/luna_firing_judge.json").read_text(encoding="utf-8"))
    prof = json.loads((ROOT / "results/corpus_profile_summary.json").read_text(encoding="utf-8"))
    jobs = d["jobs"]
    valid = {k: v for k, v in jobs.items() if v.get("valid")}
    bad = [k for k, v in jobs.items() if not v.get("valid")]
    con = {k: v for k, v in valid.items() if not v["control"]}; ctl = {k: v for k, v in valid.items() if v["control"]}
    L = [f"Luna firing judge: {len(jobs)} jobs, {len(valid)} valid ({len(con)} concept neurons, {len(ctl)} controls); invalid/skipped: "
         f"{len(bad)} {bad}. Estimated spend ${d.get('estimated_usd', 0):.2f} at the assumed (pessimistic) price.\n"]
    def frac(vs):
        c = Counter(x for v in vs for x in v["labels"]); n = sum(c.values())
        return {k: c[k] / n for k in "ABC"} if n else {}
    for name, vs in (("concept neurons", con.values()), ("controls (random neurons, same concept named)", ctl.values())):
        f = frac(list(vs)); k = Counter(cls(v["labels"]) for v in vs); n = sum(k.values())
        L.append(f"{name}: snippets A {f.get('A', 0):.1%}  B {f.get('B', 0):.1%}  C {f.get('C', 0):.1%} | classes: "
                 + ", ".join(f"{c_} {k[c_]} ({k[c_] / n:.0%})" for c_ in ("dedicated", "ingredient", "mixed", "unrelated")))
    L.append("\nconcept neurons by layer band: n | A | B | C | dedicated / ingredient / mixed / unrelated")
    for lo, hi in ((0, 16), (16, 24), (24, 30), (30, 36)):
        vs = [v for v in con.values() if lo <= v["layer"] < hi]
        if vs:
            f = frac(vs); k = Counter(cls(v["labels"]) for v in vs)
            L.append(f"  L{lo}-{hi - 1}: {len(vs):>3} | {f['A']:.0%} | {f['B']:.0%} | {f['C']:.0%} | "
                     f"{k['dedicated']} / {k['ingredient']} / {k['mixed']} / {k['unrelated']}")
    # agreement with the keyword measure
    pairs = [(sum(x == "A" for x in v["labels"]) / 12, prof[k.split("|")[0]]["on"]) for k, v in con.items() if k.split("|")[0] in prof]
    if pairs:
        hi_ = [a for a, o in pairs if o >= 0.2]; lo_ = [a for a, o in pairs if o < 0.02]
        L.append(f"\nagreement with the keyword measure: Luna's A share median {stx.median(hi_) if hi_ else float('nan'):.2f} where the "
                 f"keyword on-concept share >= 0.2 (n={len(hi_)}) vs {stx.median(lo_) if lo_ else float('nan'):.2f} where < 0.02 (n={len(lo_)})")
    L.append("\nthemes Luna gave for the unrelated firings of 'unrelated' and 'mixed' concept neurons (all of them):")
    for k, v in con.items():
        if cls(v["labels"]) in ("unrelated", "mixed"):
            L.append(f"  {k.split('|')[0]:<10} {v['concept'][:30]:<30} {''.join(v['labels'])}  -> {v.get('theme_of_unrelated')}")
    L.append("\nexamples of 'ingredient' neurons (concept -> what Luna saw):")
    for k, v in list(con.items()):
        if cls(v["labels"]) == "ingredient":
            L.append(f"  {k.split('|')[0]:<10} {v['concept'][:30]:<30} {''.join(v['labels'])}")
    txt = "\n".join(L)
    (ROOT / "results/luna_judge_summary.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
