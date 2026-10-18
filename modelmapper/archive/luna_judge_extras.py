"""Two follow-ups on luna_firing_judge.json (no model, no API) -> results/luna_judge_extras.txt
  1. Token concentration: share of a neuron's strong web-text firings on its single most common token (and top 5),
     by Luna class and for the controls - are 'unrelated' neurons just token detectors (e.g. ' the', '.')?
  2. Strength: Luna's labels against each judged firing's activation relative to the neuron's peak on its own concept
     (the judged firings are re-drawn with the judge's own seeded sampling) - monosemantic at the top, mixed below?
"""
import json
import random
import statistics as stx
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from config import DEFAULT_MODEL, model_path
from luna_judge_summary import cls

ROOT = Path(__file__).resolve().parents[1]
BINS = [(0.5, 0.75), (0.75, 1.0), (1.0, 1.5), (1.5, float("inf"))]


def main():
    tok = AutoTokenizer.from_pretrained(model_path(DEFAULT_MODEL))
    rows = json.loads((ROOT / "results/corpus_profile_units.json").read_text(encoding="utf-8"))
    idx = {(r["neuron"], r["control"]): i for i, r in enumerate(rows)}
    J = json.loads((ROOT / "results/luna_firing_judge.json").read_text(encoding="utf-8"))["jobs"]
    z = np.load(ROOT / "results/corpus_profile.npz"); unit, doc, pos, val = z["unit"], z["doc"], z["pos"].astype(np.int64), z["val"].astype(np.float32)
    d = np.load(ROOT / f"data/atlas_{DEFAULT_MODEL.split('/')[-1]}_docs.npz")
    toks, lens = d["tokens"], d["lengths"]; starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    order = np.argsort(unit, kind="stable"); unit, doc, pos, val = unit[order], doc[order], pos[order], val[order]
    b = np.searchsorted(unit, np.arange(len(rows) + 1))
    res, ex = defaultdict(list), defaultdict(list); cnt = defaultdict(Counter); n_snip = 0
    for k, v in J.items():
        if not v.get("valid"):
            continue
        u = k.split("|")[0]; i = idx[(u, v["control"])]; sl = slice(b[i], b[i + 1])
        g = starts[doc[sl]] + pos[sl]
        t = Counter(toks[g].tolist()); top, c1 = t.most_common(1)[0]
        c = "control" if v["control"] else cls(v["labels"])
        res[c].append((c1 / len(g), sum(x for _, x in t.most_common(5)) / len(g))); ex[c].append((c1 / len(g), u, tok.decode([top]), len(g)))
        if not v["control"]:
            ch = random.Random(f"21|{k}").sample(list(range(b[i], b[i + 1])), 12)
            ref = abs(rows[i]["own_act"])
            for kk, lab in zip(ch, v["labels"]):
                r = val[kk] / ref; n_snip += 1
                for lo, hi in BINS:
                    if lo <= r < hi:
                        cnt[(lo, hi)][lab] += 1
    L = ["1. Share of a neuron's strong firings on its single most common token (median), top-5 tokens, and neurons with one token >= 50%:"]
    for c in ("dedicated", "ingredient", "mixed", "unrelated", "control"):
        r = res[c]
        L.append(f"  {c:<11} n={len(r):<4} top-1 {stx.median(x for x, _ in r):.2f}  top-5 {stx.median(y for _, y in r):.2f}  "
                 f"one token >= 50%: {sum(x >= 0.5 for x, _ in r)}")
    L.append("  most token-bound 'unrelated' concept neurons:")
    for s, u, t_, n in sorted(ex["unrelated"], reverse=True)[:12]:
        L.append(f"    {u:<10} {s:.0%} of {n} firings on {t_!r}")
    L.append(f"\n2. {n_snip} judged firings of concept neurons, by activation relative to the neuron's peak on its own concept:")
    for lo, hi in BINS:
        c = cnt[(lo, hi)]; n = sum(c.values())
        if n:
            L.append(f"  {lo:>4}-{hi:<4} x peak: n={n:<6} A {c['A'] / n:.1%}  B {c['B'] / n:.1%}  C {c['C'] / n:.1%}")
    txt = "\n".join(L)
    (ROOT / "results/luna_judge_extras.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
