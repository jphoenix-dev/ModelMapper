"""Summarise staging_*.json (logit lens over positions x layers): where the answer becomes readable before it is
spoken. Ranks among ~152k words; control = another concept's answer token. (no model, no API)
Usage: python src/staging_summary.py  -> results/staging_summary.txt
"""
import json
import statistics as stx
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    x = {}
    for f in ("results/concepts_v3/staging_cr2.json", "results/concepts_w2/staging_w2.json"):
        p = ROOT / f
        if p.exists():
            x.update(json.loads(p.read_text(encoding="utf-8")))
    agg = defaultdict(lambda: {"t10": [], "t1": [], "r": defaultdict(list), "c": defaultdict(list), "n": 0})
    for v in x.values():
        for pos, r in v.items():
            a = agg[pos]; a["n"] += 1; a["t10"] += r["first_top10"]; a["t1"] += r["first_top1"]
            for l, rk in r["rank_at"].items():
                a["r"][int(l)].append(rk)
            for l, rk in r["ctrl_rank_at"].items():
                a["c"][int(l)].append(rk)
    L = [f"Logit lens, {len(x)} concepts: where is the answer readable before the answer position? (per-concept median ranks,"
         " then median over concepts; positions present in >= 10 concepts)\n",
         f"{'position':<26} {'concepts':>8}  top-10 reached (prompts)  median first layer  top-1   rank at L20/25/30/33/35   | control"]
    for k, a in sorted(agg.items(), key=lambda kv: -kv[1]["n"]):
        if a["n"] < 10:
            continue
        t10 = [z for z in a["t10"] if z is not None]; t1 = [z for z in a["t1"] if z is not None]
        L.append(f"{k:<26} {a['n']:>8}  {len(t10):>4}/{len(a['t10']):<4} {len(t10) / len(a['t10']):>5.0%}        "
                 f"{(stx.median(t10) if t10 else '-'):>5}        {len(t1) / len(a['t1']):>4.0%}   "
                 + "/".join(str(int(stx.median(a['r'][l]))) for l in sorted(a['r'])) + "   | "
                 + "/".join(str(int(stx.median(a['c'][l]))) for l in sorted(a['c'])))
    txt = "\n".join(L)
    (ROOT / "results/staging_summary.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
