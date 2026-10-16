"""Summarise bowtie_*.json: do the concept's top-5 neurons act directly, through others, or at other positions?
(no model, no API)
Usage: python src/bowtie_summary.py   -> results/bowtie_summary.txt
"""
import json
import statistics as stx
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    d = {}
    for f in ("results/concepts_v3/bowtie_cr2.json", "results/concepts_w2/bowtie_w2.json"):
        p = ROOT / f
        if p.exists():
            d.update(json.loads(p.read_text(encoding="utf-8")))
    d = {c: v for c, v in d.items() if "allpos_logodds_drop" in v["own"]}
    n = len(d); md = lambda xs: round(stx.median(xs), 2)
    o = [v["own"] for v in d.values()]; r = [v["random"] for v in d.values()]
    L = [f"Bow-tie test, {n} concepts: the concept's top-{next(iter(d.values()))['k']} neurons switched off "
         f"(answer position only, or every position); control = same number of random neurons from the same layers.\n",
         "ANSWER POSITION ONLY (answer score = answer logit minus mean logit):",
         f"  total drop median {md([x['total_drop'] for x in o])}; what the neurons wrote directly {md([x['direct'] for x in o])}; "
         f"indirect (total - direct) {md([x['indirect'] for x in o])}",
         f"  indirect < 0 (the rest of the network partly makes up for the loss) in {sum(x['indirect'] < 0 for x in o)}/{n}; "
         f"indirect > direct (amplified through others: a bow-tie) in {sum(x['indirect'] > x['direct'] for x in o)}/{n}",
         f"  of the indirect part: via heads median {md([x['via_heads'] for x in o])}, via other neurons {md([x['via_other_neurons'] for x in o])}",
         f"  random neurons: total drop median {md([x['total_drop'] for x in r])}",
         "\nRECALL (answer first-token log-odds) drop:",
         f"  answer position only: median {md([x['logodds_drop'] for x in o])}   every position: median {md([x['allpos_logodds_drop'] for x in o])}   "
         f"random every position: {md([x['allpos_logodds_drop'] for x in r])}",
         f"  share of the every-position effect already there at the answer position alone: median "
         f"{md([x['logodds_drop'] / x['allpos_logodds_drop'] for x in o if x['allpos_logodds_drop'] > 0.2])} "
         f"(concepts with every-position drop > 0.2: {sum(x['allpos_logodds_drop'] > 0.2 for x in o)}/{n})",
         f"  every-position drop > 2x the answer-position drop in {sum(x['allpos_logodds_drop'] > 2 * max(x['logodds_drop'], 0.05) for x in o)}/{n}",
         "\nper concept: answer-pos score drop = direct + indirect | log-odds drop answer pos / every pos | random every pos"]
    for c, v in sorted(d.items(), key=lambda kv: -kv[1]["own"]["allpos_logodds_drop"]):
        x, y = v["own"], v["random"]
        L.append(f"  {c:<30} {x['total_drop']:+6.1f} = {x['direct']:+5.1f} {x['indirect']:+5.1f} | {x['logodds_drop']:+5.2f} / {x['allpos_logodds_drop']:+5.2f} | {y['allpos_logodds_drop']:+5.2f}")
    txt = "\n".join(L)
    (ROOT / "results/bowtie_summary.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
