"""Does 'a handful of neurons' hold for obscure concepts? Summary of the fame-stratified wave (no model, no API).

Reads results/concepts_fame/wave_free.json (wave_budget.py round-robin run; categories '<tier>#k') and
results/wiki_sample.json (annual 2025 page views). Per tier: how many concepts were tried, prompt failures, recall rate
(>= 5 of 20 questions answered), and for recalled ones the log-odds measures k_mean (top-k off lowers answer log-odds by
>= 3 on average), k_flip (top-k off flips >= half the correct answers), baseline log-odds, and steering (held-out
share of fluent on-topic replies; random-neuron control). Plus Spearman correlations with log10(views) over recalled concepts.
Usage: python src/fame_summary.py  -> results/fame_summary.txt
"""
import json
import math
import statistics as stx
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def spearman(x, y):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            for k in range(i, j + 1):
                r[o[k]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = rank(x), rank(y); mx, my = stx.mean(rx), stx.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry)); den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def main():
    w = json.loads((ROOT / "results/concepts_fame/wave_free.json").read_text(encoding="utf-8"))
    views = json.loads((ROOT / "results/wiki_sample.json").read_text(encoding="utf-8"))["views"]
    tiers = defaultdict(list)
    for c, r in w.items():
        tiers[r["category"].split("#")[0]].append((c, r))
    md = lambda xs: (stx.median(xs) if xs else float("nan"))
    kv = lambda x: x if x is not None else 99              # 'not reached within 50' sorts last; shown as >50
    L = [f"Fame-stratified wave: {len(w)} concepts tried. k values: smallest k in 1,2,3,5,10,20,50 meeting the bar ('>50' = never).\n",
         f"{'tier':<13} {'tried':>5} {'prompts failed':>14} {'recalled':>9} | median k_mean  k_mean<=5  median k_flip  k_flip<=5 | base LO | steer held-out (random)"]
    allx = []
    for t in sorted(tiers):
        rows = tiers[t]; ok = [r for _, r in rows if r.get("status") == "ok"]
        pf = sum(r.get("status") == "prompts_failed" for _, r in rows)
        km = [kv(r["k_mean"]) for r in ok]; kf = [kv(r["k_flip"]) for r in ok]
        st = [r["steer"]["heldout"] for r in ok if "steer" in r]; sr = [r["steer"]["heldout_random"] for r in ok if "steer" in r]
        L.append(f"{t:<13} {len(rows):>5} {pf:>14} {len(ok):>4}/{len(rows) - pf:<4} | {md(km):>13} {sum(k <= 5 for k in km):>5}/{len(ok):<4} "
                 f"{md(kf):>11} {sum(k <= 5 for k in kf):>6}/{len(ok):<4} | {md([r['baseline_mean_logodds'] for r in ok]):>7.2f} | "
                 f"{md(st):.2f} ({md(sr):.2f})")
        for c, r in rows:
            if r.get("status") == "ok":
                allx.append((math.log10(max(views.get(c, 1), 1)), kv(r["k_mean"]), kv(r["k_flip"]), r["baseline_mean_logodds"]))
    if len(allx) >= 5:
        lv = [a for a, *_ in allx]
        L.append(f"\nSpearman with log10(page views), recalled concepts (n={len(allx)}): k_mean {spearman(lv, [b for _, b, _, _ in allx]):+.2f}, "
                 f"k_flip {spearman(lv, [c for _, _, c, _ in allx]):+.2f}, baseline log-odds {spearman(lv, [d for *_, d in allx]):+.2f}")
    L.append("\nper concept (tier | views | status | k_mean / k_flip | baseline):")
    for t in sorted(tiers):
        for c, r in tiers[t]:
            L.append(f"  {t:<12} {views.get(c, 0):>10,} {c[:40]:<40} {r.get('status'):<14} "
                     + (f"{r['k_mean']}/{r['k_flip']}  base {r['baseline_mean_logodds']}" if r.get("status") == "ok" else f"recall {r.get('recall', '-')}"))
    txt = "\n".join(L)
    (ROOT / "results/fame_summary.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
