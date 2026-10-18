"""Cross-position summary (no model, no API). Reads results/cross_position.jsonl (see cross_position.py).
Per (concept, U -> S pair, prompt), in S's own sign, relative to |S as is|:
  early = drop of S when U is zeroed at every earlier word; ans = drop when U is zeroed at the answer word only;
  null = drop when an activation-matched neuron is zeroed at every earlier word.
  attribution: each head's share of S's 'early' change (gradient x the head's output change at the answer word).
(c1) earlier-word part real: median early >= 3x null and sign test early > null p < 0.05 (per pair, mean over prompts).
(c2) few heads: real patching of the top-3 heads recovers >= 50% of the in-batch drop (median, pair-prompts with a drop
     >= 5% of |S|) vs <= 10% for 3 random heads; and the top-3 heads' share of the summed head attribution.
(c3) recurrence: how often each head is among a pair-prompt's top-3 carriers vs the uniform expectation (3 / number of
     heads in layers L_U+1..L_S), over pair-prompts whose head attribution total is >= 1% of |S|.
Usage: python src/cross_position_summary.py -> results/cross_position_summary.txt
"""
import json
import math
import statistics as stx
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
H = 32


def sign_p(pos, neg):
    n = pos + neg
    return 1.0 if not n else min(1.0, 2 * sum(math.comb(n, i) for i in range(max(pos, neg), n + 1)) / 2 ** n)


def spearman(a, b):
    def rk(x):
        o = sorted(range(len(x)), key=lambda i: x[i]); r = [0.0] * len(x)
        for k, i in enumerate(o):
            r[i] = k
        return r
    ra, rb = rk(a), rk(b); ma, mb = stx.mean(ra), stx.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb)); den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else float("nan")


def lay(u):
    return int(u.split(":")[0][1:])


def main():
    R = [json.loads(l) for l in (ROOT / "results/cross_position.jsonl").read_text(encoding="utf-8").splitlines()]
    pp = []                                        # pair-prompts
    for r in R:
        for p in r["prompts"]:
            for q in p["pairs"]:
                q["concept"] = r["concept"]; pp.append(q)
    per_pair = defaultdict(list)
    attr_x, attr_y = [], []
    rec_top, rec_rnd, top_share = [], [], []
    obs, exp_, conc = Counter(), defaultdict(float), defaultdict(set)
    gate = sum(q["gate_maxdz"] > 0 for q in pp)
    for q in pp:
        s0 = q["s0"]
        if abs(s0) < 1e-3:
            continue
        sg = 1.0 if s0 > 0 else -1.0; a = abs(s0)
        e = sg * (s0 - q["s_early"]) / a; an = sg * (s0 - q["s_ans"]) / a; nu = sg * (s0 - q["s_null"]) / a
        per_pair[(q["concept"], q["U"], q["S"])].append((e, an, nu))
        attr_x.append(-sg * q["attr_sum"]); attr_y.append(sg * (s0 - q["s_early"]))
        b0 = q["s0_batch2"]; d2 = sg * (b0 - q["s_early_batch2"])
        if d2 >= 0.05 * abs(b0):
            rec_top.append(sg * (b0 - q["s_patch_top3"]) / d2); rec_rnd.append(sg * (b0 - q["s_patch_rnd3"]) / d2)
        Lu = lay(q["U"])
        dc = [(-sg * x, Lu + 1 + i, h) for i, row in enumerate(q["head_attr"]) for h, x in enumerate(row)]
        tot = sum(x for x, _, _ in dc)
        if tot >= 0.01 * a:
            top_share.append(sum(x for x, _, _ in sorted(dc, reverse=True)[:3]) / tot)
            for x, L, h in sorted(dc, reverse=True)[:3]:
                obs[(L, h)] += 1; conc[(L, h)].add(q["concept"])
            nh = len(dc)
            for _, L, h in dc:
                exp_[(L, h)] += 3 / nh
    md = stx.median
    pairs = {k: (stx.mean(x[0] for x in v), stx.mean(x[1] for x in v), stx.mean(x[2] for x in v)) for k, v in per_pair.items()}
    E = [v[0] for v in pairs.values()]; A = [v[1] for v in pairs.values()]; N = [v[2] for v in pairs.values()]
    w = sum(e > n for e, _, n in pairs.values()); l_ = sum(e < n for e, _, n in pairs.values())
    share = [e / (e + an) for e, an, _ in pairs.values() if e > 0 and an > 0]
    L = [f"Cross-position path: {len(R)} concepts, {len(pairs)} U -> S pairs, {len(pp)} pair-prompts; gate (heads at or before L_U changed): {gate}",
         "relative drop of the 'say' neuron S's activation at the answer word (S's own sign; median over pairs, each the mean over prompts):",
         f"  U zeroed at every EARLIER word: {md(E):+.4f} (mean {stx.mean(E):+.4f})",
         f"  U zeroed at the ANSWER word only: {md(A):+.4f} (mean {stx.mean(A):+.4f})",
         f"  activation-matched null at every earlier word: {md(N):+.4f} (mean {stx.mean(N):+.4f})",
         f"  earlier > null in {w} pairs, < in {l_} (sign p {sign_p(w, l_):.2g})",
         f"  earlier-word share of (earlier + answer-word) drop, pairs where both > 0 ({len(share)}): median {md(share) if share else float('nan'):.2f}"]
    c1 = md(E) >= 3 * max(md(N), 1e-9) and md(E) > 0 and sign_p(w, l_) < 0.05 and w > l_
    L.append(f"(c1) {'HOLDS' if c1 else 'FAILS'}: earlier-word drop >= 3x null and sign test")
    L.append(f"\nattribution check: Spearman(attribution total, actual earlier-word drop) over pair-prompts = {spearman(attr_x, attr_y):.2f} (n={len(attr_x)})")
    L.append(f"(c2) real patching, pair-prompts with an in-batch drop >= 5% of |S| ({len(rec_top)}): recovered by the top-3 heads median "
             f"{md(rec_top) if rec_top else float('nan'):.2f}, by 3 random heads {md(rec_rnd) if rec_rnd else float('nan'):.2f}")
    L.append(f"     attribution: top-3 heads' share of the summed head attribution, median {md(top_share) if top_share else float('nan'):.2f} "
             f"(pair-prompts with head total >= 1% of |S|: {len(top_share)})")
    if rec_top:
        c2 = md(rec_top) >= 0.5 and md(rec_rnd) <= 0.1
        L.append(f"(c2) {'HOLDS' if c2 else 'FAILS'} on real patching (>= 0.5 top-3, <= 0.1 random)")
    L.append(f"\n(c3) heads most often among the top-3 carriers (observed vs uniform expectation; distinct concepts):")
    ranked = sorted(obs, key=lambda k: -obs[k])
    for k in ranked[:20]:
        L.append(f"  L{k[0]}H{k[1]:<3} {obs[k]:>5} vs {exp_[k]:7.1f} expected (x{obs[k] / exp_[k]:.1f}); {len(conc[k])} concepts")
    tot_obs = sum(obs.values()); top10 = sum(obs[k] for k in ranked[:10])
    L.append(f"  the 10 most frequent heads take {top10 / tot_obs:.0%} of all top-3 slots ({tot_obs}); {len(obs)} distinct heads ever appear")
    txt = "\n".join(L)
    (ROOT / "results/cross_position_summary.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
