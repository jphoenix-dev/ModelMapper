"""Attention study, steps 1-2 summary: heads vs neurons, and shared vs concept-specific heads.

Inputs: results/concepts_v3/head_logodds_cr2.json, results/concepts_w2/head_logodds_w2.json, and the neuron log-odds v2
files for the same concepts (same prompts and answers). Usage: python src/head_summary.py
"""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAIRS = [("results/concepts_v3/head_logodds_cr2.json", "results/concepts_v3/logodds_cr2_V3_v2.json"),
         ("results/concepts_w2/head_logodds_w2.json", "results/concepts_w2/logodds_w2_V3_v2.json")]


def le(k, c):
    return k is not None and k <= c


def main():
    heads, neur = {}, {}
    for hp, np_ in PAIRS:
        h = json.loads((ROOT / hp).read_text(encoding="utf-8")) if (ROOT / hp).exists() else {}
        n = json.loads((ROOT / np_).read_text(encoding="utf-8"))
        for c, v in h.items():
            heads[c] = v
            if c in n:
                neur[c] = n[c]
    cs = sorted(heads)
    N = len(cs)
    print(f"{N} concepts with head results\n")
    print(f"{'':<34} {'heads':>8} {'neurons':>8}")
    for name, kh, kn in (("k_mean <= 2", lambda v: le(v['k_mean'], 2), lambda v: le(v['k_LO3_measured'], 2)),
                         ("k_mean <= 5", lambda v: le(v['k_mean'], 5), lambda v: le(v['k_LO3_measured'], 5)),
                         ("k_mean <= 10", lambda v: le(v['k_mean'], 10), lambda v: le(v['k_LO3_measured'], 10)),
                         ("k_flip <= 5", lambda v: le(v['k_flip'], 5), lambda v: le(v.get('k_flip_half_prompts_measured'), 5)),
                         ("k_flip <= 50 (flips at all)", lambda v: le(v['k_flip'], 50), lambda v: le(v.get('k_flip_half_prompts_measured'), 50))):
        a = sum(kh(heads[c]) for c in cs); b = sum(kn(neur[c]) for c in cs if c in neur)
        print(f"{name:<34} {a:>4}/{N:<3} {b:>4}/{len(neur)}")
    d5 = [heads[c]["measured"][3]["mean_drop"] for c in cs]; r5 = [heads[c]["random"][3]["mean_drop"] for c in cs]
    r50 = [heads[c]["random"][-1]["mean_drop"] for c in cs]
    print(f"\nmean log-odds drop, top-5 heads {sum(d5)/N:.2f} | random 5 heads {sum(r5)/N:+.2f} | random 50 heads {sum(r50)/N:+.2f} "
          f"(min {min(r50):+.2f}, max {max(r50):+.2f})")
    nd5 = [neur[c]["drop_measured"][3] for c in cs if c in neur]
    print(f"mean log-odds drop, top-5 neurons {sum(nd5)/len(nd5):.2f} (same concepts)")
    # step 2: shared heads
    cnt = Counter(tuple(h) for c in cs for h in heads[c]["top10_heads"])
    print(f"\nheads appearing in the top-10 of many concepts (of {N}); 1,152 heads exist, so by chance a head would appear in "
          f"~{10 / 1152 * N:.2f} concepts:")
    for (l, hh), k in cnt.most_common(15):
        print(f"  L{l}H{hh:<3} in {k:>2} concepts ({k / N:.0%})")
    shared = {h for h, k in cnt.items() if k >= max(3, N // 5)}
    per = [sum(tuple(h) in shared for h in heads[c]["top10_heads"]) for c in cs]
    print(f"\n'shared' heads (in >= {max(3, N // 5)} concepts' top-10): {len(shared)}; on average {sum(per)/N:.1f} of each concept's "
          f"top-10 heads are shared ones")
    lay = Counter(l for c in cs for l, _ in heads[c]["top10_heads"][:5])
    print("layers of the top-5 heads (all concepts):", dict(sorted(lay.items())))
    (ROOT / "results/head_summary.json").write_text(json.dumps(
        {"n": N, "head_counts": {f"L{l}H{h}": k for (l, h), k in cnt.most_common()}, "shared": [f"L{l}H{h}" for l, h in sorted(shared)]},
        indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
