"""Summarise answer_readout_*.json across concepts: who writes the answer word (attention study; no model, no API).

Per concept the answer's score (vs the mean word) splits exactly into heads + neurons + embedding. Reports medians of
shares by layer band, how much the concept's own top-10 neurons write DIRECTLY, how spread out the writing is, and
which heads / neurons recur as top writers across concepts. Families are a hand split (people = says_name.PEOPLE;
places = PLACES below; everything else = things & ideas) - only for comparing groups, not a claim about the model.
Usage: python src/answer_readout_summary.py <answer_readout json> [...]   -> results/answer_readout_summary.txt/.json
"""
import json
import statistics as stx
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PEOPLE = {"Albert Einstein", "William Shakespeare", "Napoleon Bonaparte", "Wolfgang Amadeus Mozart", "Frida Kahlo",
          "Harry Potter", "Marie Curie", "Charles Darwin"}            # same set as says_name.py
PLACES = {"Eiffel Tower", "Golden Gate Bridge", "Mount Everest", "Venice", "Tokyo", "the Amazon rainforest",
          "the Great Wall of China", "Egypt's pyramids"}


def fam(c):
    return "people" if c in PEOPLE else "places" if c in PLACES else "things & ideas"


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(stx.median(xs), 2) if xs else None


def main():
    rows = {}
    for f in sys.argv[1:]:
        rows.update(json.loads(Path(f).read_text(encoding="utf-8")))
    L = []
    w = lambda s: L.append(s)
    n = len(rows)
    w(f"Who writes the answer word? {n} concepts. Score = answer token's logit minus the mean logit, at the answer position,")
    w("split exactly into what each head, each neuron and the embedding add (final normalisation scale held fixed).\n")
    sh = lambda r, k: r[k] / r["score_vs_mean"]
    w(f"median score {med([r['score_vs_mean'] for r in rows.values()])}; worst reconstruction error "
      f"{max(r['max_recon_err'] for r in rows.values()):.3f}; worst score error {max(r['max_score_err'] for r in rows.values()):.3f}")
    w(f"share written by heads: median {med([sh(r, 'from_heads') for r in rows.values()])} "
      f"(range {min(sh(r, 'from_heads') for r in rows.values()):.2f}-{max(sh(r, 'from_heads') for r in rows.values()):.2f}); "
      f"by neurons: median {med([sh(r, 'from_neurons') for r in rows.values()])}")
    bands = sorted({b for r in rows.values() for b in r["by_band"]})
    w("\nmedian share of the score by layer band:")
    for b in bands:
        w(f"  {b:<16} {med([r['by_band'].get(b, 0) / r['score_vs_mean'] for r in rows.values()]):+.2f}")
    w("\nhow much the concept's OWN top-10 neurons (the ones whose switching-off flips recall) write DIRECTLY:")
    own = [sh(r, "own_concept_neurons_direct") for r in rows.values()]
    w(f"  median share {med(own)}; quartiles {[round(q, 2) for q in stx.quantiles(own, n=4)]}; "
      f"concepts where it is >25%: {sum(o > 0.25 for o in own)}/{n}; <5%: {sum(o < 0.05 for o in own)}/{n}")
    if all("neurons_for_half" in r for r in rows.values()):
        w("\nhow spread out the writing is (fewest top positive writers reaching that share of the NET total):")
        for k in ("neurons_for_half", "neurons_for_90pct", "heads_for_half", "heads_for_90pct"):
            v = [r[k] for r in rows.values()]
            w(f"  {k:<18} median {med(v)}  range {min(x for x in v if x is not None)}-{max(x for x in v if x is not None)}"
              f"  (never reached: {sum(x is None for x in v)})")
    w("\nby family (median shares):")
    for fm in ("people", "places", "things & ideas"):
        rr = [r for c, r in rows.items() if fam(c) == fm]
        if not rr:
            continue
        extra = f", neurons for half {med([r.get('neurons_for_half') for r in rr])}" if "neurons_for_half" in rr[0] else ""
        w(f"  {fm:<15} n={len(rr):<3} heads {med([sh(r, 'from_heads') for r in rr])}, own neurons direct "
          f"{med([sh(r, 'own_concept_neurons_direct') for r in rr])}{extra}")
    hc, hs, hpos = Counter(), defaultdict(list), Counter()
    for r in rows.values():
        for h, v in r["top_heads"][:5]:
            hc[h] += 1; hs[h].append(v); hpos[h] += v > 0
    w("\nheads most often among a concept's top-5 direct writers (count, how often positive, median contribution):")
    for h, c in hc.most_common(12):
        w(f"  {h:<8} {c:>3}/{n}  positive {hpos[h]}/{c}  median {med(hs[h]):+.2f}")
    nc, ns, nown = Counter(), defaultdict(list), Counter()
    for r in rows.values():
        for nn, v, isown in r["top_neurons"]:
            nc[nn] += 1; ns[nn].append(v); nown[nn] += isown
    w("\nneurons appearing among top-10 direct writers in 3+ concepts (generic writers?):")
    for nn, c in nc.most_common():
        if c < 3:
            break
        w(f"  {nn:<10} {c:>3}/{n}  median {med(ns[nn]):+.2f}  (counted as that concept's own neuron in {nown[nn]})")
    owntop = sum(any(isown for _, _, isown in r["top_neurons"][:3]) for r in rows.values())
    w(f"\nconcepts where one of the top-3 direct neuron writers is the concept's own neuron: {owntop}/{n}")
    w("\nper concept: score | heads | neurons | own direct | top head | top neuron (own?)")
    for c, r in rows.items():
        tn = r["top_neurons"][0]
        w(f"  {c:<28} {r['score_vs_mean']:+6.1f} | {sh(r, 'from_heads'):.2f} | {sh(r, 'from_neurons'):.2f} | "
          f"{sh(r, 'own_concept_neurons_direct'):+.2f} | {r['top_heads'][0][0]:<7} | {tn[0]} {tn[1]:+.1f}{' own' if tn[2] else ''}"
          + (f" | half by {r['neurons_for_half']} neurons" if "neurons_for_half" in r else ""))
    txt = "\n".join(L)
    (ROOT / "results/answer_readout_summary.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
