"""PREREG_MAP P2: out-of-sample check of the dedication curve. (no model, no API)

Prediction (written 17 Oct morning, before coverage-3 data): for the 178 hand-picked concepts' neurons (same rules as
dedication_curve.py: |own act| >= 1, 'always on' excluded once on the full set), the share firing (>= 0.5x own
activation, answer position) for NONE of the N other concepts follows dedicated(N) = 36.5% - 9.3 x log2(N/358);
'holds' = observed within +-3 points, at the N reached now (all concepts except the owner).
Note: 'always on' is re-defined on the current full set (>= 90% of others), exactly as dedication_curve.py does.
Usage: python src/p2_check.py  -> results/p2_check.txt
"""
import math
from pathlib import Path

from dedication_curve import FIRE, MIN_OWN, TOP, load

ROOT = Path(__file__).resolve().parents[1]


def main():
    xs, ui, ranks, hand = load()
    concepts = list(ranks); n_all = len(concepts) - 1
    ded = tot = 0
    for a in concepts:
        if a not in hand:
            continue
        for e in ranks[a]["top"][:TOP]:
            own = e["act_answer_pos"]
            if abs(own) < MIN_OWN:
                continue
            i = ui[e["neuron"]]; sg = 1.0 if own >= 0 else -1.0
            k = sum(1 for b in concepts if b != a and xs[b]["ans"][i] * sg / abs(own) >= FIRE)
            if k >= 0.9 * n_all:
                continue
            tot += 1; ded += k == 0
    obs = ded / tot; pred = 0.365 - 0.093 * math.log2(n_all / 358)
    ok = abs(obs - pred) <= 0.03
    txt = (f"P2 dedication curve, out of sample: N = {n_all} other concepts ({len(concepts)} concepts, {len(hand)} hand-picked owners)\n"
           f"  observed dedicated share {obs:.1%} ({ded}/{tot}) vs predicted {pred:.1%} -> difference {100 * (obs - pred):+.1f} points\n"
           f"  PREDICTION {'HOLDS' if ok else 'FAILS'} (needs within +-3 points)")
    (ROOT / "results/p2_check.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
