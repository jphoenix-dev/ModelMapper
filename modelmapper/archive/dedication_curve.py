"""How does 'dedicated' depend on how many concepts we compare against? Subsampling curve (no model, no API).

James's point: 'a neuron fires for no other concept' is only as strong as the population it was tested on. Using every
cross_activation_*.json (shared neuron union) and rank_save_*.json: for each concept's top-10 neurons (|own act| >= 1,
excluding the 'always on' neurons defined ONCE on the full set: fire for >= 90% of all other concepts), draw N other
concepts at random (20 repeats per N) and record the share of neurons that fire (>= 0.5 x own activation, answer
position) for none of them. Curves for the hand-picked concepts and for all concepts. Also each neuron's firing rate
across all others (how many concepts it serves), to see whether the curve is flattening (a truly dedicated core) or
still falling (dedication = sample-size artefact).
Usage: python src/dedication_curve.py  -> results/dedication_curve.txt
"""
import glob
import json
import random
import statistics as stx
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOP, FIRE, MIN_OWN, REPEATS = 10, 0.5, 1.0, 20


def load():
    xs = {}; units = None
    for f in sorted(glob.glob(str(ROOT / "results/cross_activation_*.json")), key=lambda f: ("_fame" in f, f)):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        assert units is None or d["units"] == units, f"{f}: different neuron union"
        units = d["units"]
        for c, v in d["concepts"].items():
            if "_fame" in Path(f).name and c in xs:
                continue
            xs[c] = v
    ranks, hand = {}, set()
    for f in sorted(glob.glob(str(ROOT / "results/concepts_*/rank_save_*.json")), key=lambda f: ("concepts_fame" in f, f)):
        if "rank_save_test" in f or f.endswith("_exact.json"):
            continue
        is_fame = "concepts_fame" in f
        for c, v in json.loads(Path(f).read_text(encoding="utf-8")).items():
            if "top" not in v or c not in xs or (is_fame and c in ranks):
                continue
            ranks[c] = v
            if not is_fame:
                hand.add(c)
    return xs, {u: i for i, u in enumerate(units)}, ranks, hand


def main():
    xs, ui, ranks, hand = load()
    concepts = list(ranks); n_all = len(concepts) - 1
    rows = []                                     # (owner, set of other concepts it fires for)
    for a in concepts:
        for e in ranks[a]["top"][:TOP]:
            own = e["act_answer_pos"]
            if abs(own) < MIN_OWN:
                continue
            i = ui[e["neuron"]]; sg = 1.0 if own >= 0 else -1.0
            fires = {b for b in concepts if b != a and xs[b]["ans"][i] * sg / abs(own) >= FIRE}
            if len(fires) >= 0.9 * n_all:
                continue                          # always on (defined once, on the full set)
            rows.append((a, fires))
    rng = random.Random(5)
    Ns = [n for n in (10, 25, 50, 100, 150, 200, 250, 300, 400, 500, 700, 1000) if n <= n_all] + [n_all]
    L = [f"Dedication vs population size: {len(concepts)} concepts ({len(hand)} hand-picked), {len(rows)} neuron-owner pairs "
         f"(always-on excluded once, on the full set). 'Dedicated' = fires for none of N randomly drawn other concepts "
         f"(mean +- sd over {REPEATS} draws).\n", f"{'N others':>9} | {'hand-picked owners':>20} | {'all owners':>12}"]
    curve = {}
    for N in sorted(set(Ns)):
        res = {"hand": [], "all": []}
        for _ in range(REPEATS):
            for grp in ("hand", "all"):
                rr = [(a, f) for a, f in rows if grp == "all" or a in hand]
                ded = 0
                for a, f in rr:
                    pool = [c for c in concepts if c != a]
                    sample = set(rng.sample(pool, N)) if N < len(pool) else set(pool)
                    ded += not (f & sample)
                res[grp].append(ded / len(rr))
        curve[N] = {g: (stx.mean(v), stx.pstdev(v)) for g, v in res.items()}
        L.append(f"{N:>9} | {curve[N]['hand'][0]:>13.1%} +- {curve[N]['hand'][1]:.1%} | {curve[N]['all'][0]:>6.1%} +- {curve[N]['all'][1]:.1%}")
    k = [len(f) for _, f in rows]
    L.append(f"\nhow many of the other {n_all} concepts each neuron fires for: 0 in {sum(x == 0 for x in k)}, 1 in {sum(x == 1 for x in k)}, "
             f"2-4 in {sum(2 <= x <= 4 for x in k)}, 5-19 in {sum(5 <= x <= 19 for x in k)}, 20+ in {sum(x >= 20 for x in k)} (of {len(k)})")
    Nh = sorted(curve)
    L.append("loss of dedicated share per doubling of N (hand-picked owners): " + ", ".join(
        f"{Nh[j]}->{Nh[j + 1]}: {(curve[Nh[j]]['hand'][0] - curve[Nh[j + 1]]['hand'][0]) / max(1e-9, __import__('math').log2(Nh[j + 1] / Nh[j])):.1%}"
        for j in range(len(Nh) - 1)))
    txt = "\n".join(L)
    (ROOT / "results/dedication_curve.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
