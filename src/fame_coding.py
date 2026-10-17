"""Do obscure concepts get private neurons too? Dedication of concept neurons by fame tier (no model, no API).

Uses every cross_activation_*.json (all sets, one shared neuron union) and every rank_save_*.json. For each concept's
top-10 neurons with |own activation| >= 1 at the answer position, excluding 'always on' neurons (fire for >= 90% of the
other concepts, as in cross_activation_summary.py): how many OTHER concepts (of all, any set) it fires for at >= 0.5 of
its own activation. Reported per fame tier (concepts_fame, by number of Wikipedia language editions) and for the
original hand-picked sets. Also the share of each concept's top-10 that is 'always on'.
Usage: python src/fame_coding.py  -> results/fame_coding.txt
"""
import glob
import json
import statistics as stx
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOP, FIRE, MIN_OWN = 10, 0.5, 1.0


def main():
    xs = {}; units = None
    for f in sorted(glob.glob(str(ROOT / "results/cross_activation_*.json")), key=lambda f: ("_fame" in f, f)):   # fame files last
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        assert units is None or d["units"] == units, f"{f}: different neuron union"
        units = d["units"]
        for c, v in d["concepts"].items():                  # fame-sample duplicates of earlier concepts are skipped (the
            if "_fame" in Path(f).name and c in xs:          # earlier entry is kept); fame files sort after cr2/cr3
                continue
            xs[c] = v
    ui = {u: i for i, u in enumerate(units)}
    ranks, group = {}, {}
    fame_tier = {}
    for wf in sorted(glob.glob(str(ROOT / "results/concepts_fame*/wave_free.json"))):
        for c, r in json.loads(Path(wf).read_text(encoding="utf-8")).items():
            fame_tier.setdefault(c, r["category"].split("#")[0] + (" (coverage draw)" if "fame2" in wf else ""))
    for f in sorted(glob.glob(str(ROOT / "results/concepts_*/rank_save_*.json")), key=lambda f: ("concepts_fame" in f, f)):
        if "rank_save_test" in f or f.endswith("_exact.json"):
            continue
        is_fame = "concepts_fame" in f
        for c, v in json.loads(Path(f).read_text(encoding="utf-8")).items():
            if "top" not in v or c not in xs or (is_fame and c in ranks):      # originals win over fame duplicates
                continue
            ranks[c] = v; group[c] = fame_tier[c] if is_fame else "original (hand-picked)"
    concepts = list(ranks); n_others = len(concepts) - 1
    stats = defaultdict(lambda: {"pairs": 0, "zero": 0, "n_other": [], "always": 0, "slots": 0, "concepts": 0})
    for a in concepts:
        g = stats[group[a]]; g["concepts"] += 1
        for e in ranks[a]["top"][:TOP]:
            g["slots"] += 1
            own = e["act_answer_pos"]
            if abs(own) < MIN_OWN:
                continue
            i = ui[e["neuron"]]; sg = 1.0 if own >= 0 else -1.0
            n = sum(xs[b]["ans"][i] * sg / abs(own) >= FIRE for b in concepts if b != a)
            if n >= 0.9 * n_others:
                g["always"] += 1; continue
            g["pairs"] += 1; g["zero"] += n == 0; g["n_other"].append(n)
    L = [f"Dedication of concept neurons by fame ({len(concepts)} recalled concepts in all; 'others' = the other {n_others}).\n",
         f"{'group':<24} {'concepts':>8} | {'neurons':>7} {'dedicated':>10} {'median others':>14} | always-on share of top-10 slots"]
    for g in sorted(stats, key=lambda k: (k.startswith("original"), k)):
        s = stats[g]
        L.append(f"{g:<24} {s['concepts']:>8} | {s['pairs']:>7} {s['zero'] / max(1, s['pairs']):>10.0%} "
                 f"{stx.median(s['n_other']) if s['n_other'] else float('nan'):>14} | {s['always'] / max(1, s['slots']):.0%}")
    txt = "\n".join(L)
    (ROOT / "results/fame_coding.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
