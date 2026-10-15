"""Leniency control that is mixed at the level of the evidence: "spliced" dossiers (ISSUES #25).

Planted activation-mixtures turned out weak: one component's firings usually dominate the top-k, so the dossier looks
like one job (Luna called 41/60 "ONE"). Here, a spliced unit = 10 top examples of neuron A + 10 of an unrelated
neuron B (different layer band, random), shuffled together. That is a two-job dossier by construction.

Each spliced unit goes through exactly the census pipeline: Luna explains + verdict (label_batch settings), then the
rubric re-grader (soft coverage on its 20 examples, random-snippet fit). Output: the same JSON shape as label_batch /
regrade, so calibrate_clean.py can use spliced units as the known-mixed reference.

Usage: python src/spliced_control.py <census.json> <out_prefix> [n=100]
"""
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import oai
from atlas import Atlas
from label_batch import SYSTEM, clip

ROOT = Path(__file__).resolve().parents[1]


def main():
    census = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["units"]
    prefix = sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    at = Atlas("Qwen/Qwen3-8B")
    rng = random.Random(25)
    live = [k for k, v in census.items() if (v.get("pos") or {}).get("explanation") and abs(at.a["top_v"][v["layer"], v["neuron"], 0].item()) >= 0.05]
    units = {}
    for i in range(n):
        a, b = rng.sample(live, 2)
        while abs(census[a]["layer"] - census[b]["layer"]) < 6:      # unrelated: different depth band
            b = rng.choice(live)
        exs = {}
        for pole in ("pos", "neg"):
            ea = at.examples(census[a]["layer"], census[a]["neuron"], pole, n=10)
            eb = at.examples(census[b]["layer"], census[b]["neuron"], pole, n=10)
            # each example keeps strength relative to its own source's peak, so neither half looks weaker
            pa = max(abs(ea[0]["value"]), 1e-9) if ea else 1; pb = max(abs(eb[0]["value"]), 1e-9) if eb else 1
            mix = [dict(x, rel=abs(x["value"]) / pa) for x in ea] + [dict(x, rel=abs(x["value"]) / pb) for x in eb]
            rng.shuffle(mix)
            mix.sort(key=lambda x: -x["rel"])
            exs[pole] = mix
        units[f"splice_{a}_{b}"] = exs
    # write a scan-format file so label_batch / regrade can read it with --scan
    scan = {"model": "Qwen/Qwen3-8B", "units": {u: {"pos": [{"value": x["rel"], "text": x["text"]} for x in e["pos"]],
                                                    "neg": [{"value": -x["rel"], "text": x["text"]} for x in e["neg"]]}
                                                for u, e in units.items()}}
    out = ROOT / f"results/{prefix}_scan.json"
    out.write_text(json.dumps(scan, ensure_ascii=False), encoding="utf-8")
    print(f"{len(units)} spliced units -> {out}")


if __name__ == "__main__":
    main()
