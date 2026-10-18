"""Summarise the step-2 causal sweeps (route_cut_sweep_*.json, chain_cut_sweep_*.json). No model, no API.

Rules (stated before reading the sweep results):
  informative  switching off the neuron itself (route test: neuron_off) / the source neuron (chain test: s_off) drops
               the answer's log-odds by >= 0.2 - otherwise the test cannot show anything and is reported separately
  causal       the targeted cut drops recall by >= 0.1 AND by more than 2x its control (random route / matched feeder)
  share        targeted-cut drop as a fraction of the neuron-off (route) or source-off (chain) drop
Usage: python src/causal_sweep_summary.py   -> results/causal_sweep_summary.txt
"""
import json
import statistics as stx
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = {"route": [ROOT / "results/concepts_v3/route_cut_sweep_cr2.json", ROOT / "results/concepts_w2/route_cut_sweep_w2.json"],
         "chain": [ROOT / "results/concepts_v3/chain_cut_sweep_cr2.json", ROOT / "results/concepts_w2/chain_cut_sweep_w2.json"]}


def load(kind):
    d = {}
    for f in FILES[kind]:
        if f.exists():
            d.update(json.loads(f.read_text(encoding="utf-8")))
    return d


def main():
    L = []
    r = load("route")
    rows = [(c, v["neuron"], v["route_cut"]["logodds_drop"], v["route_rand"]["logodds_drop"], v["neuron_off"]["logodds_drop"],
             v["heads_off"]["logodds_drop"], v["base"]["act"], v["route_cut"]["act"]) for c, v in r.items()]
    inf = [x for x in rows if x[4] >= 0.2]
    cz = [x for x in inf if x[2] >= 0.1 and x[2] > 2 * max(x[3], 0)]
    L.append(f"HEAD -> NEURON ROUTES: {len(rows)} concepts tested; informative (neuron off drops recall >= 0.2): {len(inf)}")
    if inf:
        L.append(f"  causal (cut drop >= 0.1 and > 2x random route): {len(cz)}/{len(inf)}")
        L.append(f"  median recall drop: route cut {stx.median(x[2] for x in inf):+.2f} vs random route {stx.median(x[3] for x in inf):+.2f}; "
                 f"neuron off {stx.median(x[4] for x in inf):+.2f}; the 5 heads fully off {stx.median(x[5] for x in inf):+.2f}")
        L.append(f"  route cut as share of neuron-off: median {stx.median(x[2] / x[4] for x in inf):.2f}; "
                 f"neuron activation kept after route cut: median {stx.median(x[7] / x[6] for x in inf if x[6]):.2f}")
        L.append(f"  random route larger than route cut in {sum(x[3] >= x[2] for x in inf)}/{len(inf)}")
    L.append("  not informative: " + ", ".join(f"{x[0]} ({x[1]}, neuron off {x[4]:+.2f})" for x in rows if x[4] < 0.2))
    L.append("  per concept: neuron | route cut | random | neuron off | heads off")
    for x in sorted(rows, key=lambda x: -x[2]):
        L.append(f"    {x[0]:<30} {x[1]:<10} {x[2]:+.2f} | {x[3]:+.2f} | {x[4]:+.2f} | {x[5]:+.2f}")
    ch = load("chain")
    rows = [(k, v["chain_cut"]["recall_drop"], v["matched"]["recall_drop"], v["s_off"]["recall_drop"], v["base"]["t_act"],
             v["chain_cut"]["t_act"], v["s_off"]["t_act"]) for k, v in ch.items()]
    inf = [x for x in rows if x[3] >= 0.2]
    cz = [x for x in inf if x[1] >= 0.1 and x[1] > 2 * max(x[2], 0)]
    L.append(f"\nNEURON -> NEURON CHAIN LINKS: {len(rows)} links tested; informative (source off drops recall >= 0.2): {len(inf)}")
    if inf:
        L.append(f"  causal (cut drop >= 0.1 and > 2x matched non-concept feeder): {len(cz)}/{len(inf)}")
        L.append(f"  median recall drop: chain cut {stx.median(x[1] for x in inf):+.2f} vs matched feeder {stx.median(x[2] for x in inf):+.2f}; "
                 f"source fully off {stx.median(x[3] for x in inf):+.2f}")
        L.append(f"  chain cut as share of source-off: median {stx.median(x[1] / x[3] for x in inf):.2f}; target activation kept after "
                 f"chain cut: median {stx.median(x[5] / x[4] for x in inf if x[4]):.2f}")
    L.append("  per link: chain cut | matched | source off | target act base -> cut")
    for x in sorted(rows, key=lambda x: -x[1]):
        L.append(f"    {x[0]:<52} {x[1]:+.2f} | {x[2]:+.2f} | {x[3]:+.2f} | {x[4]:.1f} -> {x[5]:.1f}")
    txt = "\n".join(L)
    (ROOT / "results/causal_sweep_summary.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
