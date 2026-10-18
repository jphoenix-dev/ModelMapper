"""Pick, for every concept, one head->neuron route and one neuron->neuron chain link to test causally (step 2).

From feeders_all_*.json (top-3 neurons per concept) and the concept runs:
  route:  the active concept neuron (|act| >= 0.5) with the highest share of input from heads -> route_cut.py "<c>=L:n"
  chain:  the strongest active concept neuron T in layers >= 30, and its largest POSITIVE feeder S that is one of the
          concept's top-10 neurons and lies below T -> chain_cut.py "<c>=LS:s>LT:t"
Writes results/causal_sweep_<tag>_routes.txt and ..._chains.txt (one arg per line). No model needed.
Usage: python src/causal_sweep_args.py <feeders_all json> <concept dir> <tag>
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    fd = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")); cdir = Path(sys.argv[2]); tag = sys.argv[3]
    import concept_pipeline as cp
    routes, chains = [], []
    for c, v in fd.items():
        x = json.loads((cdir / f"{cp.slug(c)}__clamp_reply.json").read_text(encoding="utf-8"))
        own = {f"L{q['layer']}:{q['neuron']}" for q in x["top_neurons"][:10]}
        act = {k: r for k, r in v["neurons"].items() if abs(r["act_signed"]) >= 0.5}
        if act:
            k = max(act, key=lambda k: act[k]["heads_share_of_abs"])
            if any(val > 0 for _, val in act[k]["top_heads"]):
                routes.append(f"{c}={k}")
        late = {k: r for k, r in act.items() if int(k[1:].split(":")[0]) >= 30}
        if late:
            t = max(late, key=lambda k: late[k]["act_signed"]); lt = int(t[1:].split(":")[0])
            src = [(n, val) for n, val in late[t]["top_neurons"] if n in own and val > 0 and int(n[1:].split(":")[0]) < lt]
            if src:
                s = max(src, key=lambda kv: kv[1])[0]
                chains.append(f"{c}={s}>{t}")
    (ROOT / f"results/causal_sweep_{tag}_routes.txt").write_text("\n".join(routes) + "\n", encoding="utf-8")
    (ROOT / f"results/causal_sweep_{tag}_chains.txt").write_text("\n".join(chains) + "\n", encoding="utf-8")
    print(f"{tag}: {len(routes)} route tests, {len(chains)} chain tests (of {len(fd)} concepts)")


if __name__ == "__main__":
    main()
