"""Summarise feeders_*.json: heads vs earlier neurons as inputs to concept neurons; within-concept neuron chains;
overlap of feeding heads with the knockout-important heads (head_logodds) and with the shared heads (head_summary).

Usage: python src/feeders_summary.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = [("results/concepts_v3/feeders_v3.json", "results/concepts_v3", "results/concepts_v3/head_logodds_cr2.json"),
       ("results/concepts_w2/feeders_w2.json", "results/concepts_w2", "results/concepts_w2/head_logodds_w2.json")]


def main():
    shared = set(json.loads((ROOT / "results/head_summary.json").read_text(encoding="utf-8"))["shared"])
    import concept_pipeline as cp
    for fp, cdir, hp in SRC:
        if not (ROOT / fp).exists():
            continue
        fd = json.loads((ROOT / fp).read_text(encoding="utf-8")); hd = json.loads((ROOT / hp).read_text(encoding="utf-8"))
        for c, v in fd.items():
            x = json.loads((ROOT / cdir / f"{cp.slug(c)}__clamp_reply.json").read_text(encoding="utf-8"))
            own = {f"L{t['layer']}:{t['neuron']}" for t in x["top_neurons"][:10]}
            imp = {f"L{l}H{h}" for l, h in hd.get(c, {}).get("top10_heads", [])}
            print(f"\n== {c} (reconstruction error <= {v['max_recon_err']})")
            for k, r in v["neurons"].items():
                if abs(r["act_signed"]) < 0.5:
                    print(f"  {k:<10} weak (act {r['act_signed']:+.3f}), skipped"); continue
                tot = abs(r["from_heads"]) + abs(r["from_mlp_neurons"]) + abs(r["from_embedding"])
                chain = sum(val for n, val in r["top_neurons"] if n in own)
                fh = [h for h, _ in r["top_heads"][:8]]
                lay = int(k[1:].split(":")[0])
                same_layer = sum(int(h[1:].split("H")[0]) == lay for h in fh)
                print(f"  {k:<10} act {r['act_signed']:+7.2f} | heads {abs(r['from_heads'])/tot:.0%} / earlier neurons "
                      f"{abs(r['from_mlp_neurons'])/tot:.0%} | from the concept's own top-10 neurons {chain:+.2f} "
                      f"({chain / r['from_mlp_neurons']:.0%} of the neuron input) | top-8 feeding heads: {same_layer} same-layer, "
                      f"{sum(h in imp for h in fh)} knockout-important, {sum(h in shared for h in fh)} shared")
                print(f"             feeding heads {r['top_heads'][:5]}")
                print(f"             feeding neurons {r['top_neurons'][:4]}")


if __name__ == "__main__":
    main()
