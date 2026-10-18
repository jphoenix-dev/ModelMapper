"""Generality of the feeder picture across all concepts (feeders_all_*.json, top-3 neurons per concept).

Reports: head share of input by neuron layer band; share of neuron input coming from the concept's own top-10 neurons
(chain) by band; which heads most often appear among a concept neuron's top-5 POSITIVE feeders and top-3 NEGATIVE
feeders (vs the chance rate). Usage: python src/feeders_general.py
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = [("results/concepts_v3/feeders_all_cr2.json", "results/concepts_v3"), ("results/concepts_w2/feeders_all_w2.json", "results/concepts_w2")]


def main():
    import concept_pipeline as cp
    band = lambda l: "0-15" if l < 16 else "16-23" if l < 24 else "24-29" if l < 30 else "30-35"  # noqa: E731
    hs, ch = defaultdict(list), defaultdict(list)
    pos, neg = Counter(), Counter(); n_neur = 0; errs = []
    for fp, cdir in SRC:
        if not (ROOT / fp).exists():
            continue
        fd = json.loads((ROOT / fp).read_text(encoding="utf-8"))
        for c, v in fd.items():
            errs.append(v["max_recon_err"])
            x = json.loads((ROOT / cdir / f"{cp.slug(c)}__clamp_reply.json").read_text(encoding="utf-8"))
            own = {f"L{t['layer']}:{t['neuron']}" for t in x["top_neurons"][:10]}
            for k, r in v["neurons"].items():
                if abs(r["act_signed"]) < 0.5:
                    continue
                n_neur += 1
                l = int(k[1:].split(":")[0])
                tot = abs(r["from_heads"]) + abs(r["from_mlp_neurons"]) + abs(r["from_embedding"]) or 1e-9
                hs[band(l)].append(abs(r["from_heads"]) / tot)
                chain = sum(val for nn, val in r["top_neurons"] if nn in own)
                ch[band(l)].append(chain / r["from_mlp_neurons"] if abs(r["from_mlp_neurons"]) > 1e-6 else 0)
                for h, val in [hv for hv in r["top_heads"] if hv[1] > 0][:5]:
                    pos[h] += 1
                for h, val in sorted([hv for hv in r["top_heads"] if hv[1] < 0], key=lambda hv: hv[1])[:3]:
                    neg[h] += 1
    med = lambda xs: sorted(xs)[len(xs) // 2] if xs else float("nan")  # noqa: E731
    print(f"{n_neur} active concept neurons from {len(errs)} concepts; max reconstruction error {max(errs):.3f}\n")
    print(f"{'layer band':<10} {'n':>4} {'median head share':>18} {'median own-chain share of neuron input':>40}")
    for b in ("0-15", "16-23", "24-29", "30-35"):
        print(f"{b:<10} {len(hs[b]):>4} {med(hs[b]):>18.2f} {med(ch[b]):>40.2f}")
    print(f"\nmost frequent POSITIVE feeding heads (top-5 per neuron; chance ~{5 / 1152 * n_neur:.1f} neurons each):")
    for h, k in pos.most_common(12):
        print(f"  {h:<7} feeds {k:>3} concept neurons ({k / n_neur:.0%})")
    print(f"\nmost frequent NEGATIVE feeding heads (top-3 per neuron):")
    for h, k in neg.most_common(8):
        print(f"  {h:<7} pushes down {k:>3} concept neurons ({k / n_neur:.0%})")
    (ROOT / "results/feeders_general.json").write_text(json.dumps({"n": n_neur, "pos": pos.most_common(40), "neg": neg.most_common(40)}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
