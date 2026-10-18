"""Per-concept circuit maps and what is shared across concepts ("fog of war" step; no model, no API).

Joins the attention-study outputs for every concept into one map:
  reader heads     heads pushing UP the concept's top-3 neurons (feeders_all: top_heads with value > 0)
  brake heads      heads pushing them DOWN (value < 0)
  concept neurons  the concept's top-3 neurons (by recall attribution)
  internal links   feeders of those neurons that are themselves among the concept's top-10 (value > 0)
  writer neurons   top-10 direct writers of the answer word (answer_readout), labelled content / format / mixed
                   (writer_specificity); anti-writer neurons = negative direct contribution
  writer heads     top-5 heads by direct contribution to the answer (+ = writer, - = anti-writer)
Then per role: how many concepts each component appears in. A component in >= 25% of concepts is called SHARED.
Usage: python src/circuit_maps.py   -> results/circuit_maps.json, results/circuit_maps_summary.txt
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETS = [("results/concepts_v3", "cr2"), ("results/concepts_w2", "w2")]


def jl(p):
    return json.loads((ROOT / p).read_text(encoding="utf-8"))


def main():
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    import concept_pipeline as cp
    spec = jl("results/writer_specificity.json")
    maps = {}
    for d, tag in SETS:
        fd = jl(f"{d}/feeders_all_{tag}.json"); ar = jl(f"{d}/answer_readout_{tag}.json")
        for c, v in fd.items():
            x = jl(f"{d}/{cp.slug(c)}__clamp_reply.json")
            own10 = {f"L{q['layer']}:{q['neuron']}" for q in x["top_neurons"][:10]}
            mp = {"concept_neurons": [], "reader_heads": set(), "brake_heads": set(), "internal_links": [],
                  "writer_neurons": {"content": [], "format": [], "mixed": [], "anti": []}, "writer_heads": [], "anti_writer_heads": []}
            for k, r in v["neurons"].items():
                mp["concept_neurons"].append({"neuron": k, "act": r["act_signed"], "heads_share": r["heads_share_of_abs"]})
                for h, val in r["top_heads"]:
                    (mp["reader_heads"] if val > 0 else mp["brake_heads"]).add(h)
                for n, val in r["top_neurons"]:
                    if n in own10 and val > 0:
                        mp["internal_links"].append(f"{n}>{k}")
            for e in spec.get(c, []):
                mp["writer_neurons"]["anti" if e["contrib"] < 0 else e["label"]].append(e["neuron"])
            for h, val in ar[c]["top_heads"][:5]:
                (mp["writer_heads"] if val > 0 else mp["anti_writer_heads"]).append(h)
            mp["reader_heads"] = sorted(mp["reader_heads"]); mp["brake_heads"] = sorted(mp["brake_heads"])
            mp["answer_split"] = {"heads": ar[c]["from_heads"], "neurons": ar[c]["from_neurons"], "own_direct": ar[c]["own_concept_neurons_direct"],
                                  "neurons_for_half": ar[c].get("neurons_for_half")}
            maps[c] = mp
    n = len(maps)
    roles = {
        "reader heads (feed concept neurons up)": lambda m: m["reader_heads"],
        "brake heads (push concept neurons down)": lambda m: m["brake_heads"],
        "concept neurons (top-3)": lambda m: [e["neuron"] for e in m["concept_neurons"]],
        "internal-link sources": lambda m: sorted({l.split(">")[0] for l in m["internal_links"]}),
        "content writer neurons": lambda m: m["writer_neurons"]["content"],
        "format writer neurons": lambda m: m["writer_neurons"]["format"],
        "anti-writer neurons": lambda m: m["writer_neurons"]["anti"],
        "writer heads (+)": lambda m: m["writer_heads"],
        "anti-writer heads (-)": lambda m: m["anti_writer_heads"],
    }
    L = [f"Circuit maps for {n} concepts. SHARED = component in >= 25% of concepts (>= {n / 4:.0f}).\n",
         f"{'role':<42} {'appearances':>11} {'distinct':>8} {'shared':>6} {'from shared':>11}  shared components (concepts)"]
    summary = {}
    for role, f in roles.items():
        cnt = Counter(); app = 0
        for m in maps.values():
            items = set(f(m)); app += len(items); cnt.update(items)
        shared = [(k, v) for k, v in cnt.most_common() if v >= n / 4]
        fs = sum(v for _, v in shared) / app if app else 0
        summary[role] = {"appearances": app, "distinct": len(cnt), "shared": shared, "share_from_shared": round(fs, 2),
                         "top": cnt.most_common(8)}
        L.append(f"{role:<42} {app:>11} {len(cnt):>8} {len(shared):>6} {fs:>11.0%}  " + ", ".join(f"{k} ({v})" for k, v in shared[:8]))
    L.append("\nmost frequent components per role, even if not shared (concepts):")
    for role, s in summary.items():
        L.append(f"  {role:<42} " + ", ".join(f"{k} ({v})" for k, v in s["top"]))
    L.append("\nlayers of concept neurons (top-3): " + str(sorted(Counter(int(e["neuron"][1:].split(":")[0]) for m in maps.values() for e in m["concept_neurons"]).items())))
    txt = "\n".join(L)
    (ROOT / "results/circuit_maps.json").write_text(json.dumps({"maps": maps, "summary": summary}, indent=1, ensure_ascii=False), encoding="utf-8")
    (ROOT / "results/circuit_maps_summary.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
