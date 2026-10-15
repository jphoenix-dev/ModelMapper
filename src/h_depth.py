"""H-DEPTH (pre-registered in PREREG.md): are concept-specific neurons concentrated at particular depths?

1. Collect each recalled concept's top-N attributed neurons from the concept-pipeline results.
2. Label every unique neuron blind with label_batch.py (default census settings, for comparability).
3. Jev classifies each (concept, neuron) pair as CONCEPT_SPECIFIC / GENERIC / UNRELATED from the neuron's blind
   explanations and output words (option order rotated to cancel position bias).
4. Report each class's depth distribution against the depth distribution of the whole top-N set (the baseline the
   pre-registration specifies, since attribution itself favours some layers), and which generic neurons recur.

Usage: python src/h_depth.py label    [N=50] [tag]  -> results/hdepth_units.json, then label_batch -> hdepth_labels<tag>.json
       python src/h_depth.py classify [N=50] [tag]  -> results/hdepth_classes<tag>.json + printed analysis
       (tag, e.g. "_rep2", makes an independent repeat for the noise check)
"""
import json
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
CLASSES = {
    "CONCEPT_SPECIFIC": "About the concept itself or one of its distinctive parts or attributes "
                        "(e.g. for the Golden Gate Bridge: gold, bridges, San Francisco, landmarks).",
    "GENERIC": "General-purpose machinery not about any topic: grammar, punctuation, formatting, spelling or "
               "word-piece completion, completing proper names in general, answer formatting.",
    "UNRELATED": "About some other specific topic that has nothing to do with the concept.",
}


def concepts(N):
    out = {}
    for f in sorted(R.glob("concepts/*__clamp_reply.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        if isinstance(r, dict) and r.get("status") == "ok":
            out[r["concept"]] = [(t["layer"], t["neuron"]) for t in r["top_neurons"][:N]]
    return out


def main():
    mode = sys.argv[1]
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    tag = sys.argv[3] if len(sys.argv) > 3 else ""          # e.g. "_rep2" for an independent repeat (noise check)
    cs = concepts(N)
    units = sorted({u for us in cs.values() for u in us})
    if mode == "label":
        (R / "hdepth_units.json").write_text(json.dumps([{"layer": l, "neuron": n} for l, n in units]), encoding="utf-8")
        print(f"{len(cs)} concepts, {sum(map(len, cs.values()))} (concept, neuron) pairs, {len(units)} unique neurons")
        subprocess.run([sys.executable, "-u", str(ROOT / "src/label_batch.py"), "Qwen/Qwen3-8B",
                        str(R / "hdepth_units.json"), str(R / f"hdepth_labels{tag}.json")], check=True)
        return

    import jev
    from transformers import AutoTokenizer  # noqa: F401  (keeps import order consistent with other scripts)
    labels = json.loads((R / f"hdepth_labels{tag}.json").read_text(encoding="utf-8"))["units"]
    # output words for each neuron: reuse the ones the concept pipeline recorded where available
    writes = {}
    for f in R.glob("concepts/*__clamp*.json"):
        r = json.loads(f.read_text(encoding="utf-8"))
        if isinstance(r, dict):
            for k, v in (r.get("labels") or {}).items():
                writes[k] = v.get("writes_up", [])
    keys = list(CLASSES)

    def classify(pair):
        concept, (l, n) = pair
        uid = f"L{l}_N{n}"
        lab = labels.get(uid, {})
        desc = {"positive pole": (lab.get("pos") or {}).get("explanation"),
                "negative pole": (lab.get("neg") or {}).get("explanation")}
        if uid in writes:
            desc["words it pushes the model to say"] = ", ".join(writes[uid][:8])
        instr = (f"This describes one unit inside a language model. Relative to the concept '{concept}', "
                 "which category best fits the unit?")
        total = {k: 0.0 for k in keys}
        for rot in range(3):                                  # rotate option order: Jev has a first-option bias
            order = keys[rot:] + keys[:rot]
            for k, p in jev.choice(desc, instr, {k: CLASSES[k] for k in order}).items():
                total[k] += p / 3
        return concept, uid, l, max(total, key=total.get), {k: round(v, 3) for k, v in total.items()}

    pairs = [(c, u) for c, us in cs.items() for u in us]
    with ThreadPoolExecutor(128) as ex:
        res = list(ex.map(classify, pairs))
    (R / f"hdepth_classes{tag}.json").write_text(json.dumps(res, indent=0), encoding="utf-8")

    bands = [(0, 5), (6, 12), (13, 21), (22, 32), (33, 35)]
    band = lambda l: next(f"{a}-{b}" for a, b in bands if a <= l <= b)   # noqa: E731
    all_b = Counter(band(l) for _, _, l, _, _ in res)
    print(f"\n{len(res)} pairs over {len(cs)} concepts. Share of each class by depth band "
          "(and over/under-representation vs all top-N neurons in that band):")
    for cls in keys:
        sub = [l for _, _, l, c, _ in res if c == cls]
        cb = Counter(band(l) for l in sub)
        cells = []
        for a, b in bands:
            k = f"{a}-{b}"
            share = cb[k] / max(len(sub), 1); base = all_b[k] / len(res)
            cells.append(f"{k}: {share:.0%} (x{share / base:.1f})" if base else f"{k}: -")
        print(f"  {cls:<17} n={len(sub):>3} | " + " | ".join(cells))
    gen = defaultdict(set)
    for c, u, _, cls, _ in res:
        if cls == "GENERIC":
            gen[u].add(c)
    rec = sorted(((len(v), u) for u, v in gen.items()), reverse=True)[:10]
    print(f"\nmost recurrent GENERIC neurons (in how many of {len(cs)} concepts' top-{N}):",
          ", ".join(f"{u} ({k})" for k, u in rec))
    print(f"Jev: {jev.usage}, ~${jev.cost():.3f}")


if __name__ == "__main__":
    main()
