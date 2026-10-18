"""Are entity (knowledge) neurons shared between entities - and what else do they respond to? (no model, no API)

James (18 Oct): towards polysemanticity. Uses facts2's direct entity neurons (top-3 per entity, mean drop > 0.2, zeroed
only at the name tokens) and the specificity test (each zeroed at the name tokens of every fact).
  causal sharing   for each entity neuron: OTHER entities whose facts drop by > max(0.2, 25% of its own-entity drop)
  map overlap      neurons appearing in the top-20 entity maps of >= 2 entities
  input side       its 20 strongest web-text firings (atlas, 10M tokens; top or bottom side by the sign it takes at the
                   entity's name): the firing token in +-8 tokens of context; whether the entity's own name (any word of
                   it) is at the firing token or within the context -> share of firings that are 'about its entity'
Output: D:/already-clean-data/facts2/entity_sharing.json, .txt.  Usage: python src/loop_v2/entity_sharing.py
"""
import json
import re
import statistics as stx
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
D = Path("D:/already-clean-data/facts2")


def main():
    from transformers import AutoTokenizer
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from config import DEFAULT_MODEL, model_path
    tok = AutoTokenizer.from_pretrained(model_path(DEFAULT_MODEL))
    EM = [json.loads(l) for l in (D / "entity_maps.jsonl").read_text(encoding="utf-8").splitlines()]
    S = [json.loads(l) for l in (D / "entity_spec.jsonl").read_text(encoding="utf-8").splitlines()]
    units = [tuple(x) for x in json.loads((D / "entity_units.json").read_text(encoding="utf-8"))]
    kind_of = {r["entity"]: r["kind"] for r in EM}
    by_ent = defaultdict(lambda: defaultdict(list))
    for s in S:
        for u, d in s["drops"].items():
            by_ent[u][s["entity"]].append(d)
    L = []
    rows = []
    for u, e in units:
        means = {ent: stx.mean(v) for ent, v in by_ent[u].items()}
        own = means.get(e, 0); thr = max(0.2, 0.25 * own)
        others = sorted([(ent, m) for ent, m in means.items() if ent != e and m > thr], key=lambda x: -x[1])
        rows.append({"unit": u, "entity": e, "kind": kind_of.get(e), "own": round(own, 3), "shared_with": [[a, round(b, 3)] for a, b in others]})
    n_sh = sum(bool(r["shared_with"]) for r in rows)
    L.append(f"CAUSAL sharing ({len(rows)} entity neurons x {len({s['entity'] for s in S})} entities; shared = another entity's facts drop > max(0.2, 25% of own)):")
    L.append(f"  shared with >= 1 other entity: {n_sh}/{len(rows)} ({n_sh / len(rows):.0%}); distribution of #others: {dict(sorted(Counter(len(r['shared_with']) for r in rows).items()))}")
    same_kind = sum(all(kind_of.get(a) == r["kind"] for a, _ in r["shared_with"]) for r in rows if r["shared_with"])
    L.append(f"  of shared ones, all partners of the same kind: {same_kind}/{n_sh}")
    for r in sorted([r for r in rows if r["shared_with"]], key=lambda r: -len(r["shared_with"]))[:15]:
        L.append(f"    {r['unit']:<10} {r['entity'][:24]:<24} own {r['own']:.2f} | also: {[(a, b) for a, b in r['shared_with'][:5]]}")
    cnt = Counter(x["neuron"] for r in EM for x in r["top"][:20])
    multi = {u: [r["entity"] for r in EM if any(x["neuron"] == u for x in r["top"][:20])] for u, c in cnt.items() if c >= 2}
    L.append(f"\nMAP overlap: {len(cnt)} distinct neurons in the 86 entity maps' top-20; in >= 2 maps: {len(multi)}")
    for u, es in sorted(multi.items(), key=lambda kv: -len(kv[1]))[:12]:
        L.append(f"    {u:<10} {len(es)} entities: {es[:6]}")
    # input side
    A = torch.load(ROOT / f"data/atlas_{DEFAULT_MODEL.split('/')[-1]}.pt", map_location="cpu", weights_only=False)
    d = np.load(ROOT / f"data/atlas_{DEFAULT_MODEL.split('/')[-1]}_docs.npz"); toks, lens = d["tokens"], d["lengths"]
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    st = torch.load(ROOT / f"data/stats_{DEFAULT_MODEL.split('/')[-1]}.pt", map_location="cpu", weights_only=False)
    inp = []
    for r in rows:
        l, n = (int(x) for x in r["unit"][1:].split(":"))
        side = "top" if st["max"][l, n] >= -st["min"][l, n] else "bot"
        words = [w.lower() for w in re.findall(r"\w{3,}", r["entity"]) if w.lower() not in {"the", "and", "of"}]
        about, snips = 0, []
        for c in A[f"{side}_c"][l, n, :20].tolist():
            dd, p = c // 1024, c % 1024; g = starts[dd] + p
            lo = max(starts[dd], g - 8); hi = min(starts[dd] + lens[dd], g + 9)
            ctx = tok.decode(toks[lo:hi]); fire = tok.decode(toks[g:g + 1])
            hit = any(w in ctx.lower() for w in words)
            about += hit
            snips.append((tok.decode(toks[lo:g]) + "[[" + fire + "]]" + tok.decode(toks[g + 1:hi])).replace("\n", " "))
        r["input_about_entity"] = about / 20; r["input_snippets"] = snips; inp.append(about / 20)
    L.append(f"\nINPUT side (20 strongest web-text firings each; 'about its entity' = a word of the entity's name in +-8 tokens):")
    L.append(f"  share of firings about its own entity: median {stx.median(inp):.0%}; neurons with >= 50% about their entity: "
             f"{sum(x >= 0.5 for x in inp)}/{len(inp)}; with 0%: {sum(x == 0 for x in inp)}/{len(inp)}")
    for kd in ("landmark", "person", "work"):
        xs = [r["input_about_entity"] for r in rows if r["kind"] == kd]
        if xs:
            L.append(f"  {kd:<9} median {stx.median(xs):.0%} (n={len(xs)})")
    L.append("  examples (strongest entity neurons: share about own entity | 3 web firings):")
    for r in sorted(rows, key=lambda r: -r["own"])[:10]:
        L.append(f"    {r['unit']:<10} {r['entity'][:20]:<20} {r['input_about_entity']:.0%} | " + " || ".join(s[:70] for s in r["input_snippets"][:3]))
    txt = "\n".join(L)
    (D / "entity_sharing.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    (D / "entity_sharing.txt").write_text(txt + "\n", encoding="utf-8"); print(txt)


if __name__ == "__main__":
    main()
