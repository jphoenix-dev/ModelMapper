"""The fact route, causally: does an entity's early neuron (at its name) drive the shared VALUE neuron at the answer?
(GPU; no API) - runs on facts2's data after its grid.

James (18 Oct): facts look like a factorised lookup - entity neurons early at the name (rows), relation neurons (columns),
answer-value neurons late (values, shared by every entity with that value). Test the route.
Value neurons: from facts2's grid (each fact neuron zeroed on every fact), neurons whose mean drop on OTHER facts with the
SAME answer is > max(0.2, 25% of own) and > 3x their drop on same-relation facts with a different answer.
For every value neuron V and every fact g with V's answer: g's entity's top-1 direct entity neuron E (facts2 entity maps)
zeroed ONLY at g's name tokens; V's activation at the last prompt position (sign-aligned to its as-is value):
  entity   E zeroed                                   -> relative drop of V
  null     a random neuron from E's layer zeroed at the same tokens
  cross    another entity's top-1 entity neuron (an entity WITHOUT V's answer), zeroed at g's name tokens
Also the answer's log-odds drop in each case. One pass per fact (as-is + 3 ablated copies).
Prediction (stated before running): median relative drop of V under 'entity' >= 20%, with null and cross ~0.
Output: D:/already-clean-data/facts2/chain.jsonl, chain.txt.  Usage: python src/loop_v2/chain.py
"""
import json
import random
import statistics as stx
import sys
from collections import defaultdict
from pathlib import Path

import torch
from modelmapper.core.settings import DATA_DIR, REPO_ROOT

ROOT = REPO_ROOT

from modelmapper.experiments import facts as F1  # noqa: E402

D = (DATA_DIR / "facts2")
import os
K = int(os.environ.get("CHAIN_K", "1"))   # follow-up (after K=1 failed its 20% prediction): zero the entity's top-K entity neurons together


def main():
    from modelmapper.core.neurons import DEVICE, mlps
    Fs = json.loads((D / "facts.json").read_text(encoding="utf-8"))
    G = {(g["entity"], g["relation"]): g["drops"] for g in map(json.loads, (D / "grid.jsonl").read_text(encoding="utf-8").splitlines())}
    units = json.loads((D / "grid_units.json").read_text(encoding="utf-8"))
    EM = {r["entity"]: r for r in map(json.loads, (D / "entity_maps.jsonl").read_text(encoding="utf-8").splitlines())}
    ans = {(f["entity"], f["relation"]): f["answer"].lower() for f in Fs}
    fact_of = {(f["entity"], f["relation"]): f for f in Fs}
    vals = {}
    for e, r, u in units:
        a = ans[(e, r)]; own = G[(e, r)].get(u, 0)
        same = [G[k][u] for k in G if k != (e, r) and ans.get(k) == a and u in G[k]]
        diff = [G[k][u] for k in G if k[1] == r and ans.get(k) != a and u in G[k]]
        if same and stx.mean(same) > max(0.2, 0.25 * own) and stx.mean(same) > 3 * max(stx.mean(diff) if diff else 0, 0.03):
            vals[(u, a)] = [k for k in G if ans.get(k) == a]
    model, tok = F1.setup(); W = model.config.intermediate_size; rng = random.Random(161)
    ents_with = defaultdict(set)
    for k, a in ans.items():
        ents_with[a].add(k[0])
    tag = "" if K == 1 else f"_k{K}"
    f = open(D / f"chain{tag}.jsonl", "w", encoding="utf-8"); rows = []
    for (vu, a), facts_ in vals.items():
        vl, vn = int(vu[1:].split(":")[0]), int(vu.split(":")[1])
        for k in facts_:
            g = fact_of[k]; e = k[0]
            if e not in EM or not EM[e]["top"]:
                continue
            eus = [x["neuron"] for x in EM[e]["top"][:K] if int(x["neuron"][1:].split(":")[0]) < vl]
            if not eus:
                continue                                     # entity neurons must sit before the value neuron
            eu = eus[0]; el, en = int(eu[1:].split(":")[0]), int(eu.split(":")[1])
            ents_units = [(int(x[1:].split(":")[0]), int(x.split(":")[1])) for x in eus]
            others = [x for x in EM if x not in ents_with[a] and EM[x]["top"] and int(EM[x]["top"][0]["neuron"][1:].split(":")[0]) < vl]
            if not others:
                continue
            ce = rng.choice(sorted(others))
            cross_units = [(int(x["neuron"][1:].split(":")[0]), int(x["neuron"].split(":")[1])) for x in EM[ce]["top"][:K]
                           if int(x["neuron"][1:].split(":")[0]) < vl][:len(ents_units)]
            cu = f"L{cross_units[0][0]}:{cross_units[0][1]}"
            null_units = [(l_, rng.randrange(W)) for l_, _ in ents_units]; nn = null_units[0][1]
            P = len(g["prompts"]); conds = [None, ents_units, null_units, cross_units]
            grp = []
            for c in conds:
                grp += [(p, g["answer_ids"], [(l_, n_, sp) for l_, n_ in c] if (c and sp) else []) for p, sp in zip(g["prompts"], g["subject_pos"])]
            cap = {}
            hk = mlps(model)[vl].down_proj.register_forward_pre_hook(lambda mod, a_: cap.__setitem__("v", a_[0][:, :, vn].float().clone()))
            try:
                lo = F1.logodds_groups(model, tok, [grp])[0]
            finally:
                hk.remove()
            act = [cap["v"][i, len(grp[i][0]) - 1].item() for i in range(len(grp))]          # V at the last prompt position
            base = act[:P]; sg = [1.0 if b > 0 else -1.0 for b in base]
            rel = lambda j: stx.mean(s_ * (b - x) / max(abs(b), 1e-3) for s_, b, x in zip(sg, base, act[P * j:P * (j + 1)]))
            lod = lambda j: stx.mean(b - x for b, x in zip(lo[:P], lo[P * j:P * (j + 1)]))
            rec = {"k": len(ents_units), "value_neuron": vu, "answer": a, "entity": e, "relation": k[1], "entity_neuron": eu, "cross_neuron": cu,
                   "null_neuron": f"L{el}:{nn}", "v_base": round(stx.mean(base), 3),
                   "v_rel_drop": {"entity": round(rel(1), 4), "null": round(rel(2), 4), "cross": round(rel(3), 4)},
                   "answer_drop": {"entity": round(lod(1), 4), "null": round(lod(2), 4), "cross": round(lod(3), 4)}}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush(); rows.append(rec)
    md = stx.median
    L = [f"[top-{K} entity neurons zeroed together] Fact route test: {len(vals)} value neurons, {len(rows)} (value neuron, fact) pairs with an earlier entity neuron",
         f"  value neuron's relative activation drop at the answer position: entity neuron zeroed at the name {md(r['v_rel_drop']['entity'] for r in rows):+.3f} | "
         f"random same-layer neuron {md(r['v_rel_drop']['null'] for r in rows):+.3f} | another entity's neuron {md(r['v_rel_drop']['cross'] for r in rows):+.3f}",
         f"  answer log-odds drop: entity {md(r['answer_drop']['entity'] for r in rows):+.3f} | null {md(r['answer_drop']['null'] for r in rows):+.3f} | "
         f"cross {md(r['answer_drop']['cross'] for r in rows):+.3f}",
         f"  pairs where the entity drop beats both controls: {sum(r['v_rel_drop']['entity'] > max(r['v_rel_drop']['null'], r['v_rel_drop']['cross']) for r in rows)}/{len(rows)}",
         f"  PREDICTION (median entity drop >= 20%, controls ~0): "
         f"{'HOLDS' if rows and md(r['v_rel_drop']['entity'] for r in rows) >= 0.2 else 'FAILS'}"]
    L.append("  examples: " + "; ".join(f"{r['entity']} -> '{r['answer']}' ({r['value_neuron']}): entity {r['v_rel_drop']['entity']:+.2f}, null {r['v_rel_drop']['null']:+.2f}, "
                                        f"cross {r['v_rel_drop']['cross']:+.2f}" for r in sorted(rows, key=lambda r: -r['v_rel_drop']['entity'])[:8]))
    txt = "\n".join(L)
    (D / f"chain{tag}.txt").write_text(txt + "\n", encoding="utf-8"); print(txt)


if __name__ == "__main__":
    main()
