"""Follow-up to job_overlap.py (added after its rank-based result): does each late neuron RAISE the answer, even if
not to the top? Rank among 151,936 tokens is harsh - a push can raise the answer clearly and still rank it 5,000th.
Here: z = (answer logit - mean over vocabulary) / std, for each late neuron's direct push and for their sum.
  - 'raises it' = z >= 2; 'votes' = how many late neurons raise it
  - intersection/voting signature: >= 2 neurons raise it and the sum's z beats the best single neuron's z
Nulls as in job_overlap.py: answer-shuffle and random same-layer neurons (20 draws each). Also split by answer-token
type: a whole word (>= 4 letters) vs a short fragment (e.g. 'N' of a multi-token name), since fragments are shared by
many concepts. (GPU for speed, no API.)
Output: results/job_overlap_z.json, results/job_overlap_z.txt.  Usage: python src/job_overlap_z.py
"""
import json
import random
import statistics as stx
from collections import Counter
from pathlib import Path

import torch

from config import DEFAULT_MODEL
from job_overlap import LATE, NULLS, concepts
from neurons import load, mlps

ROOT = Path(__file__).resolve().parents[1]


def main():
    C = concepts()
    model, tok = load(DEFAULT_MODEL)
    for c in list(C):
        ids = tok.encode(C[c]["tok"]); assert len(ids) == 1 and tok.decode(ids) == C[c]["tok"], c
        C[c]["word"] = C[c]["tok"]; C[c]["tok"] = ids[0]
    WU = model.lm_head.weight.detach().float(); g = model.model.norm.weight.detach().float()
    downs = [m.down_proj.weight.detach() for m in mlps(model)]; W = model.config.intermediate_size

    def zs(units, t):
        V = torch.stack([a * downs[L][:, n].float() for L, n, a in units]); V = torch.cat([V, V.sum(0, keepdim=True)])
        lg = (V * g) @ WU.T
        return ((lg[:, t] - lg.mean(1)) / lg.std(1)).tolist()

    keys = [c for c in C if sum(L >= LATE for L, _, _ in C[c]["top"]) >= 2]
    rng = random.Random(12); toks = [C[c]["tok"] for c in C]

    def summarise(rows):
        votes = Counter(sum(z >= 2 for z in r[:-1]) for r in rows)
        inter = sum(sum(z >= 2 for z in r[:-1]) >= 2 and r[-1] > max(r[:-1]) for r in rows)
        return {"median_best_single_z": stx.median(max(r[:-1]) for r in rows), "median_sum_z": stx.median(r[-1] for r in rows),
                "share_any_raises": sum(max(r[:-1]) >= 2 for r in rows) / len(rows), "votes": dict(sorted(votes.items())),
                "share_voting": inter / len(rows), "share_sum_z_ge2": sum(r[-1] >= 2 for r in rows) / len(rows)}

    real = {c: zs([u for u in C[c]["top"] if u[0] >= LATE], C[c]["tok"]) for c in keys}
    na, nb = [], []
    for _ in range(NULLS):
        for c in keys:
            late = [u for u in C[c]["top"] if u[0] >= LATE]
            o = C[c]["tok"]
            while o == C[c]["tok"]:
                o = toks[rng.randrange(len(toks))]
            na.append(zs(late, o)); nb.append(zs([(L, rng.randrange(W), a) for L, _, a in late], C[c]["tok"]))
    word = [c for c in keys if len(C[c]["word"].strip()) >= 4]; frag = [c for c in keys if c not in word]
    out = {"real": summarise(list(real.values())), "answer_shuffle": summarise(na), "random_neurons": summarise(nb),
           "real_whole_word": summarise([real[c] for c in word]), "real_fragment": summarise([real[c] for c in frag]),
           "n": len(keys), "n_word": len(word), "n_fragment": len(frag)}
    L = [f"Does each late neuron RAISE the answer? z of the answer logit in each direct push (>= 2 = raises it); {len(keys)} concepts",
         f"{'':<22}{'best single z':>14}{'sum z':>8}{'any raises':>12}{'sum z>=2':>10}{'voting*':>9}  votes (how many late neurons raise it)"]
    for k in ("real", "answer_shuffle", "random_neurons", "real_whole_word", "real_fragment"):
        s = out[k]
        L.append(f"{k:<22}{s['median_best_single_z']:>14.2f}{s['median_sum_z']:>8.2f}{s['share_any_raises']:>12.0%}{s['share_sum_z_ge2']:>10.0%}"
                 f"{s['share_voting']:>9.0%}  {s['votes']}")
    L.append(f"*voting = >= 2 late neurons raise the answer AND the sum raises it more than the best single one. "
             f"whole-word answers n={len(word)}, fragments n={len(frag)}")
    (ROOT / "results/job_overlap_z.json").write_text(json.dumps({"summary": out, "per_concept_z": real}, indent=1, ensure_ascii=False), encoding="utf-8")
    (ROOT / "results/job_overlap_z.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
