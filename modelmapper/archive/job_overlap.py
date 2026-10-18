"""Is a concept the INTERSECTION of its neurons' jobs? Weights only (logit lens of each neuron's direct push).
(GPU for speed, no API, no forward passes)

James (17 Oct): each neuron has a job; a concept is what a few jobs add up to. If so, a concept's answer word should
often sit where its top neurons' pushes overlap, rather than at the top of any single neuron's push.

For each mapped concept (all rank_save files, first occurrence wins), its top-5 neurons by single-knockout drop; the
'late' ones are those in layers >= 20, where the logit lens (unembedding of the final-norm-weighted vector) is a
reasonable reading of a direct push. A neuron's push = its activation at the answer position x its down_proj column.
The answer token = the first token of the recalled answer form. Rank 1 = the token the push raises most (of 151,936).
  dedicated     some single late neuron has the answer in its top-10
  intersection  no single late neuron does, but the SUM of the late pushes does
  neither       the sum doesn't either
  overlap       in how many of the late neurons' top-1000 (0.66% of the vocabulary) the answer sits
Nulls: (a) answer-shuffle: same neurons, another concept's answer token (20 draws); (b) random neurons: each late
neuron replaced by a random neuron of the same layer with the same activation (20 draws).
Criterion (stated before running, not pre-registered in PREREG_MAP): among concepts with >= 2 late neurons and no
single-neuron top-10 hit, the sum reaches top-10 at >= 3x the rate of both nulls, with binomial p < 0.05 vs the
higher null rate.
Caveats: direct path only (a layer-30 neuron also acts through layers 31-35); the final RMSNorm scale is ignored
(a positive scalar per position, which doesn't change ranks).
Output: results/job_overlap.json (per concept: every neuron's answer rank, top-30 raised tokens, sum rank and
top-30), results/job_overlap.txt.  Usage: python src/job_overlap.py
"""
import glob
import json
import math
import random
from collections import Counter
from pathlib import Path

import torch

from config import DEFAULT_MODEL
from neurons import DEVICE, load, mlps

ROOT = Path(__file__).resolve().parents[1]
TOP, LATE, HIT, OVL, NULLS = 5, 20, 10, 1000, 20


def concepts():
    out = {}
    for f in sorted(glob.glob(str(ROOT / "results/concepts_*/rank_save_*.json"))):
        if "rank_save_test" in f or f.endswith("_exact.json"):
            continue
        for c, v in json.loads(Path(f).read_text(encoding="utf-8")).items():
            if "top" in v and c not in out:
                out[c] = {"tok": v["answer_token"], "set": Path(f).parent.name,
                          "top": [(int(e["neuron"].split(":")[0][1:]), int(e["neuron"].split(":")[1]), float(e["act_answer_pos"]))
                                  for e in v["top"][:TOP]]}
    return out


def main():
    C = concepts()
    model, tok = load(DEFAULT_MODEL)
    bad = []                                                          # answer_token is saved decoded; re-encode it
    for c in list(C):
        ids = tok.encode(C[c]["tok"])
        if len(ids) != 1 or tok.decode(ids) != C[c]["tok"]:
            bad.append((c, C[c]["tok"], ids)); del C[c]
        else:
            C[c]["tok"] = ids[0]
    WU = model.lm_head.weight.detach().float()                       # [V, 4096]
    g = model.model.norm.weight.detach().float()
    downs = [m.down_proj.weight.detach() for m in mlps(model)]       # [4096, 12288] each
    W = model.config.intermediate_size

    def ranks(vecs, toks):
        """vecs [N,4096] float, toks [N] -> rank of each tok in its row's logits (1 = most raised), plus logits top-30 ids."""
        out_r, out_t = [], []
        for i in range(0, len(vecs), 256):
            lg = (vecs[i:i + 256] * g) @ WU.T
            t = torch.tensor(toks[i:i + 256], device=DEVICE)
            val = lg.gather(1, t[:, None])
            out_r += ((lg > val).sum(1) + 1).tolist(); out_t += lg.topk(30).indices.tolist()
        return out_r, out_t

    def push(L, n, a):
        return a * downs[L][:, n].float()

    keys = list(C)
    answer_toks = [C[c]["tok"] for c in keys]
    res = {}
    # real
    for c in keys:
        late = [(L, n, a) for L, n, a in C[c]["top"] if L >= LATE]
        if not late:
            res[c] = {"late": 0}; continue
        V = torch.stack([push(*u) for u in late]); S = V.sum(0, keepdim=True)
        r, t = ranks(torch.cat([V, S]), [C[c]["tok"]] * (len(late) + 1))
        lg = (V * g) @ WU.T; ov = int((lg >= lg.topk(OVL).values[:, -1:])[:, C[c]["tok"]].sum())
        res[c] = {"late": len(late), "neurons": [f"L{L}:{n}" for L, n, _ in late], "acts": [round(a, 3) for *_, a in late],
                  "answer": tok.decode([C[c]["tok"]]), "ranks": r[:-1], "sum_rank": r[-1], "overlap_top1000": ov,
                  "top30": [[tok.decode([i]) for i in x] for x in t[:-1]], "sum_top30": [tok.decode([i]) for i in t[-1]]}

    def classify(single_ranks, sum_rank):
        if min(single_ranks) <= HIT:
            return "dedicated"
        return "intersection" if sum_rank <= HIT else "neither"

    rng = random.Random(11)
    elig = [c for c in keys if res[c]["late"] >= 2]
    real = Counter(classify(res[c]["ranks"], res[c]["sum_rank"]) for c in elig)
    real_ov = Counter(res[c]["overlap_top1000"] for c in elig)
    # null (a): answer shuffle
    na = Counter(); na_ov = Counter()
    for _ in range(NULLS):
        for c in elig:
            late = [(L, n, a) for L, n, a in C[c]["top"] if L >= LATE]
            other = C[c]["tok"]
            while other == C[c]["tok"]:
                other = answer_toks[rng.randrange(len(answer_toks))]
            V = torch.stack([push(*u) for u in late]); S = V.sum(0, keepdim=True)
            r, _ = ranks(torch.cat([V, S]), [other] * (len(late) + 1))
            na[classify(r[:-1], r[-1])] += 1
            lg = (V * g) @ WU.T; na_ov[int((lg >= lg.topk(OVL).values[:, -1:])[:, other].sum())] += 1
    # null (b): random same-layer neurons, same activation
    nb = Counter()
    for _ in range(NULLS):
        for c in elig:
            late = [(L, rng.randrange(W), a) for L, n, a in C[c]["top"] if L >= LATE]
            V = torch.stack([push(*u) for u in late]); S = V.sum(0, keepdim=True)
            r, _ = ranks(torch.cat([V, S]), [C[c]["tok"]] * (len(late) + 1))
            nb[classify(r[:-1], r[-1])] += 1

    def share(cn, k):
        return cn[k] / max(1, sum(cn.values()))

    L = [f"answer tokens that do not round-trip to one id (excluded, listed): {len(bad)} {bad}",
         f"Concept = intersection of jobs? {len(keys)} concepts; {len(elig)} with >= 2 late (L>={LATE}) neurons in their top-{TOP}",
         f"late neurons per concept: {dict(sorted(Counter(res[c]['late'] for c in keys).items()))}",
         f"\n{'':<14}{'real':>10}{'answer-shuffle':>16}{'random neurons':>16}"]
    for k in ("dedicated", "intersection", "neither"):
        L.append(f"{k:<14}{share(real, k):>10.1%}{share(na, k):>16.1%}{share(nb, k):>16.1%}")
    # criterion: among concepts with no single top-10 hit
    def cond(cn):
        d = cn["intersection"] + cn["neither"]; return cn["intersection"] / d if d else 0.0
    n_cond = real["intersection"] + real["neither"]; p0 = max(cond(na), cond(nb))
    k_ = real["intersection"]
    p = sum(math.comb(n_cond, i) * p0 ** i * (1 - p0) ** (n_cond - i) for i in range(k_, n_cond + 1)) if n_cond else 1.0
    L.append(f"\nno single-neuron top-{HIT} hit: the sum reaches top-{HIT} in {k_}/{n_cond} ({cond(real):.1%}); nulls: answer-shuffle "
             f"{cond(na):.2%}, random neurons {cond(nb):.2%}; binomial p vs the higher null = {p:.2g}")
    ok = n_cond and cond(real) >= 3 * max(cond(na), cond(nb), 1e-9) and p < 0.05
    L.append(f"CRITERION {'MET' if ok else 'NOT MET'} (>= 3x both nulls and p < 0.05)")
    L.append(f"\nanswer in how many late neurons' top-{OVL} (concepts with >= 2 late): real {dict(sorted(real_ov.items()))}; "
             f"answer-shuffle (x{NULLS}) {dict(sorted(na_ov.items()))}")
    med = lambda xs: sorted(xs)[len(xs) // 2]
    L.append(f"median best single-neuron rank {med([min(res[c]['ranks']) for c in elig])}; median sum rank {med([res[c]['sum_rank'] for c in elig])}; "
             f"sum better than best single in {sum(res[c]['sum_rank'] < min(res[c]['ranks']) for c in elig)}/{len(elig)}")
    L.append("\nexamples (intersection): concept | answer | each neuron's rank -> sum rank | each neuron's top-5 raised tokens")
    ex = [c for c in elig if classify(res[c]["ranks"], res[c]["sum_rank"]) == "intersection"]
    for c in random.Random(3).sample(ex, min(12, len(ex))):
        r = res[c]
        L.append(f"  {c} | {r['answer']!r} | {r['ranks']} -> {r['sum_rank']} | " + " / ".join(",".join(repr(x) for x in t5[:5]) for t5 in r["top30"]))
    txt = "\n".join(L)
    (ROOT / "results/job_overlap.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    (ROOT / "results/job_overlap.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
