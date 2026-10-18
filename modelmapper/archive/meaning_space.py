"""Where do a neuron's meanings sit in Qwen's OWN representation space? (stage C of the meaning / Ur-concept test; GPU)

For every valid unit in meaning_judge.json: the 30 judged firing contexts are run through Qwen and, at each firing
position, the input the neuron's MLP reads (post-attention-norm output of that layer; for spliced units the layer of
their first neuron) is captured. Vectors are
  - centred and whitened with that layer's covariance, estimated from ~16k random corpus tokens (so distances reflect
    directions the model distinguishes, not a few huge dimensions), and
  - stripped of the neuron's own read directions (its gate and up weight rows; both, for spliced units), because every
    firing context shares "this neuron fired here", which would make all of a neuron's meanings look alike.
Per unit: Luna's meaning groups (stage A) -> group centroids. Measures:
  between   cosine between group centroids (for each pair, next to Luna's 0-3 relatedness rating)
  within    mean cosine between snippets of the same group (reference for 'same meaning')
  hub       for units with >= 3 groups: each group's mean cosine to the other groups' centroids; hub score = top minus
            the mean of the rest; null = 300 random reassignments of snippets to groups (sizes kept) -> per-unit p.
Output: results/meaning_space.json, results/meaning_space_vectors.npz (every unit's whitened context vectors, so
Luna-free groupings can be computed later); summary in meaning_space_summary.py.
Usage: python src/meaning_space.py
"""
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from config import DEFAULT_MODEL
from neurons import DEVICE, load, mlps

ROOT = Path(__file__).resolve().parents[1]
COV_TOKENS, NULL = 16000, 300


def main():
    J = json.loads((ROOT / "results/meaning_judge.json").read_text(encoding="utf-8"))["jobs"]
    units = {k: v for k, v in J.items() if v.get("valid") and len(v["result"]["groups"]) >= 2}
    model, tok = load(DEFAULT_MODEL)
    d = np.load(ROOT / f"data/atlas_{DEFAULT_MODEL.split('/')[-1]}_docs.npz")
    toks, lens = d["tokens"], d["lengths"]; starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    L = model.config.num_hidden_layers; H = model.config.hidden_size
    # 1. per-layer covariance of the MLP input over random corpus tokens
    S1 = torch.zeros(L, H, device=DEVICE, dtype=torch.float64); S2 = torch.zeros(L, H, H, device=DEVICE, dtype=torch.float64); n = 0
    cap = {}
    hk = [mlps(model)[l].register_forward_pre_hook(lambda m, a, l=l: cap.__setitem__(l, a[0][0])) for l in range(L)]
    rng = random.Random(41); docs = rng.sample(range(len(lens)), 400)
    try:
        for di in docs:
            if n >= COV_TOKENS:
                break
            ids = torch.tensor(toks[starts[di]:starts[di] + min(lens[di], 256)].astype(np.int64), device=DEVICE)[None]
            with torch.no_grad():
                model.model(input_ids=ids)
            for l in range(L):
                x = cap[l][1:].double()                     # skip the attention-sink token
                S1[l] += x.sum(0); S2[l] += x.T @ x
            n += ids.shape[1] - 1
    finally:
        for h in hk:
            h.remove()
    mu = S1 / n; Wh = {}
    def whitener(l):
        if l not in Wh:
            C = S2[l] / n - torch.outer(mu[l], mu[l])
            ev, V = torch.linalg.eigh(C)
            Wh[l] = (V @ torch.diag(ev.clamp_min(ev.max() * 1e-6) ** -0.5) @ V.T).float()
        return Wh[l]
    print(f"covariance from {n} tokens", flush=True)
    # 2. per unit: capture the firing-position MLP inputs
    out = {}; vec_store = {}
    for key, v in units.items():
        names = v["units"]; l = int(names[0][1:].split(":")[0])
        reads = []
        for u in names:
            lu, nu = (int(x) for x in u[1:].split(":"))
            reads += [mlps(model)[lu].gate_proj.weight[nu].detach().float(), mlps(model)[lu].up_proj.weight[nu].detach().float()]
        vecs = []
        for di, p in v["doc_pos"]:
            ids = torch.tensor(toks[starts[di]:starts[di] + p + 1].astype(np.int64), device=DEVICE)[None]
            got = {}
            h = mlps(model)[l].register_forward_pre_hook(lambda m, a: got.__setitem__("x", a[0][0, -1].float()))
            try:
                with torch.no_grad():
                    model.model(input_ids=ids)
            finally:
                h.remove()
            vecs.append(got["x"])
        X = (torch.stack(vecs) - mu[l].float()) @ whitener(l)
        Rd = torch.stack([r @ whitener(l) for r in reads])           # read directions in the same whitened space
        Q, _ = torch.linalg.qr(Rd.T)                                  # orthonormal basis of the read directions
        X = X - (X @ Q) @ Q.T                                         # project them out
        X = X / X.norm(dim=1, keepdim=True)
        groups = v["result"]["groups"]
        lab = {}
        for gi, g in enumerate(groups):
            for m in g["members"]:
                lab[int(m) - 1] = gi
        idx = [i for i in range(len(vecs)) if i in lab]                # misc snippets left out of the geometry
        G = len(groups)

        def geometry(assign):
            cents = []
            for gi in range(G):
                mem = [i for i in idx if assign[i] == gi]
                c = X[mem].mean(0) if mem else torch.zeros(X.shape[1], device=X.device)
                cents.append(c / (c.norm() + 1e-9))
            C = torch.stack(cents); return C @ C.T
        S = geometry(lab)
        within = []
        for gi in range(G):
            mem = [i for i in idx if lab[i] == gi]
            if len(mem) >= 2:
                M = X[mem] @ X[mem].T; within.append(((M.sum() - M.trace()) / (len(mem) * (len(mem) - 1))).item())
        rel = {(int(p["a"]), int(p["b"])): int(p["rating"]) for p in v["result"]["relatedness"]}
        pairs = [{"a": i, "b": j, "cos": round(S[i, j].item(), 4), "luna": rel.get((i, j), rel.get((j, i)))}
                 for i in range(G) for j in range(i + 1, G)]
        rec = {"kind": v["kind"], "units": names, "layer": l, "n_groups": G, "pairs": pairs,
               "within_mean": round(sum(within) / len(within), 4) if within else None,
               "labels": [g.get("label") for g in groups], "sizes": [len(g["members"]) for g in groups]}
        if G >= 3:
            def hub_score(Sm):
                cen = [(sum(Sm[i, j].item() for j in range(G) if j != i)) / (G - 1) for i in range(G)]
                top = max(range(G), key=lambda i: cen[i])
                return cen[top] - (sum(cen) - cen[top]) / (G - 1), top, cen
            hs, top, cen = hub_score(S)
            r_ = random.Random(key); null = []
            labels = [lab[i] for i in idx]
            for _ in range(NULL):
                r_.shuffle(labels); null.append(hub_score(geometry(dict(zip(idx, labels))))[0])
            rec.update(hub_score=round(hs, 4), hub_group=top, hub_label=groups[top].get("label"),
                       centrality=[round(c, 4) for c in cen], hub_p=round((sum(x >= hs for x in null) + 1) / (NULL + 1), 4))
        out[key] = rec
        vec_store[key] = X.half().cpu().numpy()             # raw whitened context vectors, for Luna-free analyses
        print(f"{key[:55]:<55} groups {G} | between {min(p['cos'] for p in pairs):+.2f}..{max(p['cos'] for p in pairs):+.2f} | "
              f"within {rec['within_mean']}" + (f" | hub '{rec['hub_label']}' p={rec['hub_p']}" if G >= 3 else ""), flush=True)
    (ROOT / "results/meaning_space.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(ROOT / "results/meaning_space_vectors.npz", keys=np.array(list(vec_store)), **{f"v{i}": a for i, a in enumerate(vec_store.values())})


if __name__ == "__main__":
    main()
