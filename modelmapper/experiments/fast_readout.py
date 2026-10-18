"""Fast forced-neuron readout: a weighted word list per neuron, cheap enough for every neuron. (GPU; no API)

James (18 Oct): confirm the cheap output side first - validate, then map Qwen with it. Lean version of
neuron_readout.py: backgrounds = 32 random web snippets (24 tokens) + 16 'empty' (a single newline); the target is
clamped at the LAST position only, to m x its own web-text peak (default m = 4; sign = the larger of max / |min|, or
given); next token only; NB neurons per forward pass (each in its own rows), one unforced baseline per background.
Per neuron and background set: dlogp = forced - unforced next-token log-probs.
  words      top-50 by mean dlogp over backgrounds: mean dlogp, consistency (share of backgrounds where the word's dlogp
             is in that background's top-50), mean forced probability
  web_vs_empty  Jaccard of the top-10 words from the web vs the empty backgrounds (low-bias check)
  strength   the top word's mean dlogp
Modes:
  answers   the top 'say' neuron (else the top-1) of every mapped run-1 concept: rank of the concept's answer token in
            the readout vs in the weights-only direct push (roles_top30 answer_rank)
  layers    50 random neurons from every layer (seed 131): consistency, web-vs-empty agreement and strength by layer
  list      neurons given on the command line
Output: D:/already-clean-data/fast_readout/<mode>.jsonl (per neuron: top-50 weighted words for each set + metrics),
<mode>.txt.  Usage: python src/loop_v2/fast_readout.py answers | layers | list L:n[:sign] ...
"""
import glob
import json
import os
import random
import statistics as stx
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from modelmapper.core.config import DEFAULT_MODEL  # noqa: E402
from modelmapper.core.neurons import DEVICE, load, mlps  # noqa: E402
from modelmapper.core.settings import DATA_DIR, REPO_ROOT

ROOT = REPO_ROOT
OUT = DATA_DIR / "fast_readout"
NWEB, NEMPTY, CTX, NB, TOPW = 32, 16, 24, int(os.environ.get("NB", "16")), 50
MULT = float(os.environ.get("MULT", "4"))


def main():
    mode = sys.argv[1]; OUT.mkdir(parents=True, exist_ok=True)
    model, tok = load(DEFAULT_MODEL); L, W = model.config.num_hidden_layers, model.config.intermediate_size
    st = torch.load(ROOT / f"data/stats_{DEFAULT_MODEL.split('/')[-1]}.pt", map_location="cpu", weights_only=False)
    smax, smin = st["max"], st["min"]
    rng = random.Random(131)
    query = {}                                         # neuron -> answer token ids to rank (mode answers)
    direct = {}
    if mode == "answers":
        for b in glob.glob((DATA_DIR.as_posix() + "/loop_v2/run1/results/**/map/*/records.jsonl"), recursive=True):
            for line in Path(b).read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                if r["row"]["status"] != "ok":
                    continue
                roles = r["v2"]["roles_top30"][:5]
                pick = next((x for x in roles if x["role"] == "say"), roles[0])
                act = next(e["act_answer_pos"] for e in r["rank"]["top"] if e["neuron"] == pick["neuron"])
                key = (pick["neuron"], 1.0 if act > 0 else -1.0)
                query.setdefault(key, []).append((r["concept"], r["rank"]["answer_ids"][0], pick["answer_rank"], pick["role"]))
        units = sorted(query)
    elif mode == "layers":
        units = [(f"L{l}:{rng.randrange(W)}", None) for l in range(L) for _ in range(50)]
    else:
        units = []
        for s in sys.argv[2:]:
            p = s.split(":"); units.append((f"{p[0]}:{p[1]}", float(p[2]) if len(p) > 2 else None))
    d = np.load(ROOT / f"data/atlas_{DEFAULT_MODEL.split('/')[-1]}_docs.npz"); toks, lens = d["tokens"], d["lengths"]
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]]); ok = np.where(lens > CTX + 1)[0]
    bg = []
    for _ in range(NWEB):
        di = int(ok[rng.randrange(len(ok))]); p = rng.randrange(1, int(lens[di]) - CTX)
        bg.append([int(t) for t in toks[starts[di] + p: starts[di] + p + CTX]])
    bg += [tok.encode("\n")] * NEMPTY
    B = len(bg); T = max(map(len, bg)); pad = tok.pad_token_id or 0
    base_ids = torch.full((B, T), pad, device=DEVICE); base_mask = torch.zeros_like(base_ids)
    for i, s in enumerate(bg):                          # left padding: the last real token sits at -1 for every row
        base_ids[i, T - len(s):] = torch.tensor(s, device=DEVICE); base_mask[i, T - len(s):] = 1
    with torch.no_grad():
        base_lp = model(input_ids=base_ids, attention_mask=base_mask).logits[:, -1].float().log_softmax(-1)   # [B, V]
    f = open(OUT / f"{mode}.jsonl", "w", encoding="utf-8"); t0 = time.time(); recs = []
    for c0 in range(0, len(units), NB):
        chunk = units[c0:c0 + NB]; K = len(chunk)
        spec = []
        for u, sg in chunk:
            l, n = (int(x) for x in u[1:].split(":"))
            if sg is None:
                sg = 1.0 if smax[l, n] >= -smin[l, n] else -1.0
            pk = float(smax[l, n] if sg > 0 else -smin[l, n])
            spec.append((u, l, n, sg, sg * MULT * pk))
        ids = base_ids.repeat(K, 1); mask = base_mask.repeat(K, 1)
        by = defaultdict(list)
        for k, (_, l, n, _, v) in enumerate(spec):
            by[l].append((k, n, v))
        hks = []
        for l, items in by.items():
            rows = torch.cat([torch.arange(k * B, (k + 1) * B, device=DEVICE) for k, _, _ in items])
            idx = torch.tensor([n for _, n, _ in items for _ in range(B)], device=DEVICE)
            vals = torch.tensor([v for _, _, v in items for _ in range(B)], device=DEVICE)

            def h(mod, a, rows=rows, idx=idx, vals=vals):
                y = a[0].clone(); y[rows, -1, idx] = vals.to(y.dtype); return (y,)
            hks.append(mlps(model)[l].down_proj.register_forward_pre_hook(h))
        try:
            with torch.no_grad():
                lp = model(input_ids=ids, attention_mask=mask).logits[:, -1].float().log_softmax(-1)          # [K*B, V]
        finally:
            for hk in hks:
                hk.remove()
        for k, (u, l, n, sg, v) in enumerate(spec):
            dl = lp[k * B:(k + 1) * B] - base_lp; pr = lp[k * B:(k + 1) * B].exp()
            out = {"neuron": u, "layer": l, "sign": sg, "value": round(v, 3)}
            tops = {}
            for name, sl in (("web", slice(0, NWEB)), ("empty", slice(NWEB, B))):
                dd = dl[sl]; m_ = dd.mean(0); tw = m_.topk(TOPW)
                per_top = dd.topk(TOPW, dim=1).indices                                         # [n_bg, TOPW]
                cons = torch.zeros_like(m_); cons.index_add_(0, per_top.flatten(), torch.ones(per_top.numel(), device=DEVICE))
                cons /= dd.shape[0]
                out[f"{name}_words"] = [[tok.decode([i]), round(float(m_[i]), 3), round(float(cons[i]), 3), round(float(pr[sl][:, i].mean()), 4)]
                                        for i in tw.indices.tolist()]
                tops[name] = tw.indices[:10].tolist()
                if name == "web":
                    allm = m_
            out["web_vs_empty_top10_jaccard"] = len(set(tops["web"]) & set(tops["empty"])) / len(set(tops["web"]) | set(tops["empty"]))
            # per background (James, 18 Oct: do the words / probabilities depend on what else is active?): the 20 most probable
            # next tokens when forced, with forced and unforced probability; context dependence = 1 - mean pairwise Jaccard of
            # the per-background top-10 (by dlogp) over the web backgrounds (0 = the same words whatever the context)
            fp = pr; up = base_lp.exp()
            top20 = fp.topk(20, dim=1)
            out["per_background"] = [[[tok.decode([i]), round(float(p), 4), round(float(up[b, i]), 4)] for i, p in zip(top20.indices[b].tolist(), top20.values[b].tolist())]
                                     for b in range(B)]
            t10 = [set(x) for x in dl[:NWEB].topk(10, dim=1).indices.tolist()]
            jj = [len(a & b) / len(a | b) for ai, a in enumerate(t10) for b in t10[ai + 1:]]
            out["context_dependence"] = round(1 - sum(jj) / len(jj), 3)
            out["strength"] = round(float(dl[:NWEB].mean(0).max()), 3)
            if mode == "answers":
                full = dl.mean(0)
                out["answers"] = [{"concept": c, "answer_rank_forced": int((full > full[a]).sum()) + 1, "answer_rank_direct": dr, "role": ro}
                                  for c, a, dr, ro in query[(u, sg)]]
            f.write(json.dumps(out, ensure_ascii=False) + "\n"); recs.append(out)
        if (c0 // NB) % 20 == 0:
            el = time.time() - t0
            print(f"{c0 + K}/{len(units)} neurons, {el:.0f} s ({1000 * el / (c0 + K):.0f} ms/neuron)", flush=True)
    f.flush(); el = time.time() - t0
    md = stx.median
    Lines = [f"fast readout ({mode}): {len(recs)} neurons in {el:.0f} s ({1000 * el / max(1, len(recs)):.0f} ms per neuron); x{MULT:g} peak, "
             f"{NWEB} web + {NEMPTY} empty backgrounds, last position, next token"]
    if mode == "answers":
        a = [x for r in recs for x in r["answers"]]
        for role in ("say", None):
            aa = [x for x in a if (role is None or x["role"] == role)]
            if not aa:
                continue
            fr = [x["answer_rank_forced"] for x in aa]; dr = [x["answer_rank_direct"] for x in aa]
            Lines.append(f"  {('say neurons' if role else 'all picked neurons')} ({len(aa)} concept-neuron pairs): answer token in the forced readout's "
                         f"top-1 {sum(x <= 1 for x in fr) / len(fr):.0%}, top-10 {sum(x <= 10 for x in fr) / len(fr):.0%}, top-50 {sum(x <= 50 for x in fr) / len(fr):.0%} "
                         f"(median rank {md(fr)}); weights-only direct push: top-10 {sum(x <= 10 for x in dr) / len(dr):.0%} (median {md(dr)})")
    by_l = defaultdict(list)
    for r in recs:
        by_l[r["layer"] // 4 * 4].append(r)
    if all("context_dependence" in r for r in recs):
        Lines.append(f"  context dependence (1 - mean pairwise Jaccard of per-web-background top-10; 0 = same words in every context): "
                     f"median {md(r['context_dependence'] for r in recs):.2f}")
    Lines.append("  by layer (bins of 4): web-vs-empty top-10 agreement | top word's mean dlogp | top word's consistency (web)")
    for b in sorted(by_l):
        rs = by_l[b]
        Lines.append(f"    L{b:>2}-{b + 3:<2} n={len(rs):<5} agreement {md(r['web_vs_empty_top10_jaccard'] for r in rs):.2f} | strength {md(r['strength'] for r in rs):5.2f} | "
                     f"consistency {md(r['web_words'][0][2] for r in rs):.2f}" + (f" | context dependence {md(r['context_dependence'] for r in rs):.2f}" if all('context_dependence' in r for r in rs) else ""))
    rr = random.Random(5).sample(recs, min(12, len(recs)))
    Lines.append("  examples (neuron: top web words with mean dlogp, consistency):")
    for r in rr:
        Lines.append(f"    {r['neuron']:<10} " + ", ".join(f"{w!r} {d:+.1f} ({c:.0%})" for w, d, c, _ in r["web_words"][:6]))
    txt = "\n".join(Lines)
    (OUT / f"{mode}.txt").write_text(txt + "\n", encoding="utf-8"); print(txt)


if __name__ == "__main__":
    main()
