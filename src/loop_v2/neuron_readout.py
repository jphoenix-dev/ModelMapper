"""What does Qwen say when ONE neuron is forced high? A low-bias readout of a neuron's most common expression.
(GPU; no API; no labels)

James (18 Oct): like the original "Eiff eiff eiff" when a say-it neuron was clamped hard - garbage is fine; what
repeats across many different backgrounds is the neuron's own expression, not the context's.
Backgrounds (each a batch of repeats):
  web      200 random 24-token snippets of web text (the atlas documents), real context
  noise    a single newline token, with 50 RANDOM unrelated neurons (any layer) clamped to their own peak (the larger of
           their web-text max / |min|, in that sign) at every position - a different random set per repeat (200)
  nothing  a single newline token, nothing else (1 repeat)
Target: clamped to sign x m x its own peak, m in {1, 4, 16}, at the last context position and every generated
position (the context itself is read normally, apart from the noise clamps). Sign = the sign it takes for the concepts
that use it (maps); peak from data/stats_<model>.pt.
Read out: (1) next token - mean change in log-probability vs the same background unforced, and how often each token is
the forced argmax; (2) 8 greedy tokens with the clamp held - token frequency over all repeats. Controls: random target
neurons from the same layers, same procedure; 'specific' = target frequency minus the controls' mean.
Output: D:/already-clean-data/neuron_readout/<tag>.json (everything), .txt.  Usage: python src/loop_v2/neuron_readout.py <tag> <L:n[:sign]> ...
"""
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DEFAULT_MODEL  # noqa: E402
from neurons import DEVICE, load, mlps  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("AC_DATA_DIR", "D:/already-clean-data")) / "neuron_readout"
NREP, CTX, NOISE, STEPS, MULTS = 200, 24, 50, 8, (1, 4, 16)


def main():
    tag = sys.argv[1]; specs = sys.argv[2:]
    OUT.mkdir(parents=True, exist_ok=True)
    model, tok = load(DEFAULT_MODEL); L, W = model.config.num_hidden_layers, model.config.intermediate_size
    st = torch.load(ROOT / f"data/stats_{DEFAULT_MODEL.split('/')[-1]}.pt", map_location="cpu", weights_only=False)
    smax, smin = st["max"], st["min"]
    rng = random.Random(121)
    targets = []
    for s in specs:
        parts = s.split(":"); l, n = int(parts[0][1:]), int(parts[1])
        sg = float(parts[2]) if len(parts) > 2 else (1.0 if smax[l, n] >= -smin[l, n] else -1.0)
        targets.append((f"L{l}:{n}", l, n, sg, "target"))
    for l in sorted({t[1] for t in targets})[:3]:
        n = rng.randrange(W); targets.append((f"L{l}:{n}", l, n, 1.0 if smax[l, n] >= -smin[l, n] else -1.0, "control"))
    peak = lambda l, n, sg: float(smax[l, n] if sg > 0 else -smin[l, n])
    d = np.load(ROOT / f"data/atlas_{DEFAULT_MODEL.split('/')[-1]}_docs.npz"); toks, lens = d["tokens"], d["lengths"]
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]]); ok = np.where(lens > CTX + 1)[0]
    web = []
    for _ in range(NREP):
        di = int(ok[rng.randrange(len(ok))]); p = rng.randrange(1, int(lens[di]) - CTX)
        web.append([int(t) for t in toks[starts[di] + p: starts[di] + p + CTX]])
    nl = tok.encode("\n")
    noise_sets = []
    for _ in range(NREP):
        units = []
        for _ in range(NOISE):
            l = rng.randrange(L); n = rng.randrange(W); sg = 1.0 if smax[l, n] >= -smin[l, n] else -1.0
            units.append((l, n, sg * peak(l, n, sg)))
        noise_sets.append(units)
    backgrounds = {"web": (web, None), "noise": ([nl] * NREP, noise_sets), "nothing": ([nl], None)}

    def run(seqs, noise, target, m):
        """Greedy STEPS tokens; returns (first-step log-probs [B, V] float32 cpu, generated ids [B, STEPS])."""
        B = len(seqs); ctx_len = [len(s) for s in seqs]; cur = [list(s) for s in seqs]; first = None; gen = []
        for step in range(STEPS):
            T = max(map(len, cur)); pad = tok.pad_token_id or 0
            x = torch.full((B, T), pad, device=DEVICE); mask = torch.zeros_like(x)
            for i, s in enumerate(cur):                                   # left padding: last token at -1
                x[i, T - len(s):] = torch.tensor(s, device=DEVICE); mask[i, T - len(s):] = 1
            hks = []
            if noise is not None:
                by = {}
                for i, units in enumerate(noise):
                    for (l, n, v) in units:
                        by.setdefault(l, []).append((i, n, v))
                for l, items in by.items():
                    rows = torch.tensor([a for a, _, _ in items], device=DEVICE); idx = torch.tensor([b for _, b, _ in items], device=DEVICE)
                    vals = torch.tensor([c for _, _, c in items], device=DEVICE)

                    def h(mod, a, rows=rows, idx=idx, vals=vals):
                        y = a[0].clone(); y[rows, :, idx] = vals[:, None].to(y.dtype); return (y,)
                    hks.append(mlps(model)[l].down_proj.register_forward_pre_hook(h))
            if target is not None:
                _, tl, tn, tsg, _ = target; tv = tsg * m * peak(tl, tn, tsg)
                startpos = [T - len(s) + ctx_len[i] - 1 for i, s in enumerate(cur)]   # last context position onwards

                def ht(mod, a, startpos=startpos, tn=tn, tv=tv):
                    y = a[0].clone()
                    for i, p0 in enumerate(startpos):
                        y[i, p0:, tn] = tv
                    return (y,)
                hks.append(mlps(model)[tl].down_proj.register_forward_pre_hook(ht))
            try:
                with torch.no_grad():
                    lg = model(input_ids=x, attention_mask=mask).logits[:, -1].float()
            finally:
                for hk in hks:
                    hk.remove()
            if step == 0:
                first = lg.log_softmax(-1).cpu()
            nxt = lg.argmax(-1).tolist(); gen.append(nxt)
            for i, t in enumerate(nxt):
                cur[i].append(t)
        return first, torch.tensor(gen).T
    res = {"targets": [t[0] for t in targets], "readouts": {}}
    base = {k: run(seqs, noise, None, 0) for k, (seqs, noise) in backgrounds.items()}
    for t in targets:
        for m in MULTS:
            for k, (seqs, noise) in backgrounds.items():
                lp, g = run(seqs, noise, t, m)
                dlp = (lp - base[k][0]).mean(0)
                top_d = dlp.topk(15)
                am = Counter(lp.argmax(-1).tolist())
                gc = Counter(g.flatten().tolist())
                res["readouts"][f"{t[0]}|{m}|{k}"] = {
                    "kind": t[4], "sign": t[3], "value": t[3] * m * peak(t[1], t[2], t[3]),
                    "next_mean_dlogp_top": [[tok.decode([i]), round(float(v), 3)] for i, v in zip(top_d.indices.tolist(), top_d.values.tolist())],
                    "next_argmax_top": [[tok.decode([i]), c] for i, c in am.most_common(10)],
                    "gen_token_counts": [[tok.decode([i]), c] for i, c in gc.most_common(30)],
                    "gen_examples": [tok.decode(row) for row in g[:4].tolist()]}
        print(f"done {t[0]} ({t[4]})", flush=True)
    # specific = target generation frequency minus the controls' mean, per (m, background)
    L_ = []
    for m in MULTS:
        for k in backgrounds:
            ctrl = [Counter(dict((w, c) for w, c in res["readouts"][f"{t[0]}|{m}|{k}"]["gen_token_counts"])) for t in targets if t[4] == "control"]
            cm = Counter()
            for c in ctrl:
                for w, v in c.items():
                    cm[w] += v / max(1, len(ctrl))
            for t in targets:
                if t[4] != "target":
                    continue
                r = res["readouts"][f"{t[0]}|{m}|{k}"]
                spec = sorted(((w, c - cm.get(w, 0)) for w, c in r["gen_token_counts"]), key=lambda x: -x[1])[:8]
                r["gen_specific"] = [[w, round(v, 1)] for w, v in spec]
    for t in targets:
        L_.append(f"\n=== {t[0]} ({t[4]}, sign {t[3]:+.0f}, peak {peak(t[1], t[2], t[3]):.1f}) ===")
        for m in MULTS:
            for k in backgrounds:
                r = res["readouts"][f"{t[0]}|{m}|{k}"]
                L_.append(f"  x{m:<2} {k:<7} next+: {[w for w, _ in r['next_mean_dlogp_top'][:8]]} | argmax: {r['next_argmax_top'][:4]} | "
                          f"gen: {r.get('gen_specific', r['gen_token_counts'][:8])} | e.g. {r['gen_examples'][0]!r}")
    txt = "\n".join(L_)
    (OUT / f"{tag}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    (OUT / f"{tag}.txt").write_text(txt + "\n", encoding="utf-8"); print(txt)


if __name__ == "__main__":
    main()
