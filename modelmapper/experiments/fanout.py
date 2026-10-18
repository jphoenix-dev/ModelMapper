"""Fan-out readout for early / mid neurons: which LATE neurons (and words) move when only this neuron is varied?
(GPU; no API) - smoke test

James (18 Oct): the forced word readout works only for late neurons; describe early / mid ones by what they drive.
Hold the context fixed (identical input), vary ONLY the target through a dose series d x sign x its web-text peak,
d in {-4, -2, -1, 0, 1, 2, 4, 8} (0 = zeroed), applied either at the last position ('last') or at every position
('all'). Read at the last position: every neuron of layers max(L+1, 24)..35, and the next-token log-probs.
Backgrounds: 16 random web snippets (two halves of 8, for repeatability); for concept-specific targets also the
concept's own recall prompts ('own').
Per late neuron: least-squares slope of its activation vs dose in each background, R^2 of the linear fit;
fan-out = late neurons by |mean slope|; sign consistency across backgrounds; split-half: Jaccard of the top-20 from
the two halves of the web backgrounds. Words: slope of next-token log-prob vs dose. v2 (after the smoke test: a few generic last-layer hub neurons topped every list): late neurons ranked by the slope in
units of their own natural spread (std across backgrounds at dose 0), minus the random targets' mean for the same late
neuron ('specific' fan-out). Each top late neuron is described by
the 5 tokens its weights push (logit lens, in the direction the slope drives it).
Output: D:/already-clean-data/fanout/smoke.jsonl, smoke.txt.  Usage: python src/loop_v2/fanout.py
"""
import glob
import json
import os
import random
import statistics as stx
import sys
from pathlib import Path

import numpy as np
import torch

from modelmapper.core.config import DEFAULT_MODEL  # noqa: E402
from modelmapper.core.neurons import DEVICE, load, mlps  # noqa: E402
from modelmapper.core.settings import DATA_DIR, REPO_ROOT

ROOT = REPO_ROOT
OUT = DATA_DIR / "fanout"
DOSES = [-4, -2, -1, 0, 1, 2, 4, 8]; NWEB, CTX, LATE = 16, 24, 24
SPECIFIC = [("L12:2572", -1, "Clovis I"), ("L4:2016", 1, "Strawberry"), ("L8:7216", 1, "Volleyball"), ("L13:9500", 1, "Johann Strauss II"),
            ("L7:5203", -1, "Oedipus Rex"), ("L9:8962", 1, "Monopoly (game)"), ("L10:9119", 1, "Marathi language"),
            ("L9:5260", -1, "In Search of Lost Time"), ("L9:8207", 1, "Antigonus I Monophthalmus"), ("L5:3638", -1, "Florence Griffith Joyner"),
            ("L10:8168", 1, "Simple machine")]
SHARED = [("L20:7473", -1), ("L19:9651", 1), ("L23:12108", 1), ("L20:301", -1), ("L20:2258", -1), ("L22:2516", 1), ("L23:2192", -1), ("L16:9154", 1)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model, tok = load(DEFAULT_MODEL); L, W = model.config.num_hidden_layers, model.config.intermediate_size
    st = torch.load(ROOT / f"data/stats_{DEFAULT_MODEL.split('/')[-1]}.pt", map_location="cpu", weights_only=False)
    smax, smin = st["max"], st["min"]
    WU = model.lm_head.weight.detach(); g = model.model.norm.weight.detach().float()
    rng = random.Random(141)
    prompts = {}
    for b in glob.glob((DATA_DIR.as_posix() + "/loop_v2/run1/results/**/map/*/records.jsonl"), recursive=True):
        for line in Path(b).read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["row"]["status"] == "ok":
                prompts[r["concept"]] = r["rank"]["prompts"]
    targets = [(u, sg, "specific", c) for u, sg, c in SPECIFIC] + [(u, sg, "shared", None) for u, sg in SHARED]
    targets = [(f"L{l}:{rng.randrange(W)}", None, "random", None) for l in rng.sample(range(4, 24), 10)] + targets   # random first: baseline
    base_z = {}
    d = np.load(ROOT / f"data/atlas_{DEFAULT_MODEL.split('/')[-1]}_docs.npz"); toks, lens = d["tokens"], d["lengths"]
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]]); okd = np.where(lens > CTX + 1)[0]
    web = []
    for _ in range(NWEB):
        di = int(okd[rng.randrange(len(okd))]); p = rng.randrange(1, int(lens[di]) - CTX)
        web.append([int(t) for t in toks[starts[di] + p: starts[di] + p + CTX]])
    dz = torch.tensor(DOSES, dtype=torch.float32, device=DEVICE)

    def push_tokens(l, n, sgn):
        v = (sgn * model.model.layers[l].mlp.down_proj.weight[:, n].detach().float() * g).to(WU.dtype)
        return [tok.decode([i]) for i in (WU @ v).float().topk(5).indices.tolist()]

    def sweep(seqs, l, n, val_unit, mode):
        """returns slopes [nb, nlate, W], r2 [nb, nlate, W], word slopes [nb, V]; rows = doses x backgrounds."""
        nb = len(seqs); D = len(DOSES); T = max(map(len, seqs)); pad = tok.pad_token_id or 0
        x = torch.full((nb, T), pad, device=DEVICE); mask = torch.zeros_like(x)
        for i, s in enumerate(seqs):
            x[i, T - len(s):] = torch.tensor(s, device=DEVICE); mask[i, T - len(s):] = 1
        x = x.repeat(D, 1); mask = mask.repeat(D, 1)                     # row = dose * nb + background
        vals = (dz * val_unit).repeat_interleave(nb)
        startpos = [T - len(s) for s in seqs] * D
        late = list(range(max(l + 1, LATE), L)); cap = {}

        def ht(mod, a):
            y = a[0].clone()
            if mode == "last":
                y[:, -1, n] = vals.to(y.dtype)
            else:
                for r_, p0 in enumerate(startpos):
                    y[r_, p0:, n] = vals[r_].to(y.dtype)
            return (y,)
        hks = [mlps(model)[l].down_proj.register_forward_pre_hook(ht)]
        hks += [mlps(model)[ll].down_proj.register_forward_pre_hook(lambda mod, a, ll=ll: cap.__setitem__(ll, a[0][:, -1].float())) for ll in late]
        try:
            with torch.no_grad():
                lp = model(input_ids=x, attention_mask=mask).logits[:, -1].float().log_softmax(-1)
        finally:
            for h in hks:
                h.remove()
        A = torch.stack([cap[ll] for ll in late], 1).view(D, nb, len(late), W)          # [D, nb, nl, W]
        dd = dz - dz.mean(); den = (dd ** 2).sum()
        Am = A - A.mean(0, keepdim=True)
        slope = torch.einsum("d,dbnw->bnw", dd, Am) / den
        ss_tot = (Am ** 2).sum(0) + 1e-9; ss_res = ((Am - slope[None] * dd[:, None, None, None]) ** 2).sum(0)
        r2 = 1 - ss_res / ss_tot
        LP = lp.view(D, nb, -1); LPm = LP - LP.mean(0, keepdim=True)
        wslope = torch.einsum("d,dbv->bv", dd, LPm) / den
        sd0 = A[DOSES.index(0)].std(0)                                              # [nl, W] natural spread at dose 0
        return late, slope, r2, wslope, sd0
    tag = os.environ.get("FANOUT_TAG", "smoke2")
    f = open(OUT / f"{tag}.jsonl", "w", encoding="utf-8"); res = []
    for u, sg, kind, concept in targets:
        l, n = (int(x) for x in u[1:].split(":"))
        if sg is None:
            sg = 1.0 if smax[l, n] >= -smin[l, n] else -1.0
        pk = float(smax[l, n] if sg > 0 else -smin[l, n]); unit = sg * pk
        sets = {"web": web}
        if concept and concept in prompts:
            sets["own"] = [tok.encode(p) for p in prompts[concept][:8]]
        for mode in ("last", "all"):
            for sname, seqs in sets.items():
                late, slope, r2, ws, sd0 = sweep(seqs, l, n, unit, mode)
                z = slope.mean(0) / (sd0 + 1e-3)                                       # slope per dose unit, in natural-spread units
                if kind == "random":
                    base_z.setdefault((mode, sname, late[0]), []).append(z.abs())
                ms = slope.mean(0)
                bz = base_z.get((mode, sname, late[0]))
                spec_score = z.abs() - (torch.stack(bz).mean(0) if bz and kind != "random" else 0)
                flat = ms.abs().flatten(); top = spec_score.flatten().topk(20).indices   # ranked by SPECIFIC normalised fan-out
                consist = ((slope.sign() == ms.sign()[None]).float().mean(0)).flatten()
                fan = []
                for i in top.tolist():
                    li, ni = divmod(i, W); ll = late[li]; s_ = float(ms.flatten()[i])
                    fan.append({"neuron": f"L{ll}:{ni}", "slope": round(s_, 4), "z": round(float(z.flatten()[i]), 3), "specific": round(float(spec_score.flatten()[i]), 3), "sign_consistency": round(float(consist[i]), 3),
                                "r2_median": round(float(r2[:, li, ni].median()), 3), "push": push_tokens(ll, ni, 1.0 if s_ > 0 else -1.0)})
                split = None
                if sname == "web":
                    h1 = set(slope[: NWEB // 2].mean(0).abs().flatten().topk(20).indices.tolist())
                    h2 = set(slope[NWEB // 2:].mean(0).abs().flatten().topk(20).indices.tolist())
                    split = len(h1 & h2) / len(h1 | h2)
                wm = ws.mean(0); up = wm.topk(10); dn = (-wm).topk(10)
                rec = {"target": u, "kind": kind, "concept": concept, "mode": mode, "set": sname, "sign": sg, "peak": round(pk, 3),
                       "fanout_top20": fan, "split_half_jaccard": split, "fan_strength": round(float(flat.max()), 4),
                       "words_up": [[tok.decode([i]), round(float(v), 3)] for i, v in zip(up.indices.tolist(), up.values.tolist())],
                       "words_down": [[tok.decode([i]), round(float(v), 3)] for i, v in zip(dn.indices.tolist(), (-dn.values).tolist())]}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush(); res.append(rec)
        print(f"done {u} ({kind})", flush=True)
    md = stx.median
    Lines = ["Fan-out smoke test: early/mid neurons varied alone (doses -4..8 x peak), late neurons + words read at the last position"]
    for kind in ("specific", "shared", "random"):
        for mode in ("last", "all"):
            rs = [r for r in res if r["kind"] == kind and r["mode"] == mode and r["set"] == "web"]
            if rs:
                Lines.append(f"  {kind:<8} {mode:<4} (web): split-half top-20 Jaccard median {md(r['split_half_jaccard'] for r in rs):.2f}; top fan-out "
                             f"neurons' sign consistency median {md(x['sign_consistency'] for r in rs for x in r['fanout_top20']):.2f}, linear R^2 median "
                             f"{md(x['r2_median'] for r in rs for x in r['fanout_top20']):.2f}; strength median {md(r['fan_strength'] for r in rs):.3f}")
    Lines.append("\nper target (mode all; web, and own prompts where available): top-5 driven late neurons (slope, consistency, R^2, pushed tokens) | words up")
    for r in res:
        if r["mode"] != "all":
            continue
        Lines.append(f"  {r['target']:<10} {r['kind']:<8} {(r['concept'] or '')[:22]:<22} [{r['set']}] split {r['split_half_jaccard'] if r['split_half_jaccard'] is not None else '-'}")
        for x in r["fanout_top20"][:5]:
            Lines.append(f"      -> {x['neuron']:<10} z {x['z']:+.2f} (specific {x['specific']:+.2f}; {x['sign_consistency']:.0%}, R2 {x['r2_median']:.2f}) {x['push']}")
        Lines.append(f"      words up: {[w for w, _ in r['words_up'][:8]]}")
    txt = "\n".join(Lines)
    (OUT / f"{tag}.txt").write_text(txt + "\n", encoding="utf-8"); print(txt)


if __name__ == "__main__":
    main()
