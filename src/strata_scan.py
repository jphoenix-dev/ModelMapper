"""One pass over web text collecting (a) top/bottom firings for planted mixtures and (b) middle-of-range
samples for chosen neurons.

(a) Planted mixtures: two random neurons from the same layer, blended with weights 1/peak each, so both
    halves contribute equally. A unit we *know* has two unrelated jobs: the census's leniency check.
(b) Strata: for each chosen neuron, a random sample (reservoir) of firings whose activation falls in bands
    around 50%, 25% and 10% of its peak (positive pole, gate open). Tests whether a neuron's main
    explanation still fits below its top firings ("meaning radiates outward") or turns into a jumble.

Output JSON has the same shape as scan.py's ("units" -> {"pos": [...], "neg": [...]}) for mixtures, plus
"strata" -> {neuron_id: {"50": [...], "25": [...], "10": [...]}} with rendered text, so label_batch.py
can read both.

Usage: python src/strata_scan.py <model> <census.json> <out.json> [n_tokens=10e6] [n_mix=60] [n_strata=100]
"""
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from atlas import Atlas
from neurons import load, mlps, DEVICE, IS_CUDA
from scan import docs

ROOT = Path(__file__).resolve().parents[1]
CTX, K, R = 1024, 40, 15
BANDS = {"50": (0.40, 0.60), "25": (0.20, 0.30), "10": (0.08, 0.12)}


def main():
    model_name, census_path, out_path = sys.argv[1:4]
    n_tokens = int(float(sys.argv[4])) if len(sys.argv) > 4 else 10_000_000
    n_mix = int(sys.argv[5]) if len(sys.argv) > 5 else 60
    n_strata = int(sys.argv[6]) if len(sys.argv) > 6 else 100
    if IS_CUDA:
        torch.cuda.set_per_process_memory_fraction(0.92)
    model, tok = load(model_name)
    L, W = model.config.num_hidden_layers, model.config.intermediate_size
    stats = torch.load(ROOT / f"data/stats_{model_name.split('/')[-1]}.pt")
    rng = random.Random(7)

    # (a) planted mixtures
    mixes = []
    for _ in range(n_mix):
        l = rng.randrange(L); a, b = rng.sample(range(W), 2)
        mixes.append({"id": f"mix_L{l}_N{a}_N{b}", "layer": l, "weights": {a: 1 / max(stats["max"][l, a].item(), 1e-3),
                                                                           b: 1 / max(stats["max"][l, b].item(), 1e-3)}})
    # (b) strata neurons: a random subset of the census
    census = json.loads(Path(census_path).read_text(encoding="utf-8"))["units"]
    strata_ids = rng.sample(sorted(census), min(n_strata, len(census)))
    strata = [(census[u]["layer"], census[u]["neuron"]) for u in strata_ids]
    at = Atlas(model_name)
    peak = {(l, n): at.a["top_v"][l, n, 0].float().item() for l, n in strata}   # same 10M-token corpus => same peak

    layers = sorted({m["layer"] for m in mixes} | {l for l, _ in strata})
    mix_by_layer = {l: [m for m in mixes if m["layer"] == l] for l in layers}
    Wm = {}
    for l, ms in mix_by_layer.items():
        if ms:
            M = torch.zeros(W, len(ms))
            for j, m in enumerate(ms):
                for n, w in m["weights"].items():
                    M[n, j] = w
            Wm[l] = M.to(DEVICE)
    st_by_layer = {l: [(i, n) for i, (ll, n) in enumerate(strata) if ll == l] for l in layers}

    top_v = {m["id"]: [] for m in mixes}
    bot_v = {m["id"]: [] for m in mixes}
    reservoirs = {(i, b): [] for i in range(len(strata)) for b in BANDS}
    seen_band = {(i, b): 0 for i in range(len(strata)) for b in BANDS}
    state = {}

    def gate_hook(l):
        def hook(m, args, out):
            state["gpos"] = out[state["valid"]] >= 0
        return hook

    def act_hook(l):
        def hook(m, args):
            a = args[0][state["valid"]].float()                 # [N, W]
            g = state.pop("gpos")
            codes = state["codes"]
            if l in Wm:
                proj = a @ Wm[l]                                 # [N, n_mix_in_layer]
                for j, m_ in enumerate(mix_by_layer[l]):
                    v = proj[:, j]
                    for store, largest in ((top_v, True), (bot_v, False)):
                        k = min(K, v.numel())
                        tv, ti = v.topk(k, largest=largest)
                        store[m_["id"]].extend(zip(tv.tolist(), codes[ti].tolist()))
                        store[m_["id"]].sort(key=lambda x: -x[0] if largest else x[0])
                        del store[m_["id"]][K:]
            for i, n in st_by_layer.get(l, []):
                pk = peak[(strata[i][0], n)]
                col, gcol = a[:, n], g[:, n]
                for b, (lo, hi) in BANDS.items():
                    mask = gcol & (col >= lo * pk) & (col <= hi * pk)
                    cnt = int(mask.sum())
                    if not cnt:
                        continue
                    seen_band[(i, b)] += cnt
                    # uniform random sample: give every candidate a random key, keep the R largest keys overall
                    keys = torch.rand(cnt, device=col.device)
                    kk, kv, kc = keys, col[mask], codes[mask]
                    old = reservoirs[(i, b)]
                    if old:
                        ok, ov, oc = (torch.tensor(x, device=col.device) for x in zip(*old))
                        kk, kv, kc = torch.cat([ok, kk]), torch.cat([ov, kv]), torch.cat([oc, kc])
                    top = kk.topk(min(R, kk.numel())).indices
                    reservoirs[(i, b)] = list(zip(kk[top].tolist(), kv[top].tolist(), kc[top].tolist()))
        return hook

    hooks = []
    for l in layers:
        hooks.append(mlps(model)[l].gate_proj.register_forward_hook(gate_hook(l)))
        hooks.append(mlps(model)[l].down_proj.register_forward_pre_hook(act_hook(l)))
    all_docs, buf, seen, t0, last = [], [], 0, time.time(), 0
    gen = docs(tok, CTX)
    while seen < n_tokens:
        try:
            buf.append(next(gen))
        except StopIteration:
            break
        if sum(map(len, buf)) < 6144:
            continue
        T = max(map(len, buf))
        ids = torch.zeros((len(buf), T), dtype=torch.long); mask = torch.zeros_like(ids)
        for i, x in enumerate(buf):
            ids[i, :len(x)] = torch.as_tensor(x, dtype=torch.long); mask[i, :len(x)] = 1
        valid = mask.bool(); valid[:, 0] = False
        base = len(all_docs)
        codes = ((torch.arange(len(buf))[:, None] + base) * CTX + torch.arange(T)[None])[valid]
        state["valid"], state["codes"] = valid.to(DEVICE), codes.to(DEVICE)
        with torch.no_grad():
            model.model(input_ids=ids.to(DEVICE), attention_mask=mask.to(DEVICE))
        all_docs.extend(buf); seen += int(valid.sum()); buf = []
        if seen - last > 500_000:
            last = seen
            r = seen / (time.time() - t0)
            print(f"{seen/1e6:.1f}M tokens, {r:,.0f} tok/s, eta {(n_tokens-seen)/r/60:.0f} min", flush=True)
    for h in hooks:
        h.remove()

    def render(code, before=200, after=12):
        d, pos = code // CTX, code % CTX
        ids = [int(t) for t in all_docs[d]]
        lo, hi = max(0, pos - before), min(len(ids), pos + after)
        pieces, prev = [], ""
        for p in range(lo, hi):
            cur = tok.decode(ids[lo:p + 1]); t, prev = cur[len(prev):], cur
            pieces.append(f"[[{t}]]" if p == pos else t)
        return "".join(pieces)

    out = {"model": model_name, "tokens_scanned": seen, "units": {}, "strata": {}}
    for m in mixes:
        out["units"][m["id"]] = {"unit": {"id": m["id"], "layer": m["layer"], "weights": {str(k): v for k, v in m["weights"].items()}},
                                 "pos": [{"value": round(v, 4), "text": render(c)} for v, c in top_v[m["id"]]],
                                 "neg": [{"value": round(v, 4), "text": render(c)} for v, c in bot_v[m["id"]]]}
    for i, (l, n) in enumerate(strata):
        out["strata"][f"L{l}_N{n}"] = {"peak": peak[(l, n)],
                                      **{b: [{"value": round(v, 4), "text": render(int(c))} for _, v, c in reservoirs[(i, b)]]
                                         for b in BANDS},
                                      "band_counts": {b: seen_band[(i, b)] for b in BANDS}}
    Path(out_path).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"done: {seen:,} tokens, {len(mixes)} mixtures, {len(strata)} strata neurons, {(time.time()-t0)/60:.1f} min -> {out_path}")


if __name__ == "__main__":
    main()
