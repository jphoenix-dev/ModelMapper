"""How much preceding text does the ablate task need? Window benchmark (GPU; uses a finished observe merge).

For sampled neurons and their top firings (from merge:observe), replay the text ending at the firing token with
windows of W tokens and with the whole document up to that token ("full", what meaning_jobs.py does). Per window:
  - activation ratio: the neuron's activation at the firing token with W tokens / with the full prefix;
  - knockout KL: KL(base || neuron zeroed at that token), as the ablate task, compared with full;
  - cost: tokens processed, relative to full.
Each knockout row shares a forward pass with its unablated copy (as tasks.ablate).
Usage: python src/dist/window_bench.py <store> [--layers 3,10,18,27,34] [--per-layer 32] [--n-ctx 5] [--windows 32,64,128,256]
Writes <store>/bench/window_bench.json and prints a table.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dist.manifest import task_key  # noqa: E402
from dist.storage import open_store  # noqa: E402
from dist.tasks import Ctx, _run_rows  # noqa: E402
from neurons import DEVICE  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("store")
    ap.add_argument("--layers", default=None, help="default: 5 layers spread over the model")
    ap.add_argument("--per-layer", type=int, default=32)
    ap.add_argument("--n-ctx", type=int, default=5)
    ap.add_argument("--windows", default="32,64,128,256")
    ap.add_argument("--batch-rows", type=int, default=32)
    a = ap.parse_args()
    store = open_store(a.store); ctx = Ctx(store, "window_bench"); model = ctx.model
    L, W = model.config.num_hidden_layers, model.config.intermediate_size
    layers = [int(x) for x in a.layers.split(",")] if a.layers else sorted({round(x) for x in np.linspace(2, L - 2, 5)})
    wins = [int(x) for x in a.windows.split(",")] + [10 ** 9]
    toks, starts, lens = ctx.corpus()
    rng = np.random.default_rng(0)
    res = {w: {"act": [], "act_full": [], "kl": [], "kl_full": [], "tokens": 0, "seconds": 0.0} for w in wins}
    for layer in layers:
        m = store.get_npz(f"{task_key('merge:observe')}/L{layer:02d}.npz")
        neurons = rng.choice(W, a.per_layer, replace=False)
        nb = torch.as_tensor(np.sort(neurons), device=DEVICE)
        col = {int(n): i for i, n in enumerate(np.sort(neurons))}
        items = [(int(n), int(m["top_d"][n, j]), int(m["top_p"][n, j])) for n in neurons for j in range(a.n_ctx) if m["top_d"][n, j] >= 0]
        per_w = {}
        for w in wins:
            rows = []
            for n, d, p in items:
                x = toks[starts[d] + max(0, p - w + 1): starts[d] + p + 1]
                rows += [(x, -1), (x, n)]
            acts, kls = [], []
            t0 = time.time()
            for c0 in range(0, len(rows), a.batch_rows):
                chunk = rows[c0:c0 + a.batch_rows]
                logits, _, act, nt = _run_rows(model, layer, chunk, nb)
                lp = logits.log_softmax(-1)
                for k in range(0, len(chunk), 2):
                    n = chunk[k + 1][1]
                    acts.append(act[k, col[n]].item())
                    kls.append((lp[k].exp() * (lp[k] - lp[k + 1])).sum().item())
                res[w]["tokens"] += nt
            if DEVICE.startswith("cuda"):
                torch.cuda.synchronize()
            res[w]["seconds"] += time.time() - t0
            per_w[w] = (acts, kls)
        for w in wins:
            res[w]["act"] += per_w[w][0]; res[w]["kl"] += per_w[w][1]
            res[w]["act_full"] += per_w[wins[-1]][0]; res[w]["kl_full"] += per_w[wins[-1]][1]
        print(f"layer {layer}: {len(items)} firings", flush=True)
    out = {"layers": layers, "per_layer": a.per_layer, "n_ctx": a.n_ctx, "rows": {}}
    full_tok = res[wins[-1]]["tokens"]
    print(f"{'window':>8} {'act ratio med':>14} {'>=0.8 of full':>14} {'>=0.5':>7} {'KL spearman':>12} {'KL ratio med':>13} {'cost':>6}")
    for w in wins:
        r = res[w]
        act, af = np.array(r["act"]), np.array(r["act_full"]); kl, kf = np.array(r["kl"]), np.array(r["kl_full"])
        ratio = act / np.where(np.abs(af) > 1e-6, af, np.nan)
        sp = float(np.corrcoef(np.argsort(np.argsort(kl)), np.argsort(np.argsort(kf)))[0, 1])
        row = {"act_ratio_median": float(np.nanmedian(ratio)), "share_ge_0.8": float(np.nanmean(ratio >= 0.8)),
               "share_ge_0.5": float(np.nanmean(ratio >= 0.5)), "kl_spearman_vs_full": sp,
               "kl_ratio_median": float(np.median(kl / np.maximum(kf, 1e-12))), "tokens": r["tokens"],
               "cost_vs_full": r["tokens"] / full_tok, "seconds": round(r["seconds"], 1), "n": len(act)}
        out["rows"]["full" if w == wins[-1] else str(w)] = row
        print(f"{'full' if w == wins[-1] else w:>8} {row['act_ratio_median']:>14.3f} {row['share_ge_0.8']:>14.2f} {row['share_ge_0.5']:>7.2f} "
              f"{sp:>12.3f} {row['kl_ratio_median']:>13.3f} {row['cost_vs_full']:>6.2f}")
    store.put_json("bench/window_bench.json", {**out, "env": ctx.env, "model": ctx.model_meta})


if __name__ == "__main__":
    main()
