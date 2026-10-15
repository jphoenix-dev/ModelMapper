"""Neuron atlas: top-K positive and negative firings for EVERY MLP neuron, in one pass over web text.

For each layer we keep, per neuron, a running leaderboard of its K strongest positive and K most
negative activations, updated inside that layer's down_proj hook (so we never hold all layers'
activations at once). Each entry stores the value, a position code (doc_index * CTX + position) and
the gate's sign. The gate sign is enough to recover the sign regime, because
activation = SiLU(gate) * value and SiLU(gate) < 0 exactly when gate < 0.

Output:
  data/atlas_<model>.pt        top/bottom values (fp16), codes (int32), gate-sign (bool), each [L, W, K]
  data/atlas_<model>_docs.npz  token ids of every scanned document (to render examples on demand)

Usage: python src/build_atlas.py <model> [n_tokens=10e6] [K=40]
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

from neurons import load, mlps
from scan import docs

ROOT = Path(__file__).resolve().parents[1]
CTX = 1024
BATCH_TOKENS = 6_144  # bigger batches spill out of the 32 GB card on 8B

name = sys.argv[1]
n_tokens = int(float(sys.argv[2])) if len(sys.argv) > 2 else 10_000_000
K = int(sys.argv[3]) if len(sys.argv) > 3 else 40
torch.cuda.set_per_process_memory_fraction(0.92)
model, tok = load(name)
L, W = model.config.num_hidden_layers, model.config.intermediate_size
dev = "cuda"

top_v = torch.full((L, W, K), -1e4, device=dev); top_c = torch.zeros((L, W, K), dtype=torch.int32, device=dev)
top_g = torch.zeros((L, W, K), dtype=torch.bool, device=dev)
bot_v = torch.full((L, W, K), 1e4, device=dev); bot_c = torch.zeros((L, W, K), dtype=torch.int32, device=dev)
bot_g = torch.zeros((L, W, K), dtype=torch.bool, device=dev)
state = {}  # current batch: valid mask, position codes, and this layer's gate sign


def gate_hook(l):
    def hook(m, args, out):
        state["gate_pos"] = out[state["valid"]] >= 0           # [N, W] bool
    return hook


def act_hook(l):
    def hook(m, args):
        a = args[0][state["valid"]].float().T                  # [W, N]
        g = state.pop("gate_pos").T                            # [W, N]
        code = state["codes"].expand(W, -1)                    # [W, N]
        for vals, codes, gates, largest in ((top_v, top_c, top_g, True), (bot_v, bot_c, bot_g, False)):
            cv = torch.cat([vals[l], a], 1)
            v, idx = cv.topk(K, dim=1, largest=largest)
            codes[l] = torch.cat([codes[l], code], 1).gather(1, idx)
            gates[l] = torch.cat([gates[l], g], 1).gather(1, idx)
            vals[l] = v
    return hook


handles = []
for l in range(L):
    handles.append(mlps(model)[l].gate_proj.register_forward_hook(gate_hook(l)))
    handles.append(mlps(model)[l].down_proj.register_forward_pre_hook(act_hook(l)))

all_docs, buf, seen, t0, last = [], [], 0, time.time(), 0
gen = docs(tok, CTX)
while seen < n_tokens:
    try:
        buf.append(next(gen))
    except StopIteration:
        break
    if sum(map(len, buf)) < BATCH_TOKENS:
        continue
    T = max(map(len, buf))
    ids = torch.zeros((len(buf), T), dtype=torch.long); mask = torch.zeros_like(ids)
    for i, x in enumerate(buf):
        ids[i, :len(x)] = torch.as_tensor(x, dtype=torch.long); mask[i, :len(x)] = 1
    valid = mask.bool(); valid[:, 0] = False                  # token 0 is an attention sink
    base = len(all_docs)
    codes = (torch.arange(len(buf))[:, None] + base) * CTX + torch.arange(T)[None]
    state["valid"] = valid.to(dev)
    state["codes"] = codes[valid].to(dev, torch.int32)[None]
    with torch.no_grad():
        model.model(input_ids=ids.to(dev), attention_mask=mask.to(dev))
    all_docs.extend(buf)
    seen += int(valid.sum()); buf = []
    if seen - last > 250_000:
        last = seen
        rate = seen / (time.time() - t0)
        print(f"{seen/1e6:.2f}M tokens, {rate:,.0f} tok/s, eta {(n_tokens-seen)/rate/60:.0f} min", flush=True)
for h in handles:
    h.remove()

out = ROOT / "data" / f"atlas_{name.split('/')[-1]}.pt"
torch.save({"model": name, "tokens": seen, "K": K, "ctx": CTX,
            "top_v": top_v.half().cpu(), "top_c": top_c.cpu(), "top_g": top_g.cpu(),
            "bot_v": bot_v.half().cpu(), "bot_c": bot_c.cpu(), "bot_g": bot_g.cpu()}, out)
lengths = np.array([len(d) for d in all_docs], dtype=np.int32)
np.savez_compressed(out.with_name(out.stem + "_docs.npz"), tokens=np.concatenate(all_docs), lengths=lengths)
print(f"done: {seen:,} tokens, {len(all_docs):,} docs, {(time.time()-t0)/60:.1f} min -> {out}")
