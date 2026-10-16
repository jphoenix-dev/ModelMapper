"""Every neuron's normal range on web text: its max and min.

Needed so we can ask "how hard does this neuron fire on Golden Gate texts *relative to its own
usual peaks*?" A neuron that reaches 5% of its max on our texts isn't a Golden Gate neuron, however
selective it looks against the controls.

The running max/min is updated inside each layer's hook, so we never hold all layers'
activations at once (on Qwen3-8B that would be ~14 GB per batch).

Usage: python src/neuron_stats.py <model> [n_tokens=2e6]   ->  data/stats_<model>.pt
"""
import sys
import time
from pathlib import Path

import torch

from neurons import load, mlps, DEVICE
from scan import docs

ROOT = Path(__file__).resolve().parents[1]
name = sys.argv[1]
n_tokens = int(float(sys.argv[2])) if len(sys.argv) > 2 else 2_000_000
model, tok = load(name)
L, W = model.config.num_hidden_layers, model.config.intermediate_size
mx = torch.full((L, W), -1e9, device=DEVICE)
mn = torch.full((L, W), 1e9, device=DEVICE)
state = {}  # current batch's valid-token mask, read by the hooks


def make_hook(l):
    def hook(m, args):
        a = args[0][state["valid"]].float()        # [n_valid, W]
        mx[l] = torch.maximum(mx[l], a.max(0).values)
        mn[l] = torch.minimum(mn[l], a.min(0).values)
    return hook


handles = [mlps(model)[l].down_proj.register_forward_pre_hook(make_hook(l)) for l in range(L)]
seen, buf, t0 = 0, [], time.time()
gen = docs(tok, 1024)
while seen < n_tokens:
    buf.append(next(gen))
    if sum(map(len, buf)) < 8192:
        continue
    T = max(map(len, buf))
    ids = torch.zeros((len(buf), T), dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, x in enumerate(buf):
        ids[i, :len(x)] = torch.as_tensor(x, dtype=torch.long); mask[i, :len(x)] = 1
    valid = mask.bool().to(DEVICE); valid[:, 0] = False  # token 0 is an attention sink
    state["valid"] = valid
    with torch.no_grad():
        model.model(input_ids=ids.to(DEVICE), attention_mask=mask.to(DEVICE))
    seen += int(valid.sum()); buf = []
    if seen // 8192 % 50 == 0:
        print(f"{seen/1e6:.2f}M tokens, {seen/(time.time()-t0):,.0f} tok/s", flush=True)
for h in handles:
    h.remove()
print(f"{seen:,} tokens in {time.time()-t0:.0f}s")
out = ROOT / "data" / f"stats_{name.split('/')[-1]}.pt"
out.parent.mkdir(exist_ok=True)
torch.save({"max": mx.cpu(), "min": mn.cpu(), "tokens": seen}, out)
print("saved", out)
