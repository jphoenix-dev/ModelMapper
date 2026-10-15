"""Smoke test: load Qwen3-0.6B, read neurons, check the hook is faithful, try steering."""
import sys
import time

import torch

from neurons import capture, load, mlps, steer

name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B"
t = time.time()
model, tok = load(name)
cfg = model.config
print(f"{name}: {cfg.num_hidden_layers} layers x {cfg.intermediate_size} neurons "
      f"= {cfg.num_hidden_layers * cfg.intermediate_size:,} neurons; loaded in {time.time()-t:.1f}s")

text = "We walked across the Golden Gate Bridge in the fog."
ids = tok(text, return_tensors="pt").to("cuda")
with torch.no_grad(), capture(model, gate_value=True) as s:
    logits = model(**ids).logits

# 1. Faithfulness: SiLU(gate) * value should equal what down_proj received.
L = cfg.num_hidden_layers // 2
recon = torch.nn.functional.silu(s["gate"][L]) * s["value"][L]
rel = (recon - s["act"][L]).abs().max() / s["act"][L].abs().max()
print(f"hook check: max diff is {rel.item():.2%} of max activation (bf16 rounding; should be < 1%)")

# 2. Strongest neurons on the token " Gate"
toks = tok.convert_ids_to_tokens(ids.input_ids[0])
pos = next(i for i, t_ in enumerate(toks) if "Gate" in t_)
best = []
for layer, a in s["act"].items():
    v, n = a[0, pos].topk(3)
    best += [(v_.item(), layer, n_.item()) for v_, n_ in zip(v, n)]
best.sort(reverse=True)
print(f"tokens: {toks}")
print("top neurons on ' Gate' (activation, layer, neuron):", [(round(a, 2), l, n) for a, l, n in best[:8]])

# 3. Steering plumbing: zeroing a whole layer's neurons should change the prediction.
nxt = lambda lg: tok.decode(lg[0, -1].argmax())
with torch.no_grad(), steer(model, [(L, n, 0.0) for n in range(cfg.intermediate_size)]):
    ablated = model(**ids).logits
print("next-token guess normal:", repr(nxt(logits)), "| with layer", L, "MLP zeroed:", repr(nxt(ablated)))
print("peak GPU memory: %.1f GB" % (torch.cuda.max_memory_allocated() / 1e9))
