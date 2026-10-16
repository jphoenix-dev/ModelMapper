"""Which neurons make Qwen answer "Golden" (Gate)? Attribution, then an ablation check.

1. Attribution (activation x gradient): for each recall prompt, the metric is log P(" Golden") at the
   last position. One backward pass gives d(metric)/d(activation) for every neuron at every position;
   activation x gradient, summed over positions, estimates each neuron's contribution. Averaged over prompts.
2. Ablation: zero the top-k positively-attributed neurons (at all positions) and measure P(" Golden")
   on the recall prompts and P(answer) on control prompts with other answers. Baseline: k random
   neurons drawn from the same layers (3 seeds).

Usage: python src/attribute.py <model> [min_p=0.2]   ->  results/attribution_<model>.json
"""
import json
import random
import sys
from pathlib import Path

import torch

from neurons import load, mlps, steer, DEVICE

ROOT = Path(__file__).resolve().parents[1]
name = sys.argv[1]
min_p = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
P = json.loads((ROOT / "prompts/gg_recall.json").read_text(encoding="utf-8"))
model, tok = load(name)
model.requires_grad_(False)
L, W = model.config.num_hidden_layers, model.config.intermediate_size
target = tok.encode(P["target"])[0]


def prob(prompt, tid):
    ids = tok(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        return model(**ids).logits[0, -1].float().softmax(-1)[tid].item()


# keep prompts the unmodified model actually gets right
gg = [p for p in P["gg"] if prob(p, target) >= min_p]
controls = [(c["prompt"], tok.encode(c["answer"])[0]) for c in P["controls"]]
controls = [(p, t) for p, t in controls if prob(p, t) >= 0.5]
print(f"{len(gg)}/{len(P['gg'])} recall prompts kept; {len(controls)} controls kept")

# ---- 1. attribution -------------------------------------------------------
acts = {}
hooks = [model.model.embed_tokens.register_forward_hook(lambda m, a, out: out.requires_grad_(True))]
for l in range(L):
    def keep(m, args, l=l):
        args[0].retain_grad()
        acts[l] = args[0]
    hooks.append(mlps(model)[l].down_proj.register_forward_pre_hook(keep))
attr = torch.zeros(L, W, device=DEVICE)
per_prompt = []
for p in gg:
    ids = tok(p, return_tensors="pt").to(DEVICE)
    with torch.enable_grad():
        lp = model(**ids).logits[0, -1].float().log_softmax(-1)[target]
        lp.backward()
    a = torch.stack([(acts[l].float() * acts[l].grad.float())[0].sum(0) for l in range(L)])  # [L, W]
    attr += a / len(gg)
    per_prompt.append(a)
    acts.clear()
for h in hooks:
    h.remove()

flat = attr.flatten()
order = flat.argsort(descending=True)
pos_total = flat.clamp(min=0).sum().item()
top_ids = [divmod(i.item(), W) for i in order[:500]]
# consistency: in how many prompts is the neuron in that prompt's own top 100?
per_top = [set(a.flatten().topk(100).indices.tolist()) for a in per_prompt]
ranking = []
for r, (l, n) in enumerate(top_ids[:200]):
    i = l * W + n
    ranking.append({"rank": r + 1, "layer": l, "neuron": n, "attribution": round(flat[i].item(), 4),
                    "cum_share_of_positive": round(flat[order[:r + 1]].clamp(min=0).sum().item() / pos_total, 3),
                    "in_prompt_top100": sum(i in s for s in per_top)})

# ---- 2. ablation ----------------------------------------------------------
def mean_probs(edits):
    with steer(model, edits) if edits else torch.no_grad():
        g = sum(prob(p, target) for p in gg) / len(gg)
        c = sum(prob(p, t) for p, t in controls) / len(controls)
    return round(g, 3), round(c, 3)


base = mean_probs([])
curve = []
for k in [1, 2, 5, 10, 20, 50, 100, 200, 500]:
    top = top_ids[:k]
    rnd = []
    for seed in range(3):
        rng = random.Random(seed)
        rnd.append(mean_probs([(l, rng.randrange(W), 0.0) for l, _ in top]))
    row = {"k": k, "top": mean_probs([(l, n, 0.0) for l, n in top]),
           "random": (round(sum(r[0] for r in rnd) / 3, 3), round(sum(r[1] for r in rnd) / 3, 3))}
    curve.append(row)
    print(f"k={k:>4}: ablate top -> P(Golden) {row['top'][0]:.3f}, controls {row['top'][1]:.3f} | "
          f"ablate random -> P(Golden) {row['random'][0]:.3f}, controls {row['random'][1]:.3f}", flush=True)

out = {"model": name, "prompts": gg, "controls": [p for p, _ in controls], "baseline": base,
       "positive_attribution_total": pos_total, "top": ranking, "ablation": curve}
(ROOT / "results" / f"attribution_{name.split('/')[-1]}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(f"\nbaseline: P(Golden) {base[0]:.3f}, controls {base[1]:.3f}")
print("top 15 neurons by attribution:")
for r in ranking[:15]:
    print(f"  #{r['rank']:>2} L{r['layer']}:{r['neuron']:<6} attr {r['attribution']:+.3f}  cum share {r['cum_share_of_positive']:.2f}"
          f"  in top-100 of {r['in_prompt_top100']}/{len(gg)} prompts")
for k in (10, 50, 200):
    print(f"top {k} carry {ranking[k-1]['cum_share_of_positive']:.0%} of all positive attribution")
