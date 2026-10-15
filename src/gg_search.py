"""Step 1 of Experiment GG: contrastive search for Golden Gate neurons.

For every text we take each neuron's strongest firing (max over tokens, positive pole)
and its most negative firing (min over tokens, negative pole). A neuron is a candidate if
its typical firing on Golden Gate texts is far above its firing on *every* control group
(other bridges, other SF things, 'golden'/'gate' words, neutral text).

Usage: python src/gg_search.py Qwen/Qwen3-0.6B
"""
import json
import sys
from pathlib import Path

import torch

from neurons import capture, load

ROOT = Path(__file__).resolve().parents[1]
name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B"
groups = {k: v for k, v in json.loads((ROOT / "prompts/golden_gate.json").read_text(encoding="utf-8")).items()
          if not k.startswith("_")}

model, tok = load(name)
n_layers, width = model.config.num_hidden_layers, model.config.intermediate_size

# summary[group] -> tensors [n_texts, n_layers, width] of per-text max and min
summary = {}
for g, texts in groups.items():
    mx, mn = [], []
    for t in texts:
        ids = tok(t, return_tensors="pt").to("cuda")
        with torch.no_grad(), capture(model) as s:
            model(**ids)
        acts = torch.stack([s["act"][l][0, 1:] for l in range(n_layers)], 1)  # [seq-1, L, W]; skip token 0 (attention sink)
        mx.append(acts.max(0).values.cpu())
        mn.append(acts.min(0).values.cpu())
    summary[g] = (torch.stack(mx), torch.stack(mn))
    print(f"{g}: {len(texts)} texts done")

controls = [g for g in groups if g not in ("gg", "gg_unnamed")]
stats_path = ROOT / "data" / f"stats_{name.split('/')[-1]}.pt"
stats = torch.load(stats_path) if stats_path.exists() else None
MIN_FRAC = 0.2  # a candidate must reach at least this share of its own corpus peak on GG texts


def rank(pole, per_band=8):
    """score = (GG mean - best control group's mean) / spread of the control texts.
    'unnamed' = firing on texts that describe the bridge without naming it, as a share of firing on
    named texts: ~1 suggests a concept detector, ~0 a detector of the words 'Golden Gate'."""
    sign = 1 if pole == "pos" else -1
    k = 0 if pole == "pos" else 1
    gg = sign * summary["gg"][k]                                    # [n, L, W]
    un = sign * summary["gg_unnamed"][k]
    ctrl = {g: sign * summary[g][k] for g in controls}
    allc = torch.cat(list(ctrl.values()))
    gg_mean = gg.mean(0)
    worst_ctrl = torch.stack([c.mean(0) for c in ctrl.values()]).max(0)  # best-firing control group
    score = (gg_mean - worst_ctrl.values) / (allc.std(0) + 1e-3)
    if stats is not None:  # how hard it fires on GG texts vs its own peak on web text
        peak = stats["max"] if pole == "pos" else -stats["min"]
        frac = gg_mean / peak.clamp(min=1e-3)
        score = torch.where(frac >= MIN_FRAC, score, torch.full_like(score, -1e9))
    else:
        frac = torch.full_like(score, float("nan"))
    hit_rate = (gg > allc.max(0).values).float().mean(0)             # share of GG texts beating every control text
    unnamed = (un.mean(0) - allc.mean(0)) / (gg_mean - allc.mean(0) + 1e-6)
    bands = {"early": range(0, n_layers // 3), "middle": range(n_layers // 3, 2 * n_layers // 3),
             "late": range(2 * n_layers // 3, n_layers)}
    out = {}
    for band, ls in bands.items():
        s = score[list(ls)]
        top = s.flatten().topk(per_band)
        rows = []
        for v, i in zip(top.values, top.indices):
            li, n = divmod(i.item(), width)
            l = ls[li]
            rows.append({
                "layer": l, "neuron": n, "score": round(v.item(), 2),
                "gg_mean": round(sign * gg_mean[l, n].item(), 3),
                "closest_control": controls[worst_ctrl.indices[l, n].item()],
                "control_mean": round(sign * worst_ctrl.values[l, n].item(), 3),
                "gg_beats_all_controls": round(hit_rate[l, n].item(), 2),
                "unnamed": round(unnamed[l, n].item(), 2),
                "frac_of_peak": round(frac[l, n].item(), 2),
            })
        out[band] = rows
    return out


def rank_unnamed(pole, top_n=12):
    """EXPLORATORY (not pre-registered): neurons selective for the *unnamed* descriptions."""
    sign = 1 if pole == "pos" else -1
    k = 0 if pole == "pos" else 1
    un = sign * summary["gg_unnamed"][k]
    gg = sign * summary["gg"][k]
    ctrl = {g: sign * summary[g][k] for g in controls}
    allc = torch.cat(list(ctrl.values()))
    worst = torch.stack([c.mean(0) for c in ctrl.values()]).max(0)
    score = (un.mean(0) - worst.values) / (allc.std(0) + 1e-3)
    if stats is not None:
        peak = stats["max"] if pole == "pos" else -stats["min"]
        frac = un.mean(0) / peak.clamp(min=1e-3)
        score = torch.where(frac >= MIN_FRAC, score, torch.full_like(score, -1e9))
    top = score.flatten().topk(top_n)
    rows = []
    for v, i in zip(top.values, top.indices):
        l, n = divmod(i.item(), width)
        rows.append({"layer": l, "neuron": n, "score": round(v.item(), 2),
                     "unnamed_mean": round(sign * un.mean(0)[l, n].item(), 3),
                     "named_mean": round(sign * gg.mean(0)[l, n].item(), 3),
                     "control_mean": round(sign * worst.values[l, n].item(), 3),
                     "closest_control": controls[worst.indices[l, n].item()]})
    return rows


result = {"model": name, "pos": rank("pos"), "neg": rank("neg"),
          "exploratory_unnamed": {"pos": rank_unnamed("pos"), "neg": rank_unnamed("neg")}}
out = ROOT / "results" / f"gg_candidates_{name.split('/')[-1]}.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(result, indent=1), encoding="utf-8")
for pole in ("pos", "neg"):
    for band, rows in result[pole].items():
        print(f"\n== {pole} pole, {band} layers ==")
        print(f"{'L':>3} {'neuron':>6} {'score':>6} {'gg':>7} {'ctrl':>7}  {'beats':>5} {'unnamed':>7} {'frac':>5}  closest control")
        for r in rows:
            print(f"{r['layer']:>3} {r['neuron']:>6} {r['score']:>6} {r['gg_mean']:>7} {r['control_mean']:>7}  "
                  f"{r['gg_beats_all_controls']:>5} {r['unnamed']:>7} {r['frac_of_peak']:>5}  {r['closest_control']}")
for pole in ("pos", "neg"):
    print(f"\n== EXPLORATORY: selective for UNNAMED descriptions, {pole} pole ==")
    print(f"{'L':>3} {'neuron':>6} {'score':>6} {'unnamed':>8} {'named':>7} {'ctrl':>7}  closest control")
    for r in result["exploratory_unnamed"][pole]:
        print(f"{r['layer']:>3} {r['neuron']:>6} {r['score']:>6} {r['unnamed_mean']:>8} {r['named_mean']:>7} "
              f"{r['control_mean']:>7}  {r['closest_control']}")
print("\nsaved", out)
