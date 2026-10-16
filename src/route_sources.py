"""What do the feeding heads deliver to the entry neuron, and from which words? (attention study, deeper; no API)

A head's output at the answer position is a weighted sum over earlier positions j: z_h = sum_j a_j * v_j (a = its
attention pattern, v = its value vectors; Qwen3 shares each value head across 4 query heads). Its contribution to the
entry neuron's gate/value inputs therefore splits EXACTLY by source position:
   gate_c(j) = a_j * (w_gate . W_O,h v_j) / rms,  value_c(j) likewise,
and the first-order contribution to the neuron (as in feeders.py) splits the same way. We report, per concept and
feeding head, the source words that deliver most of the push into the neuron, summed over all recall prompts.
Sanity check: sum_j a_j v_j must reproduce the head's actual input to o_proj (relative error < 2%).

Usage: RECALL_VARIANT=V3 ANSWER_DISCOVERY=0 CONCEPT_OUT_DIR=<dir> python src/route_sources.py <feeders json> <tag> "<concept>=L:n" [...]
"""
import json
import sys
from collections import Counter
from pathlib import Path

import torch

from config import DEFAULT_MODEL
import concept_pipeline as cp
from neurons import DEVICE, mlps
from pipeline_fast import FastPipeline


def main():
    fd = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")); tag = sys.argv[2]
    pipe = FastPipeline(DEFAULT_MODEL); m = pipe.model; cfg = m.config
    H, D, KV, eps = cfg.num_attention_heads, cfg.head_dim, cfg.num_key_value_heads, cfg.rms_norm_eps
    grp = H // KV
    m.set_attn_implementation("eager")
    out = {}
    try:
        for arg in sys.argv[3:]:
            concept, ln = arg.rsplit("=", 1); L, n = (int(v) for v in ln.strip("L").split(":"))
            r = fd[concept]["neurons"][f"L{L}:{n}"]; sg = r["sign"]
            feeders = [tuple(int(v) for v in h[1:].split("H")) for h, val in r["top_heads"] if val > 0][:5]
            spec, tid, recall, _ = pipe.make_prompts(concept)
            gL = m.model.layers[L].post_attention_layernorm.weight.float()
            wg = gL * mlps(m)[L].gate_proj.weight[n].float(); wu = gL * mlps(m)[L].up_proj.weight[n].float()
            words = {h: Counter() for h in feeders}; worst = 0.0
            for q, _ in recall:
                ids = pipe.tok(q, return_tensors="pt").to(DEVICE).input_ids
                toks = [pipe.tok.decode([t]) for t in ids[0].tolist()]
                st = {"v": {}, "z": {}}
                hooks = []
                for l in {l for l, _ in feeders}:
                    hooks.append(m.model.layers[l].self_attn.v_proj.register_forward_hook(
                        lambda mod, a, o, l=l: st["v"].__setitem__(l, o[0].float())))
                    hooks.append(m.model.layers[l].self_attn.o_proj.register_forward_pre_hook(
                        lambda mod, a, l=l: st["z"].__setitem__(l, a[0][0, -1].float())))
                hooks.append(m.model.layers[L].post_attention_layernorm.register_forward_pre_hook(
                    lambda mod, a: st.__setitem__("h", a[0][0, -1].float())))
                try:
                    with torch.no_grad():
                        att = m(input_ids=ids, output_attentions=True).attentions
                finally:
                    for hk in hooks:
                        hk.remove()
                h = st["h"]; rms = torch.sqrt((h * h).mean() + eps)
                G = (wg @ h / rms); U = (wu @ h / rms)
                sig = torch.sigmoid(G); kg = (sig * (1 + G * (1 - sig)) * U).item() * sg / rms.item(); ku = (G * sig).item() * sg / rms.item()
                for (l, hh) in feeders:
                    a = att[l][0, hh, -1].float()                                   # [T]
                    v = st["v"][l].view(-1, KV, D)[:, hh // grp, :]               # [T, D]
                    z_rebuilt = (a[:, None] * v).sum(0)
                    z_true = st["z"][l][hh * D:(hh + 1) * D]
                    err = ((z_rebuilt - z_true).norm() / (z_true.norm() + 1e-6)).item(); worst = max(worst, err)
                    assert err < 0.02, f"{concept} L{l}H{hh}: value reconstruction error {err:.3f}"
                    WO = m.model.layers[l].self_attn.o_proj.weight[:, hh * D:(hh + 1) * D].float()
                    per_pos = a * (kg * (v @ (WO.T @ wg)) + ku * (v @ (WO.T @ wu)))     # [T] contribution by source
                    for j, t in enumerate(toks):
                        words[(l, hh)][t.strip() or repr(t)] += per_pos[j].item() / len(recall)
            out[concept] = {"neuron": f"L{L}:{n}", "max_value_recon_err": round(worst, 4),
                            "heads": {f"L{l}H{hh}": [(w, round(v, 3)) for w, v in sorted(c.items(), key=lambda kv: -abs(kv[1]))[:8]]
                                      for (l, hh), c in words.items()}}
            print(f"\n== {concept} -> L{L}:{n} (value reconstruction error <= {worst:.3f})", flush=True)
            for k, v in out[concept]["heads"].items():
                print(f"  {k:<7} delivers from: {v[:6]}", flush=True)
            (cp.OUT / f"route_sources_{tag}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    finally:
        m.set_attn_implementation("sdpa")


if __name__ == "__main__":
    main()
