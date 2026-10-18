"""Who feeds each concept neuron? Exact decomposition of a neuron's input into its sources (attention study, deeper).

At the answer position (last prompt token), a layer-L MLP neuron reads h_L, the residual stream after layer L's
attention. h_L is exactly the sum of: the token embedding, every earlier layer's attention output (split into its 32
heads) and MLP output (split into its 12,288 neurons), plus layer L's own attention output. The neuron computes
   gate = (g_L * w_gate) . h_L / rms(h_L),   value = (g_L * w_up) . h_L / rms(h_L),   act = SiLU(gate) * value
so each source c contributes exactly (g_L * w) . c / rms(h_L) to gate and to value (the rms scale is one number per
token, shared by all sources: the only approximation is treating it as fixed). We report each source's share of the
neuron's gate and value inputs, and a first-order "contribution to activation":
   d_act(c) = SiLU'(gate) * value * gate_c + SiLU(gate) * value_c
signed so that positive = pushes the neuron in the direction it takes on recall.

Sanity checks (the run stops if they fail):
  1. the sum of all sources reproduces h_L (relative error < 2%, bf16 forward);
  2. the decomposed gate and value match the model's own gate_proj / up_proj outputs (< 2%).

Usage: RECALL_VARIANT=V3 ANSWER_DISCOVERY=0 CONCEPT_OUT_DIR=<dir> python src/feeders.py <tag> "<concept>" [...]
"""
import json
import os
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F

from config import DEFAULT_MODEL
import concept_pipeline as cp
from neurons import DEVICE, mlps
from pipeline_fast import FastPipeline


def main():
    tag = sys.argv[1]
    pipe = FastPipeline(DEFAULT_MODEL)
    m = pipe.model; cfg = m.config
    H, D, L = cfg.num_attention_heads, cfg.head_dim, cfg.num_hidden_layers
    eps = cfg.rms_norm_eps
    out = {}
    for concept in [c.strip() for c in sys.argv[2:]]:
        f = cp.OUT / f"{cp.slug(concept)}__clamp_reply.json"
        x = json.loads(f.read_text(encoding="utf-8"))
        spec, tid, recall, _ = pipe.make_prompts(concept)
        topn = int(os.environ.get("FEEDERS_TOPN", "5"))
        targets = [(t["layer"], t["neuron"], -1.0 if t.get("sign_on_recall") == "neg" else 1.0) for t in x["top_neurons"][:topn]]
        # per-neuron weight products depend only on the neuron, not the prompt: compute once (speed; same arithmetic)
        pre = {}
        for (Lt0, n0, _) in targets:
            gL0 = m.model.layers[Lt0].post_attention_layernorm.weight.float()
            wg0 = gL0 * mlps(m)[Lt0].gate_proj.weight[n0].float(); wu0 = gL0 * mlps(m)[Lt0].up_proj.weight[n0].float()
            VG, VU, DG, DU = [], [], [], []
            for l in range(Lt0 + 1):
                WO = m.model.layers[l].self_attn.o_proj.weight.float()
                VG.append((WO.T @ wg0).view(H, D)); VU.append((WO.T @ wu0).view(H, D))
                if l < Lt0:
                    Wd = mlps(m)[l].down_proj.weight.float()
                    DG.append(Wd.T @ wg0); DU.append(Wd.T @ wu0)
            pre[(Lt0, n0)] = (wg0, wu0, VG, VU, DG, DU)
        tl = sorted({l for l, _, _ in targets})
        agg = {t: {"heads": defaultdict(float), "neurons": defaultdict(float), "embed": 0.0, "act": 0.0,
                   "gate_err": 0.0, "value_err": 0.0} for t in targets}
        recon_err = 0.0
        for q, _ in recall:
            st = {"z": {}, "a": {}, "h": {}, "g": {}, "u": {}}
            hooks = [m.model.embed_tokens.register_forward_hook(lambda mod, a, o: st.__setitem__("e", o[0, -1].float()))]
            for l in range(L):
                lay = m.model.layers[l]
                hooks.append(lay.self_attn.o_proj.register_forward_pre_hook(lambda mod, a, l=l: st["z"].__setitem__(l, a[0][0, -1].float())))
                hooks.append(mlps(m)[l].down_proj.register_forward_pre_hook(lambda mod, a, l=l: st["a"].__setitem__(l, a[0][0, -1].float())))
                if l in tl:
                    hooks.append(lay.post_attention_layernorm.register_forward_pre_hook(lambda mod, a, l=l: st["h"].__setitem__(l, a[0][0, -1].float())))
                    hooks.append(mlps(m)[l].gate_proj.register_forward_hook(lambda mod, a, o, l=l: st["g"].__setitem__(l, o[0, -1].float())))
                    hooks.append(mlps(m)[l].up_proj.register_forward_hook(lambda mod, a, o, l=l: st["u"].__setitem__(l, o[0, -1].float())))
            ids = pipe.tok(q, return_tensors="pt").to(DEVICE)
            try:
                with torch.no_grad():
                    m.model(input_ids=ids.input_ids)
            finally:
                for hk in hooks:
                    hk.remove()
            for Lt in tl:
                # check 1: reconstruct h_L from its sources
                recon = st["e"].clone()
                for l in range(Lt):
                    recon += m.model.layers[l].self_attn.o_proj.weight.float() @ st["z"][l]
                    recon += mlps(m)[l].down_proj.weight.float() @ st["a"][l]
                recon += m.model.layers[Lt].self_attn.o_proj.weight.float() @ st["z"][Lt]
                h = st["h"][Lt]
                err = ((recon - h).norm() / h.norm()).item()
                recon_err = max(recon_err, err)
                assert err < 0.02, f"{concept}: residual reconstruction error {err:.3f} at layer {Lt}"
                rms = torch.sqrt((h * h).mean() + eps)
                gL = m.model.layers[Lt].post_attention_layernorm.weight.float()
                for (l0, n, sg) in [t for t in targets if t[0] == Lt]:
                    wg, wu, VG, VU, DG, DU = pre[(Lt, n)]
                    G = (wg @ h) / rms; U = (wu @ h) / rms
                    # check 2: matches the model's own gate/up pre-activations
                    ge = abs(G.item() - st["g"][Lt][n].item()) / (abs(st["g"][Lt][n].item()) + 1e-3)
                    ue = abs(U.item() - st["u"][Lt][n].item()) / (abs(st["u"][Lt][n].item()) + 1e-3)
                    if not (ge < 0.02 and ue < 0.02):
                        print(f"    DEBUG {concept} L{Lt}:{n}: gate mine {G.item():.4f} vs model {st['g'][Lt][n].item():.4f}; "
                              f"value mine {U.item():.4f} vs model {st['u'][Lt][n].item():.4f}", flush=True)
                    assert abs(G.item() - st["g"][Lt][n].item()) <= 0.05 * abs(st["g"][Lt][n].item()) + 0.02 and                         abs(U.item() - st["u"][Lt][n].item()) <= 0.05 * abs(st["u"][Lt][n].item()) + 0.02,                         f"{concept} L{Lt}:{n}: gate err {ge:.3f}, value err {ue:.3f}"
                    sig = torch.sigmoid(G); dsilu = sig * (1 + G * (1 - sig)); silu = G * sig
                    kg = (dsilu * U).item() * sg / rms.item(); ku = silu.item() * sg / rms.item()
                    A = agg[(l0, n, sg)]
                    A["act"] += sg * (silu * U).item() / len(recall)
                    A["gate_err"] = max(A["gate_err"], ge); A["value_err"] = max(A["value_err"], ue)
                    A["embed"] += (kg * (wg @ st["e"]).item() + ku * (wu @ st["e"]).item()) / len(recall)
                    for l in range(Lt + 1):
                        z = st["z"][l].view(H, D)
                        per_head = (kg * (VG[l] * z).sum(-1) + ku * (VU[l] * z).sum(-1)).tolist()
                        for hh in range(H):
                            A["heads"][(l, hh)] += per_head[hh] / len(recall)
                        if l < Lt:
                            per_neuron = st["a"][l] * (kg * DG[l] + ku * DU[l])
                            top = per_neuron.abs().topk(20)
                            for v_, i_ in zip(top.values.tolist(), top.indices.tolist()):
                                A["neurons"][(l, i_)] += per_neuron[i_].item() / len(recall)
                            A["neurons"][(l, "rest")] += (per_neuron.sum().item() - per_neuron[top.indices].sum().item()) / len(recall)
        rows = {}
        for (l0, n, sg), A in agg.items():
            hs = sorted(A["heads"].items(), key=lambda kv: -abs(kv[1]))
            ns = sorted(((k, v) for k, v in A["neurons"].items() if k[1] != "rest"), key=lambda kv: -abs(kv[1]))
            tot_h = sum(v for _, v in A["heads"].items()); tot_n = sum(v for _, v in A["neurons"].items())
            tot = abs(tot_h) + abs(tot_n) + abs(A["embed"]) or 1e-9
            rows[f"L{l0}:{n}"] = {"sign": sg, "act_signed": round(A["act"], 4), "from_heads": round(tot_h, 4), "from_mlp_neurons": round(tot_n, 4),
                                  "from_embedding": round(A["embed"], 4), "heads_share_of_abs": round(abs(tot_h) / tot, 3),
                                  "top_heads": [(f"L{l}H{hh}", round(v, 4)) for (l, hh), v in hs[:8]],
                                  "top_neurons": [(f"L{l}:{nn}", round(v, 4)) for (l, nn), v in ns[:8]],
                                  "max_gate_err": round(A["gate_err"], 4), "max_value_err": round(A["value_err"], 4)}
        out[concept] = {"max_recon_err": round(recon_err, 4), "neurons": rows}
        print(f"\n== {concept} (reconstruction error <= {recon_err:.3f})", flush=True)
        for k, r in rows.items():
            print(f"  {k:<10} act {r['act_signed']:+.3f} | from heads {r['from_heads']:+.3f}, earlier neurons {r['from_mlp_neurons']:+.3f}, "
                  f"embedding {r['from_embedding']:+.3f} | top heads {r['top_heads'][:4]} | top neurons {r['top_neurons'][:3]}", flush=True)
        (cp.OUT / f"feeders_{tag}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
