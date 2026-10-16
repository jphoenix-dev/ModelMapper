"""Causal test of head -> neuron routes (attention study, deeper; no API calls).

For each concept's entry neuron (L, n) and every recall prompt, at the answer position (last prompt token) only:
  base       nothing changed
  route_cut  remove ONLY the contribution of the neuron's top-5 positive feeding heads (from feeders_*.json) to this
             neuron's gate and value inputs (those heads still write everywhere else, including to every other neuron)
  route_rand the same for 5 random heads from the same layers (control)
  neuron_off the neuron's activation set to 0 (the most any route cut through it could do)
  heads_off  the same 5 feeding heads switched off entirely (all routes)
Measured: the neuron's (signed) activation, and the log-odds of the answer's FIRST token at that position.
If route_cut ~ heads_off on recall, those heads matter THROUGH this neuron; if much smaller, they act mostly elsewhere.
Route removal uses the per-token normalisation scale from the unmodified pass (held fixed, as in feeders.py).

Usage: RECALL_VARIANT=V3 ANSWER_DISCOVERY=0 CONCEPT_OUT_DIR=<dir> python src/route_cut.py <feeders json> <tag> "<concept>=L:n" [...]
"""
import json
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from config import DEFAULT_MODEL
import concept_pipeline as cp
from neurons import DEVICE, mlps
from pipeline_fast import FastPipeline


def logodds(logits, tok):
    p = F.softmax(logits.float(), -1)[tok].item()
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


def main():
    fd = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")); tag = sys.argv[2]
    pipe = FastPipeline(DEFAULT_MODEL); m = pipe.model; cfg = m.config
    H, D, eps = cfg.num_attention_heads, cfg.head_dim, cfg.rms_norm_eps
    out = {}
    for arg in sys.argv[3:]:
        concept, ln = arg.rsplit("=", 1); L, n = (int(v) for v in ln.strip("L").split(":"))
        r = fd[concept]["neurons"][f"L{L}:{n}"]; sg = r["sign"]
        feeders = [tuple(int(v) for v in h[1:].split("H")) for h, val in r["top_heads"] if val > 0][:5]
        rng = random.Random(6); rand = [(l, rng.randrange(H)) for l, _ in feeders]
        spec, tid, recall, _ = pipe.make_prompts(concept); first = tid[0]
        gL = m.model.layers[L].post_attention_layernorm.weight.float()
        wg = gL * mlps(m)[L].gate_proj.weight[n].float(); wu = gL * mlps(m)[L].up_proj.weight[n].float()
        res = {k: {"act": 0.0, "lo": 0.0} for k in ("base", "route_cut", "route_rand", "neuron_off", "heads_off")}
        for q, _ in recall:
            ids = pipe.tok(q, return_tensors="pt").to(DEVICE).input_ids
            # pass 1: capture head inputs (o_proj) at the last position and h_L, for computing route contributions
            st = {"z": {}}
            hooks = [m.model.layers[l].self_attn.o_proj.register_forward_pre_hook(
                lambda mod, a, l=l: st["z"].__setitem__(l, a[0][0, -1].float())) for l in range(L + 1)]
            hooks.append(m.model.layers[L].post_attention_layernorm.register_forward_pre_hook(
                lambda mod, a: st.__setitem__("h", a[0][0, -1].float())))
            hooks.append(mlps(m)[L].down_proj.register_forward_pre_hook(lambda mod, a: st.__setitem__("act", a[0][0, -1, n].float().item())))
            try:
                with torch.no_grad():
                    base_logits = m(input_ids=ids).logits[0, -1]
            finally:
                for hk in hooks:
                    hk.remove()
            rms = torch.sqrt((st["h"] ** 2).mean() + eps)
            G = (wg @ st["h"] / rms).item(); U = (wu @ st["h"] / rms).item()

            def contrib(heads):
                dg = du = 0.0
                for l, hh in heads:
                    WO = m.model.layers[l].self_attn.o_proj.weight[:, hh * D:(hh + 1) * D].float()
                    c = WO @ st["z"][l][hh * D:(hh + 1) * D]
                    dg += (wg @ c / rms).item(); du += (wu @ c / rms).item()
                return dg, du
            res["base"]["act"] += sg * st["act"] / len(recall); res["base"]["lo"] += logodds(base_logits, first) / len(recall)

            def run_with_neuron_value(newval):
                def hk(mod, a):
                    x = a[0].clone(); x[0, -1, n] = newval; return (x,)
                h_ = mlps(m)[L].down_proj.register_forward_pre_hook(hk)
                try:
                    with torch.no_grad():
                        return m(input_ids=ids).logits[0, -1]
                finally:
                    h_.remove()
            for key, heads in (("route_cut", feeders), ("route_rand", rand)):
                dg, du = contrib(heads)
                newact = F.silu(torch.tensor(G - dg)).item() * (U - du)
                lg = run_with_neuron_value(newact)
                res[key]["act"] += sg * newact / len(recall); res[key]["lo"] += logodds(lg, first) / len(recall)
            lg = run_with_neuron_value(0.0)
            res["neuron_off"]["lo"] += logodds(lg, first) / len(recall)
            with torch.no_grad(), pipe.ablate_heads(feeders), \
                    torch.autograd.graph.save_on_cpu() if False else _null():
                st2 = {}
                h2 = mlps(m)[L].down_proj.register_forward_pre_hook(lambda mod, a: st2.__setitem__("act", a[0][0, -1, n].float().item()))
                try:
                    lg = m(input_ids=ids).logits[0, -1]
                finally:
                    h2.remove()
            res["heads_off"]["act"] += sg * st2["act"] / len(recall); res["heads_off"]["lo"] += logodds(lg, first) / len(recall)
        b = res["base"]
        out[concept] = {"neuron": f"L{L}:{n}", "feeders": [f"L{l}H{h}" for l, h in feeders], "random": [f"L{l}H{h}" for l, h in rand],
                        **{k: {"act": round(v["act"], 3), "logodds_drop": round(b["lo"] - v["lo"], 3)} for k, v in res.items()}}
        o = out[concept]
        print(f"{concept:<24} L{L}:{n} act {b['act']:.1f} | route cut: act {o['route_cut']['act']:.1f}, recall drop {o['route_cut']['logodds_drop']:+.2f} | "
              f"random route: act {o['route_rand']['act']:.1f}, drop {o['route_rand']['logodds_drop']:+.2f} | neuron off: drop "
              f"{o['neuron_off']['logodds_drop']:+.2f} | heads fully off: act {o['heads_off']['act']:.1f}, drop {o['heads_off']['logodds_drop']:+.2f}", flush=True)
        (cp.OUT / f"route_cut_{tag}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    main()
