"""Causal test of concept-neuron CHAINS: cut a single neuron -> neuron route (attention study, deeper; no API).

For a link S (layer Ls) -> T (layer Lt > Ls) at the answer position (last prompt token), on every recall prompt:
  base       nothing changed
  chain_cut  remove ONLY S's contribution to T's gate and value inputs (S still writes everywhere else)
  matched    the same for a "matched" feeder M of T: the largest-|contribution| feeder of T that is NOT among the
             concept's top-10 neurons and lies below Lt (from feeders json). Tests concept link vs a generic input
  s_off      S's activation set to 0 at that position (all of S's routes)
Measured: T's signed activation and the answer's first-token log-odds. The normalisation scale of T's layer is held at
its unmodified value for the route cuts (as in feeders.py / route_cut.py).

Usage: RECALL_VARIANT=V3 ANSWER_DISCOVERY=0 CONCEPT_OUT_DIR=<dir> python src/chain_cut.py <feeders json> <tag> "<concept>=L<s>:<n>>L<t>:<n>" [...]
"""
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from config import DEFAULT_MODEL
import concept_pipeline as cp
from neurons import DEVICE, mlps
from pipeline_fast import FastPipeline


def lo(logits, tok):
    p = min(max(F.softmax(logits.float(), -1)[tok].item(), 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


def parse(u):
    l, n = u.strip().lstrip("L").split(":"); return int(l), int(n)


def main():
    fd = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")); tag = sys.argv[2]
    pipe = FastPipeline(DEFAULT_MODEL); m = pipe.model; eps = m.config.rms_norm_eps
    out_p = cp.OUT / f"chain_cut_{tag}.json"
    out = json.loads(out_p.read_text(encoding="utf-8")) if out_p.exists() else {}
    for arg in sys.argv[3:]:
        concept, link = arg.rsplit("=", 1); s_u, t_u = (parse(x) for x in link.split(">"))
        (Ls, s), (Lt, t) = s_u, t_u
        if f"{concept}: L{Ls}:{s} > L{Lt}:{t}" in out:      # resumable
            continue
        assert Ls < Lt, "source must be below target"
        x = json.loads((cp.OUT / f"{cp.slug(concept)}__clamp_reply.json").read_text(encoding="utf-8"))
        own = {(q["layer"], q["neuron"]) for q in x["top_neurons"][:10]}
        if (Ls, s) not in own:      # a chain link must start at one of the concept's own neurons (Mozart row error)
            print(f"{concept}: source L{Ls}:{s} is not among the concept's top-10 neurons - not a chain link, skipped", flush=True)
            continue
        r = fd[concept]["neurons"][f"L{Lt}:{t}"]; sg = r["sign"]
        cands = [(parse(k), v) for k, v in r["top_neurons"] if parse(k) not in own and parse(k)[0] < Lt]
        assert cands, f"no matched control feeder for L{Lt}:{t}"
        (Lm, mm), mval = max(cands, key=lambda kv: abs(kv[1]))
        spec, tid, recall, _ = pipe.make_prompts(concept); first = tid[0]
        gL = m.model.layers[Lt].post_attention_layernorm.weight.float()
        wg = gL * mlps(m)[Lt].gate_proj.weight[t].float(); wu = gL * mlps(m)[Lt].up_proj.weight[t].float()
        ds = mlps(m)[Ls].down_proj.weight[:, s].float(); dm = mlps(m)[Lm].down_proj.weight[:, mm].float()
        acc = {k: {"act": 0.0, "lo": 0.0} for k in ("base", "chain_cut", "matched", "s_off")}
        for q, _ in recall:
            ids = pipe.tok(q, return_tensors="pt").to(DEVICE).input_ids
            st = {}
            hk = [m.model.layers[Lt].post_attention_layernorm.register_forward_pre_hook(lambda mod, a: st.__setitem__("h", a[0][0, -1].float())),
                  mlps(m)[Ls].down_proj.register_forward_pre_hook(lambda mod, a: st.__setitem__("as", a[0][0, -1, s].float().item())),
                  mlps(m)[Lm].down_proj.register_forward_pre_hook(lambda mod, a: st.__setitem__("am", a[0][0, -1, mm].float().item())),
                  mlps(m)[Lt].down_proj.register_forward_pre_hook(lambda mod, a: st.__setitem__("at", a[0][0, -1, t].float().item()))]
            try:
                with torch.no_grad():
                    base = m(input_ids=ids).logits[0, -1]
            finally:
                for h_ in hk:
                    h_.remove()
            rms = torch.sqrt((st["h"] ** 2).mean() + eps)
            G = (wg @ st["h"] / rms).item(); U = (wu @ st["h"] / rms).item()
            acc["base"]["act"] += sg * st["at"] / len(recall); acc["base"]["lo"] += lo(base, first) / len(recall)

            def set_t(val):
                def h(mod, a):
                    xx = a[0].clone(); xx[0, -1, t] = val; return (xx,)
                hh = mlps(m)[Lt].down_proj.register_forward_pre_hook(h)
                try:
                    with torch.no_grad():
                        return m(input_ids=ids).logits[0, -1]
                finally:
                    hh.remove()
            for key, vec, a_src in (("chain_cut", ds, st["as"]), ("matched", dm, st["am"])):
                c = vec * a_src
                newt = F.silu(torch.tensor(G - (wg @ c / rms).item())).item() * (U - (wu @ c / rms).item())
                lg = set_t(newt)
                acc[key]["act"] += sg * newt / len(recall); acc[key]["lo"] += lo(lg, first) / len(recall)
            stt = {}

            def zero_s(mod, a):
                xx = a[0].clone(); xx[0, -1, s] = 0.0; return (xx,)
            h1 = mlps(m)[Ls].down_proj.register_forward_pre_hook(zero_s)
            h2 = mlps(m)[Lt].down_proj.register_forward_pre_hook(lambda mod, a: stt.__setitem__("at", a[0][0, -1, t].float().item()))
            try:
                with torch.no_grad():
                    lg = m(input_ids=ids).logits[0, -1]
            finally:
                h1.remove(); h2.remove()
            acc["s_off"]["act"] += sg * stt["at"] / len(recall); acc["s_off"]["lo"] += lo(lg, first) / len(recall)
        b = acc["base"]
        key = f"{concept}: L{Ls}:{s} > L{Lt}:{t}"
        out[key] = {"matched_control": f"L{Lm}:{mm}", "matched_feeder_contrib": round(mval, 3),
                    **{k: {"t_act": round(v["act"], 2), "recall_drop": round(b["lo"] - v["lo"], 3)} for k, v in acc.items()}}
        o = out[key]
        print(f"{key:<44} T act {b['act']:.1f} | chain cut -> {o['chain_cut']['t_act']:.1f}, recall drop {o['chain_cut']['recall_drop']:+.2f} | "
              f"matched L{Lm}:{mm} -> {o['matched']['t_act']:.1f}, drop {o['matched']['recall_drop']:+.2f} | S off -> T {o['s_off']['t_act']:.1f}, "
              f"drop {o['s_off']['recall_drop']:+.2f}", flush=True)
        out_p.write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
