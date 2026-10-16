"""Bow-tie test: do a concept's few bottleneck neurons write the answer themselves, or mostly THROUGH other parts?
(attention study; no API)

Readout showed: recall flips with 1-5 concept neurons off, yet the answer word's score is written by ~14 neurons for
half of it, and the concept's own top-10 write only ~17% directly. If that is a bow-tie (few bottleneck neurons ->
many writers), then switching off the bottleneck neurons should remove far more score than they write directly.

Per concept, every recall prompt, at the answer position (last prompt token) only:
  base     exact decomposition of the answer score (answer logit minus mean logit) into heads / own neurons / other
           neurons / embedding, as in answer_readout.py (each pass uses its own final rms, so each sum is exact)
  ablated  the concept's top-K neurons (K = BOWTIE_K, default 5) set to 0 at that position; same decomposition
Reports: total drop; direct part (what the K neurons wrote in base); indirect = total - direct, split into the drop
in what heads wrote and what other neurons wrote. Control: K random neurons from the same layers (the same split).
Usage: RECALL_VARIANT=V3 ANSWER_DISCOVERY=0 CONCEPT_OUT_DIR=<dir> python src/bowtie.py <tag> "<concept>" [...]
"""
import json
import math
import os
import random
import sys

import torch

from config import DEFAULT_MODEL
import concept_pipeline as cp
from neurons import DEVICE, mlps
from pipeline_fast import FastPipeline

K = int(os.environ.get("BOWTIE_K", "5"))


def main():
    tag = sys.argv[1]
    pipe = FastPipeline(DEFAULT_MODEL); m = pipe.model; cfg = m.config
    L, eps, W = cfg.num_hidden_layers, cfg.rms_norm_eps, cfg.intermediate_size
    WU = m.lm_head.weight.float(); WUm = WU.mean(0); gf = m.model.norm.weight.float()
    out_p = cp.OUT / f"bowtie_{tag}.json"
    out = json.loads(out_p.read_text(encoding="utf-8")) if out_p.exists() else {}
    for concept in [c.strip() for c in sys.argv[2:]]:
        if concept in out:
            continue
        x = json.loads((cp.OUT / f"{cp.slug(concept)}__clamp_reply.json").read_text(encoding="utf-8"))
        own = [(q["layer"], q["neuron"]) for q in x["top_neurons"][:K]]
        rng = random.Random(len(concept)); ownset = set(own)
        rand = [(l, rng.choice([n for n in range(W) if (l, n) not in ownset])) for l, _ in own]
        spec, tid, recall, _ = pipe.make_prompts(concept); a = tid[0]
        w = (WU[a] - WUm) * gf
        WO = [m.model.layers[l].self_attn.o_proj.weight.float().T @ w for l in range(L)]
        VN = [mlps(m)[l].down_proj.weight.float().T @ w for l in range(L)]

        def run(ids, kill, which, allpos=False):
            st = {"z": {}, "a": {}}
            hk = [m.model.embed_tokens.register_forward_hook(lambda mod, i, o: st.__setitem__("e", o[0, -1].float())),
                  m.model.norm.register_forward_pre_hook(lambda mod, i: st.__setitem__("hf", i[0][0, -1].float()))]
            for l in range(L):
                hk.append(m.model.layers[l].self_attn.o_proj.register_forward_pre_hook(lambda mod, i, l=l: st["z"].__setitem__(l, i[0][0, -1].float())))
                ns = [n for ll, n in kill if ll == l]

                def h(mod, i, l=l, ns=ns):
                    if ns:
                        xx = i[0].clone()
                        if allpos:
                            xx[0, :, ns] = 0.0
                        else:
                            xx[0, -1, ns] = 0.0
                        st["a"][l] = xx[0, -1].float(); return (xx,)
                    st["a"][l] = i[0][0, -1].float()
                hk.append(mlps(m)[l].down_proj.register_forward_pre_hook(h))
            try:
                with torch.no_grad():
                    lg = m(input_ids=ids).logits[0, -1].float()
            finally:
                for h_ in hk:
                    h_.remove()
            rms = torch.sqrt((st["hf"] ** 2).mean() + eps)
            heads = sum((WO[l] @ st["z"][l]).item() for l in range(L)) / rms.item()
            pn = [st["a"][l] * VN[l] / rms for l in range(L)]
            mine = sum(pn[l][n].item() for l, n in which)
            other = sum(p.sum().item() for p in pn) - mine
            emb = (w @ st["e"] / rms).item()
            score = (lg[a] - lg.mean()).item()
            p_ = min(max(torch.softmax(lg, -1)[a].item(), 1e-9), 1 - 1e-9); lo = math.log(p_ / (1 - p_))
            assert abs(heads + mine + other + emb - score) <= 0.05 * abs(score) + 0.3, f"{concept}: decomposition {heads + mine + other + emb:.2f} vs {score:.2f}"
            return {"score": score, "heads": heads, "set": mine, "other_neurons": other, "rms": rms.item(), "logodds": lo}

        acc = {k: {} for k in ("base_own", "own_off", "own_off_allpos", "base_rand", "rand_off", "rand_off_allpos")}
        for q, _ in recall:
            ids = pipe.tok(q, return_tensors="pt").to(DEVICE).input_ids
            for key, kill, which, ap in (("base_own", [], own, False), ("own_off", own, own, False), ("own_off_allpos", own, own, True),
                                         ("base_rand", [], rand, False), ("rand_off", rand, rand, False), ("rand_off_allpos", rand, rand, True)):
                r = run(ids, kill, which, ap)
                for k, v in r.items():
                    acc[key][k] = acc[key].get(k, 0.0) + v / len(recall)
        res = {}
        for arm, b, o, oa in (("own", acc["base_own"], acc["own_off"], acc["own_off_allpos"]),
                              ("random", acc["base_rand"], acc["rand_off"], acc["rand_off_allpos"])):
            total = b["score"] - o["score"]; direct = b["set"]
            res[arm] = {"total_drop": round(total, 2), "direct": round(direct, 2), "indirect": round(total - direct, 2),
                        "via_heads": round(b["heads"] - o["heads"], 2), "via_other_neurons": round(b["other_neurons"] - o["other_neurons"], 2),
                        "rms_ratio": round(o["rms"] / b["rms"], 3), "logodds_drop": round(b["logodds"] - o["logodds"], 3),
                        "allpos_total_drop": round(b["score"] - oa["score"], 2), "allpos_logodds_drop": round(b["logodds"] - oa["logodds"], 3)}
        out[concept] = {"k": K, "neurons": [f"L{l}:{n}" for l, n in own], "random": [f"L{l}:{n}" for l, n in rand],
                        "base_score": round(acc["base_own"]["score"], 2), **res}
        o, r_ = res["own"], res["random"]
        print(f"{concept:<28} base {acc['base_own']['score']:+.1f} | own top-{K} off: drop {o['total_drop']:+.1f} = direct {o['direct']:+.1f} + "
              f"indirect {o['indirect']:+.1f} (heads {o['via_heads']:+.1f}, other neurons {o['via_other_neurons']:+.1f}) | random: drop "
              f"{r_['total_drop']:+.1f} (direct {r_['direct']:+.1f}) | log-odds drop: answer pos {o['logodds_drop']:+.2f}, all positions "
              f"{o['allpos_logodds_drop']:+.2f} (score {o['allpos_total_drop']:+.1f}); random all pos {r_['allpos_logodds_drop']:+.2f}", flush=True)
        out_p.write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
