"""Do the late 'anti-writer' heads regulate confidence through the final normalisation? (attention study; no API)

Hypothesis (after head_arms.py showed that switching L35H27 / L34H29 / L35H24 off LOWERS recall although they push the
answer down directly): these heads mostly read their own position and write roughly AGAINST the residual stream there,
shrinking its size. The final RMS norm divides by that size, so a smaller stream scales every logit UP: a sharper,
more confident next-word distribution. Removing them would grow the stream, flatten the distribution, lower recall.

Per concept, every recall prompt, at the answer position, for each head h (named heads + 3 random heads in layers
34-35): with c_h its output (W_O slice @ z) and r the residual stream entering h's layer,
  cos      cosine(c_h, r)  (negative = writes against the stream)
  rel      |c_h| / |r|
  rms_up   final-stream rms if c_h alone is removed / actual final rms (direct effect only; > 1 = head shrinks stream)
  lo_scale change in answer log-odds from ONLY that rms change (logits rescaled; no direction change)
  lo_dir   change in answer log-odds from ONLY removing c_h's direction (rms held at the actual value)
Usage: RECALL_VARIANT=V3 ANSWER_DISCOVERY=0 CONCEPT_OUT_DIR=<dir> python src/norm_heads.py <tag> "<concept>" [...]
"""
import json
import math
import random
import sys

import torch

from config import DEFAULT_MODEL
import concept_pipeline as cp
from neurons import DEVICE
from pipeline_fast import FastPipeline

NAMED = [(35, 25), (34, 28), (35, 22), (35, 27), (34, 29), (35, 24)]


def lo_from(logits, a):
    # exact log-odds, no clamp: logit_a - logsumexp(other logits) in float64 (the float32 p/(1-p) form caps at ~20.7;
    # see logodds_cap_audit.py)
    x = logits.double(); return (x[a] - torch.logsumexp(torch.cat([x[:a], x[a + 1:]]), 0)).item()


def main():
    tag = sys.argv[1]
    pipe = FastPipeline(DEFAULT_MODEL); m = pipe.model; cfg = m.config
    H, D, eps = cfg.num_attention_heads, cfg.head_dim, cfg.rms_norm_eps
    WU = m.lm_head.weight.float(); gf = m.model.norm.weight.float()
    out_p = cp.OUT / f"norm_heads_{tag}.json"
    out = json.loads(out_p.read_text(encoding="utf-8")) if out_p.exists() else {}
    for concept in [c.strip() for c in sys.argv[2:]]:
        if concept in out:
            continue
        rng = random.Random(sum(map(ord, concept)) + 7)
        rand = []
        while len(rand) < 3:
            h = (rng.choice((34, 35)), rng.randrange(H))
            if h not in NAMED and h not in rand:
                rand.append(h)
        heads = NAMED + rand
        spec, tid, recall, _ = pipe.make_prompts(concept); a = tid[0]
        acc = {h: {"cos": 0.0, "rel": 0.0, "rms_up": 0.0, "lo_scale": 0.0, "lo_dir": 0.0} for h in heads}
        for q, _ in recall:
            ids = pipe.tok(q, return_tensors="pt").to(DEVICE).input_ids
            st = {"z": {}, "r": {}}
            hk = [m.model.norm.register_forward_pre_hook(lambda mod, i: st.__setitem__("hf", i[0][0, -1].float()))]
            for l in {l for l, _ in heads}:
                hk.append(m.model.layers[l].self_attn.o_proj.register_forward_pre_hook(lambda mod, i, l=l: st["z"].__setitem__(l, i[0][0, -1].float())))
                hk.append(m.model.layers[l].register_forward_pre_hook(lambda mod, i, l=l: st["r"].__setitem__(l, i[0][0, -1].float())))
            try:
                with torch.no_grad():
                    logits = m(input_ids=ids).logits[0, -1].float()
            finally:
                for h_ in hk:
                    h_.remove()
            hf = st["hf"]; rms = torch.sqrt((hf ** 2).mean() + eps)
            mine = WU @ (gf * hf / rms)
            assert (mine - logits).abs().max().item() < 0.05 * logits.abs().max().item() + 0.5, "final logits not rebuilt"
            base_lo = lo_from(mine, a)
            for (l, hh) in heads:
                c = m.model.layers[l].self_attn.o_proj.weight[:, hh * D:(hh + 1) * D].float() @ st["z"][l][hh * D:(hh + 1) * D]
                r = st["r"][l]
                rms2 = torch.sqrt(((hf - c) ** 2).mean() + eps)
                n_ = len(recall)
                acc[(l, hh)]["cos"] += torch.nn.functional.cosine_similarity(c, r, 0).item() / n_
                acc[(l, hh)]["rel"] += (c.norm() / r.norm()).item() / n_
                acc[(l, hh)]["rms_up"] += (rms2 / rms).item() / n_
                acc[(l, hh)]["lo_scale"] += (lo_from(mine * (rms / rms2), a) - base_lo) / n_
                acc[(l, hh)]["lo_dir"] += (lo_from(WU @ (gf * (hf - c) / rms), a) - base_lo) / n_
        out[concept] = {f"L{l}H{hh}" + ("" if (l, hh) in NAMED else "*"): {k: round(v, 4) for k, v in acc[(l, hh)].items()} for (l, hh) in heads}
        print(f"{concept:<28} " + " | ".join(f"L{l}H{hh}{'' if (l, hh) in NAMED else '*'} cos {acc[(l, hh)]['cos']:+.2f} rms x{acc[(l, hh)]['rms_up']:.3f} "
                                             f"lo scale {acc[(l, hh)]['lo_scale']:+.2f} dir {acc[(l, hh)]['lo_dir']:+.2f}" for (l, hh) in heads[3:7]), flush=True)
        out_p.write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
