"""Where do the late writer heads get what they write into the answer? (attention study; no API)

A head's output at the answer position is z_h = sum_j a_j v_j, so its DIRECT contribution to the answer score
(answer logit minus mean logit, as in answer_readout.py) splits exactly by source position j:
    c(j) = a_j * (v_j . W_O,h^T w) / rms_final,   w = (W_U[answer] - mean_v W_U[v]) * g_final
Source classes: sink (position 0), qmark (the question's last '?'), template (other chat-format tokens and whitespace),
suffix ("Answer with just the name." if present, token-overlap match), question (the rest of the user's words),
self (the answer position itself, if not already classed). Reports per head the share of its write from each class,
next to the share of its ATTENTION on each class (attention weight is not information moved, so both are shown).
Sanity check: sum_j a_j v_j rebuilds the head's real o_proj input (< 2%).
Usage: RECALL_VARIANT=V3 ANSWER_DISCOVERY=0 CONCEPT_OUT_DIR=<dir> [WRITER_HEADS=L35H25,...] python src/writer_sources.py <tag> "<concept>" [...]
"""
import json
import os
import sys
from collections import Counter

import torch

from config import DEFAULT_MODEL
import concept_pipeline as cp
from neurons import DEVICE
from pipeline_fast import FastPipeline

FORMAT = {"<|im_start|>", "<|im_end|>", "user", "assistant", "<think>", "</think>"}
SUFFIX = "Answer with just the name."
HEADS = os.environ.get("WRITER_HEADS", "L35H25,L34H28,L35H22,L35H27,L34H29,L35H24")


def classes(toks):
    text = "".join(toks); sfx = text.rfind(SUFFIX); qm = max(i for i, t in enumerate(toks) if "?" in t) if any("?" in t for t in toks) else -1
    out, pos = [], 0
    for j, t in enumerate(toks):
        # 'self' first: the answer position is itself a template token (v1 classed it as template - fixed)
        k = ("sink" if j == 0 else "self" if j == len(toks) - 1 else "qmark" if j == qm else
             "suffix" if sfx >= 0 and pos < sfx + len(SUFFIX) and pos + len(t) > sfx else
             "template" if t.strip() in FORMAT or t.strip() == "" else "question")
        out.append(k); pos += len(t)
    return out


def main():
    tag = sys.argv[1]
    heads = [tuple(int(v) for v in h.strip()[1:].split("H")) for h in HEADS.split(",")]
    pipe = FastPipeline(DEFAULT_MODEL); m = pipe.model; cfg = m.config
    H, D, KV, eps = cfg.num_attention_heads, cfg.head_dim, cfg.num_key_value_heads, cfg.rms_norm_eps
    grp = H // KV
    WU = m.lm_head.weight.float(); WUm = WU.mean(0); gf = m.model.norm.weight.float()
    out_p = cp.OUT / f"writer_sources_{tag}.json"
    out = json.loads(out_p.read_text(encoding="utf-8")) if out_p.exists() else {}
    m.set_attn_implementation("eager")
    try:
        for concept in [c.strip() for c in sys.argv[2:]]:
            if concept in out:
                continue
            spec, tid, recall, _ = pipe.make_prompts(concept); a_tok = tid[0]
            w = (WU[a_tok] - WUm) * gf
            write = {h: Counter() for h in heads}; tmpl = {h: Counter() for h in heads}; look = {h: Counter() for h in heads}; worst = 0.0; qtok = Counter()
            for q, _ in recall:
                ids = pipe.tok(q, return_tensors="pt").to(DEVICE).input_ids
                toks = [pipe.tok.decode([t]) for t in ids[0].tolist()]; cls = classes(toks)
                st = {"v": {}, "z": {}}
                hk = [m.model.norm.register_forward_pre_hook(lambda mod, i: st.__setitem__("hf", i[0][0, -1].float()))]
                for l in {l for l, _ in heads}:
                    hk.append(m.model.layers[l].self_attn.v_proj.register_forward_hook(lambda mod, i, o, l=l: st["v"].__setitem__(l, o[0].float())))
                    hk.append(m.model.layers[l].self_attn.o_proj.register_forward_pre_hook(lambda mod, i, l=l: st["z"].__setitem__(l, i[0][0, -1].float())))
                try:
                    with torch.no_grad():
                        att = m(input_ids=ids, output_attentions=True).attentions
                finally:
                    for h_ in hk:
                        h_.remove()
                rms = torch.sqrt((st["hf"] ** 2).mean() + eps).item()
                for (l, hh) in heads:
                    a = att[l][0, hh, -1].float(); v = st["v"][l].view(-1, KV, D)[:, hh // grp, :]
                    z_true = st["z"][l][hh * D:(hh + 1) * D]
                    err = (((a[:, None] * v).sum(0) - z_true).norm() / (z_true.norm() + 1e-6)).item(); worst = max(worst, err)
                    assert err < 0.02, f"{concept} L{l}H{hh}: value reconstruction error {err:.3f}"
                    WO = m.model.layers[l].self_attn.o_proj.weight[:, hh * D:(hh + 1) * D].float()
                    per = a * (v @ (WO.T @ w)) / rms
                    for j, k in enumerate(cls):
                        write[(l, hh)][k] += per[j].item() / len(recall); look[(l, hh)][k] += a[j].item() / len(recall)
                        if k in ("template", "self"):
                            tk = f"{k}[-{len(toks) - j}]{toks[j]!r}"
                            tmpl[(l, hh)][tk] += per[j].item() / len(recall)
                qtok[toks[[i for i, k in enumerate(cls) if k == "qmark"][0]] if "qmark" in cls else "-"] += 1
            out[concept] = {"max_value_recon_err": round(worst, 4), "qmark_tokens": dict(qtok),
                            **{f"L{l}H{hh}": {"write": {k: round(v, 3) for k, v in write[(l, hh)].items()},
                                              "attention": {k: round(v, 3) for k, v in look[(l, hh)].items()},
                                              "by_template_token": dict(sorted(((k, round(v, 3)) for k, v in tmpl[(l, hh)].items()), key=lambda kv: -abs(kv[1]))[:6])} for (l, hh) in heads}}
            print(f"{concept:<28} " + " | ".join(
                f"L{l}H{hh} total {sum(write[(l, hh)].values()):+.2f}, from '?' {write[(l, hh)]['qmark']:+.2f} (attn {look[(l, hh)]['qmark']:.2f}), "
                f"self {write[(l, hh)]['self']:+.2f}, template {write[(l, hh)]['template']:+.2f}, question {write[(l, hh)]['question']:+.2f}" for (l, hh) in heads[:3]), flush=True)
            out_p.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    finally:
        m.set_attn_implementation("sdpa")


if __name__ == "__main__":
    main()
