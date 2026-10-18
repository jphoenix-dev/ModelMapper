"""Attention study, step 3: where do the important heads actually look? (no API calls)

For each concept's top-3 measured heads (head_logodds results), on EVERY recall prompt (no prompt cap: the older
head_looks() read only 8), read the attention row from the answer position (last prompt token). Tokens are split into:
  sink      the first token and pure-whitespace tokens (heads often park attention there; known "attention sinks")
  template  chat scaffolding: role tags, <think> tags, and the fixed suffix "Answer with just the name."
  content   the question's own words
Reported per head: share of attention on content, and (with sinks and template removed) the top content tokens.
Caveat: attention weight is not the whole story (the value vectors matter too). It's a first look, not proof.

Usage: RECALL_VARIANT=V3 ANSWER_DISCOVERY=0 CONCEPT_OUT_DIR=<dir> python src/head_looks2.py <head_logodds json> [concept ...]
"""
import json
import sys
from collections import Counter, defaultdict

import torch

from config import DEFAULT_MODEL
from neurons import DEVICE
from pipeline_fast import FastPipeline

SUFFIX_WORDS = {"answer", "with", "just", "the", "name", "."}
TEMPLATE = {"<|im_start|>", "<|im_end|>", "user", "assistant", "<think>", "</think>"}


def main():
    res = json.loads(open(sys.argv[1], encoding="utf-8").read())
    only = [c.strip() for c in sys.argv[2:]] or sorted(res)
    pipe = FastPipeline(DEFAULT_MODEL)
    pipe.model.set_attn_implementation("eager")
    out = {}
    try:
        for concept in only:
            if concept not in res:
                continue
            spec, tid, recall, _ = pipe.make_prompts(concept)
            heads = [tuple(h) for h in res[concept]["top10_heads"][:3]]
            acc = {h: {"content": 0.0, "template": 0.0, "sink": 0.0, "tokens": Counter()} for h in heads}
            for q, _ in recall:
                ids = pipe.tok(q, return_tensors="pt").to(DEVICE)
                toks = [pipe.tok.decode([t]) for t in ids.input_ids[0].tolist()]
                # locate the fixed suffix "Answer with just the name." near the end of the user turn
                kinds = []
                for i, t in enumerate(toks):
                    s = t.strip().lower()
                    if i == 0 or s == "":
                        kinds.append("sink")
                    elif t.strip() in TEMPLATE or s in {"user", "assistant"}:
                        kinds.append("template")
                    else:
                        kinds.append("content")
                txt = "".join(toks)
                cut = txt.rfind("Answer with just the name.")
                pos = 0
                end = cut + len("Answer with just the name.")
                for i, t in enumerate(toks):                      # tokens OVERLAPPING the suffix are template too
                    if cut >= 0 and pos < end and pos + len(t) > cut and kinds[i] == "content":
                        kinds[i] = "template"
                    pos += len(t)
                assert cut < 0 or not any(k == "content" and t.strip() == "Answer" for k, t in zip(kinds, toks)), "suffix leak"
                with torch.no_grad():
                    att = pipe.model(**ids, output_attentions=True).attentions
                for (l, h) in heads:
                    w = att[l][0, h, -1].float().tolist()
                    for wi, k, t in zip(w, kinds, toks):
                        acc[(l, h)][k] += wi / len(recall)
                        if k == "content":
                            acc[(l, h)]["tokens"][t.strip()] += wi
            out[concept] = {f"L{l}H{h}": {"content": round(a["content"], 3), "template": round(a["template"], 3),
                                          "sink": round(a["sink"], 3),
                                          "top_content_tokens": [(t, round(v / len(recall), 3)) for t, v in a["tokens"].most_common(6)]}
                            for (l, h), a in acc.items()}
            line = " | ".join(f"{k}: content {v['content']:.2f} sink {v['sink']:.2f} -> {[t for t, _ in v['top_content_tokens'][:4]]}"
                              for k, v in out[concept].items())
            print(f"{concept:<24} {line}", flush=True)
    finally:
        pipe.model.set_attn_implementation("sdpa")
    tag = sys.argv[1].replace("\\", "/").split("/")[-1].replace("head_logodds_", "head_looks_")
    open(sys.argv[1].replace("head_logodds_", "head_looks_"), "w", encoding="utf-8").write(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
