"""Test for ISSUES #35: pipeline_fast.batch_logodds (exact) vs the old clamped log(p/(1-p)) on batch_probs. No API.

Three checks on every recall prompt of the given concepts:
  1. ARITHMETIC (must pass): batch_logodds on the prompt alone (batch of 1) vs an independent float64 formula on the
     same unbatched logits - for one-token answers logit_a - logsumexp(other logits), for longer answers
     log P - log(1 - P) with P from float64 log-softmax. Must agree within 1e-6.
  2. NO CHANGE AWAY FROM SATURATION (must pass): batched new vs batched old agree within 0.01 wherever old < 10
     (above ~10 the OLD float32 value itself carries rounding error).
  3. NOISE (reported, not a pass/fail): batched vs unbatched difference of the same prompt (bf16 arithmetic depends on
     batch shape), for new and for old - the precision any log-odds near saturation can have.
Plus: every prompt where old hit the 20.72 cap, with the new value.
Usage: RECALL_VARIANT=V3 ANSWER_DISCOVERY=0 CONCEPT_OUT_DIR=<dir> python src/logodds_exact_check.py "<concept>" [...]
"""
import math
import os
import statistics as stx
import sys

import torch

from config import DEFAULT_MODEL
from neurons import DEVICE
from pipeline_fast import FastPipeline


def old_logit(p):
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


def reference(pipe, q, ans):
    ids = pipe.tok.encode(q) + ans[:-1]
    with torch.no_grad():
        lg = pipe.model(input_ids=torch.tensor([ids], device=DEVICE)).logits[0].double()
    start = len(pipe.tok.encode(q)) - 1
    if len(ans) == 1:
        x = lg[start]; a = ans[0]
        return (x[a] - torch.logsumexp(torch.cat([x[:a], x[a + 1:]]), 0)).item()
    s = sum(lg[start + k].log_softmax(-1)[t].item() for k, t in enumerate(ans))
    return math.log(math.exp(s)) - math.log(1 - math.exp(s))


def main():
    assert os.environ.get("ANSWER_DISCOVERY") == "0"
    pipe = FastPipeline(DEFAULT_MODEL, mode="add", reply_only=True)
    arith = low = 0.0; noise_new, noise_old = [], []; capped = []; n = 0
    for concept in sys.argv[1:]:
        spec, tid, recall, _ = pipe.make_prompts(concept)
        pairs = [(q, tid) for q, _ in recall]
        old_b = [old_logit(p) for p in pipe.batch_probs(pairs)]; new_b = pipe.batch_logodds(pairs)
        for (q, _), ob, nb in zip(recall, old_b, new_b):
            n += 1
            n1 = pipe.batch_logodds([(q, tid)])[0]; o1 = old_logit(pipe.batch_probs([(q, tid)])[0]); r = reference(pipe, q, tid)
            arith = max(arith, abs(n1 - r))
            if ob < 10:
                low = max(low, abs(nb - ob))
            if nb > 8:
                noise_new.append(abs(nb - n1)); noise_old.append(abs(ob - o1))
            if ob > 20.7:
                capped.append((concept, ob, nb, n1))
    print(f"{n} prompts")
    print(f"1. arithmetic: max |batch_logodds (alone) - independent float64 formula| = {arith:.2e}")
    print(f"2. max |new - old| (batched) where old < 10: {low:.4f}")
    if noise_new:
        print(f"3. noise, prompts with log-odds > 8 (n={len(noise_new)}): |batched - alone| median / max  new "
              f"{stx.median(noise_new):.3f} / {max(noise_new):.3f}   old {stx.median(noise_old):.3f} / {max(noise_old):.3f}")
    for c, ob, nb, n1 in capped:
        print(f"   capped before: {c:<22} old {ob:.2f} -> new {nb:.2f} (alone {n1:.2f})")
    assert arith < 1e-6, "batch_logodds disagrees with the independent formula"
    assert low < 0.01, "batch_logodds changes values away from saturation"
    print("PASS", flush=True)


if __name__ == "__main__":
    main()
