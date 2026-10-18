"""Attention study, step 4: do the key heads FEED the concept neurons? (no API calls)

For each concept: its top-5 measured heads (head_logodds results) and its top-10 attributed neurons (the concept run's
top_neurons, with the sign each takes on recall). On every recall prompt, read each neuron's activation at the answer
position (last prompt token) with:
  (a) nothing switched off; (b) the top-5 heads switched off; (c) 5 random heads from the same layers switched off.
Report each neuron's signed activation relative to baseline: 1.0 = unchanged, 0 = silenced.
Built-in sanity check: neurons in layers BELOW the lowest switched-off head cannot be affected (a head's output only
reaches later layers). Their ratio must be exactly 1.0, or the measurement is broken.
Also tags each neuron as "says the answer" (its direct output effect ranks the answer's first token in the top 50),
for the naming question.

Usage: RECALL_VARIANT=V3 ANSWER_DISCOVERY=0 CONCEPT_OUT_DIR=<dir> python src/heads_to_neurons.py <head_logodds json> <tag>
"""
import json
import random
import sys
from pathlib import Path

import torch

from config import DEFAULT_MODEL
import concept_pipeline as cp
from neurons import DEVICE, capture
from pipeline_fast import FastPipeline


def main():
    res = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")); tag = sys.argv[2]
    pipe = FastPipeline(DEFAULT_MODEL)
    H = pipe.model.config.num_attention_heads
    W_U = pipe.model.lm_head.weight.float(); g = pipe.model.model.norm.weight.float()
    out_p = cp.OUT / f"heads_to_neurons_{tag}.json"
    out = json.loads(out_p.read_text(encoding="utf-8")) if out_p.exists() else {}
    for f in sorted(cp.OUT.glob("*__clamp_reply.json")):
        x = json.loads(f.read_text(encoding="utf-8"))
        if not isinstance(x, dict) or x.get("status") != "ok" or x["concept"].strip() not in res or x["concept"].strip() in out:
            continue
        c = x["concept"].strip()
        spec, tid, recall, _ = pipe.make_prompts(c)
        heads = [tuple(h) for h in res[c]["top10_heads"][:5]]
        rng = random.Random(5)
        rand = [(l, rng.randrange(H)) for l, _ in heads]
        neurons = [(t["layer"], t["neuron"], -1.0 if t.get("sign_on_recall") == "neg" else 1.0) for t in x["top_neurons"][:10]]
        layers = sorted({l for l, _, _ in neurons})

        def acts(ctx):
            vals = {(l, n): 0.0 for l, n, _ in neurons}
            for q, _ in recall:
                ids = pipe.tok(q, return_tensors="pt").to(DEVICE)
                with torch.no_grad(), (ctx() if ctx else _null()), capture(pipe.model, layers) as s:
                    pipe.model.model(input_ids=ids.input_ids)
                for l, n, sg in neurons:
                    vals[(l, n)] += sg * s["act"][l][0, -1, n].item() / len(recall)
            return vals
        base = acts(None)
        abl = acts(lambda: pipe.ablate_heads(heads))
        rnd = acts(lambda: pipe.ablate_heads(rand))
        first = tid[0]; lo = min(l for l, _ in heads)
        rows = []
        for l, n, sg in neurons:
            col = pipe.model.model.layers[l].mlp.down_proj.weight[:, n].float()
            logits = sg * (W_U @ (col * g))
            says = int((logits > logits[first]).sum().item()) < 50
            b = base[(l, n)]
            rows.append({"neuron": f"L{l}:{n}", "layer": l, "says_answer": says, "baseline": round(b, 4),
                         "ratio_heads_off": round(abl[(l, n)] / b, 3) if abs(b) > 1e-6 else None,
                         "ratio_random_off": round(rnd[(l, n)] / b, 3) if abs(b) > 1e-6 else None,
                         "upstream_of_heads": l < lo})
        up = [r for r in rows if r["upstream_of_heads"] and r["ratio_heads_off"] is not None]
        assert all(abs(r["ratio_heads_off"] - 1) < 1e-3 and abs(r["ratio_random_off"] - 1) < 1e-3 for r in up), f"upstream neurons changed for {c}: {up}"
        out[c] = {"heads": [f"L{l}H{h}" for l, h in heads], "random_heads": [f"L{l}H{h}" for l, h in rand], "neurons": rows}
        dn = [(l, n) for l, n, _ in neurons if l >= lo]
        # robust summary: total signed activation after / before (a mean of per-neuron ratios blows up near zero
        # baselines), plus the median per-neuron ratio
        tb = sum(base[u] for u in dn); th = sum(abl[u] for u in dn); tr = sum(rnd[u] for u in dn)
        med = lambda xs: sorted(xs)[len(xs) // 2] if xs else None  # noqa: E731
        rh = [abl[u] / base[u] for u in dn if base[u] > 0.05]; rr = [rnd[u] / base[u] for u in dn if base[u] > 0.05]
        out[c]["total_ratio_heads_off"] = round(th / tb, 3) if tb else None
        out[c]["total_ratio_random_off"] = round(tr / tb, 3) if tb else None
        out[c]["median_ratio_heads_off"] = med(rh); out[c]["median_ratio_random_off"] = med(rr)
        if not dn or not tb:
            print(f"{c:<26} no downstream neurons with activation (all top neurons sit below the heads) - skipped in summary", flush=True)
        else:
            print(f"{c:<26} {len(dn)} downstream neurons: total activation with top heads off {th / tb:.2f}x (median "
                  f"{med(rh) if rh else float('nan'):.2f}) | random heads off {tr / tb:.2f}x (median {med(rr) if rr else float('nan'):.2f}) "
                  f"| upstream sanity rows {len(up)} unchanged", flush=True)
        (cp.OUT / f"heads_to_neurons_{tag}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")   # save as we go
    (cp.OUT / f"heads_to_neurons_{tag}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    main()
