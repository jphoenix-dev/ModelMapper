"""How many ATTENTION HEADS does recall depend on? The head counterpart of logodds_check.py v2 (attention study, step 1).

For each concept (saved specs, same answers as the neuron log-odds runs: V3, answer discovery off):
  1. attribution ranks all 36 x 32 = 1,152 heads (activation x gradient on each head's slice of o_proj's input);
  2. the top-50 are re-ranked by MEASURED single-head knockout (zeroing the head's output at every position);
  3. knock out the top-k heads (k = 1..50) in measured order and measure the answer's log-odds drop per prompt.
Same three views as for neurons: k_mean / k_median (drop >= 3, i.e. odds ~20x lower) and k_flip (P < 0.5 on at least
half the prompts the model got right). Control: random heads from the same layers. Saves each concept's top-10
measured heads for the shared-vs-specific analysis (step 2). No API calls.

Usage: RECALL_VARIANT=V3 ANSWER_DISCOVERY=0 CONCEPT_OUT_DIR=<dir> python src/head_logodds.py <tag> "<concept>" [...]
"""
import json
import math
import random
import sys

from config import DEFAULT_MODEL
import concept_pipeline as cp
from pipeline_fast import FastPipeline

KS = [1, 2, 3, 5, 10, 20, 50]


def logit(p):
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


def main():
    tag = sys.argv[1]
    pipe = FastPipeline(DEFAULT_MODEL)
    H = pipe.model.config.num_attention_heads
    out_p = cp.OUT / f"head_logodds_{tag}.json"
    out = json.loads(out_p.read_text(encoding="utf-8")) if out_p.exists() else {}
    for concept in [c.strip() for c in sys.argv[2:]]:
        if concept in out:
            continue
        spec, tid, recall, _ = pipe.make_prompts(concept)
        if len(recall) < 5:
            print(f"{concept}: not recalled", flush=True); continue
        rp = [q for q, _ in recall]; pairs = [(q, tid) for q in rp]
        base = [logit(p) for p in pipe.batch_probs(pairs)]; right = [b > 0 for b in base]

        def per_prompt(heads):
            ps = pipe.batch_probs(pairs, pipe.ablate_heads(heads) if heads else None)
            return [b - logit(p) for b, p in zip(base, ps)], ps
        A = pipe.attribute(rp, tid)
        cand = [tuple(h) for h in A["top_heads"][:50]]
        single = sorted(cand, key=lambda h: -sum(per_prompt([h])[0]))

        def curve(order):
            rows, km, kd, kf = [], None, None, None
            for k in KS:
                d, ps = per_prompt(order[:k])
                mean = sum(d) / len(d); med = sorted(d)[len(d) // 2]
                flip = sum(p < 0.5 for p, r in zip(ps, right) if r) / max(1, sum(right))
                rows.append({"k": k, "mean_drop": round(mean, 2), "median_drop": round(med, 2), "share_flipped": round(flip, 2)})
                km = km or (k if mean >= 3 else None); kd = kd or (k if med >= 3 else None); kf = kf or (k if flip >= .5 else None)
            return rows, km, kd, kf
        m_rows, km, kd, kf = curve(single)
        rng = random.Random(4)
        r_rows, _, _, _ = curve([(l, rng.randrange(H)) for l, _ in single])
        out[concept] = {"baseline_mean_logodds": round(sum(base) / len(base), 2), "k_mean": km, "k_median": kd, "k_flip": kf,
                        "measured": m_rows, "random": r_rows, "top10_heads": single[:10],
                        "top10_single_drop": [round(sum(per_prompt([h])[0]) / len(rp), 2) for h in single[:10]]}
        print(f"{concept:<26} heads: k_mean {km} k_median {kd} k_flip {kf} | drop@5 {m_rows[3]['mean_drop']} vs random "
              f"{r_rows[3]['mean_drop']} | top heads {[f'L{l}H{h}' for l, h in single[:5]]}", flush=True)
        out_p.write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
