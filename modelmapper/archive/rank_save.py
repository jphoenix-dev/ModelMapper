"""Save each recalled concept's measured top neurons, with one definition for every concept set (no API calls).

The judge-free waves (wave_free.py) ranked neurons but kept only the counts (k_mean, k_flip...), not WHICH neurons.
This repeats wave_free.py's ranking exactly - attribution top-200, then each neuron switched off alone and ordered by
the summed log-odds drop on the recall prompts - and saves the top 30 with their single-neuron drops and their mean
activation at the answer position (signed). Same pipeline settings as wave_free.py (mode add, reply-only).

Checks (a failure is recorded as 'mismatch' and printed; prompt-count mismatches are not ranked):
  - the spec must already hold the rewritten recall prompts (otherwise make_prompts would call the API) and the
    run must have ANSWER_DISCOVERY=0 (no API); both asserted
  - the number of recall prompts must equal the original run's 'recall' (wave_free.json), when the set has one
  - k_mean / k_flip recomputed here must equal the original run's, when the set has them
Log-odds use wave_free.py's logit() (p clamped to [1e-9, 1-1e-9]) so the ranking matches the original; that clamp
caps log-odds at 20.72, which logodds_cap_audit.py found reached by 3/869 recall prompts (recorded, ISSUES).
Usage: RECALL_VARIANT=V3 ANSWER_DISCOVERY=0 CONCEPT_OUT_DIR=<dir> python src/rank_save.py <tag> "<concept>" [...]
       -> <dir>/rank_save_<tag>.json
"""
import json
import os

import numpy as np
import sys

import concept_pipeline as cp
from config import DEFAULT_MODEL
from neurons import steer
from pipeline_fast import FastPipeline
from wave_free import KS, logit, p_of

TOPN = 30
EXACT = os.environ.get("LOGODDS") == "exact"     # ISSUES #35 fix: exact log-odds (pipe.batch_logodds) instead of clamped


def main():
    tag = sys.argv[1]
    assert os.environ.get("ANSWER_DISCOVERY") == "0", "set ANSWER_DISCOVERY=0 (no API calls)"
    pipe = FastPipeline(os.environ.get("MODEL", DEFAULT_MODEL), mode="add", reply_only=True)
    wf_p = cp.OUT / "wave_free.json"
    wf = json.loads(wf_p.read_text(encoding="utf-8")) if wf_p.exists() else {}
    out_p = cp.OUT / f"rank_save_{tag}{'_exact' if EXACT else ''}.json"
    out = json.loads(out_p.read_text(encoding="utf-8")) if out_p.exists() else {}
    for concept in [c.strip() for c in sys.argv[2:]]:
        if concept in out:
            continue
        spec_p = cp.OUT / f"{cp.slug(concept)}_spec.json"
        spec0 = json.loads(spec_p.read_text(encoding="utf-8"))
        assert "recall_v3" in spec0, f"{concept}: spec has no rewritten prompts - make_prompts would call the API"
        spec, tid, recall, _ = pipe.make_prompts(concept)
        orig = wf.get(concept, {})
        if "recall" in orig and len(recall) != orig["recall"]:
            out[concept] = {"mismatch": f"{len(recall)} recall prompts vs {orig['recall']} in the original run"}
            print(f"{concept:<30} MISMATCH: {out[concept]['mismatch']} - not ranked", flush=True)
            out_p.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8"); continue
        rp = [q for q, _ in recall]; pairs = [(q, tid) for q in rp]
        if EXACT:
            base = pipe.batch_logodds(pairs)
        else:
            base = [logit(p) for p in pipe.batch_probs(pairs)]
        right = [b > 0 for b in base]
        A = pipe.attribute(rp, tid); top = A["top"][:200]

        def per_prompt(units):
            ctx = steer(pipe.model, [(l, n, 0.0) for l, n in units]) if units else None
            if EXACT:
                los = pipe.batch_logodds(pairs, ctx)
                return [b - x for b, x in zip(base, los)], [p_of(x) for x in los]
            ps = pipe.batch_probs(pairs, ctx)
            return [b - logit(p) for b, p in zip(base, ps)], ps
        pp = {u: per_prompt([u])[0] for u in top}                     # per-prompt drops for all 200 (kept)
        drop1 = {u: sum(pp[u]) / len(rp) for u in top}
        single = sorted(top, key=lambda u: -drop1[u])
        km = kf = None
        for k in KS:
            d, ps = per_prompt(single[:k]); mean = sum(d) / len(d)
            flip = sum(p < .5 for p, r in zip(ps, right) if r) / max(1, sum(right))
            km = km or (k if mean >= 3 else None); kf = kf or (k if flip >= .5 else None)
        mismatch = None
        if "k_mean" in orig and (km, kf) != (orig["k_mean"], orig["k_flip"]):
            mismatch = f"k_mean/k_flip {km}/{kf} vs original {orig['k_mean']}/{orig['k_flip']}"
            print(f"{concept:<30} MISMATCH: {mismatch} (ranking saved, flagged)", flush=True)
        # save everything (James, 17 Oct): all 200 candidates with their single-neuron drops (per prompt too), and the
        # full per-neuron attribution / answer-position activation maps (L x W, fp16). Per-position attribution maps
        # (~800 MB per concept) are NOT saved - too large; their per-neuron totals are (the 'attr' array).
        arr_dir = cp.OUT / "rank_save_arrays"; arr_dir.mkdir(exist_ok=True)
        np.savez_compressed(arr_dir / f"{cp.slug(concept)}.npz", attr=A["attr"].detach().half().cpu().numpy(),
                            last_act=A["last_act"].detach().half().cpu().numpy(), top200=np.array(top, dtype=np.int32))
        per_prompt_drop = {f"L{l}:{n}": [round(x, 3) for x in pp[(l, n)]] for l, n in single}
        candidates = [{"neuron": f"L{l}:{n}", "single_drop": round(drop1[(l, n)], 3), "attr_rank": top.index((l, n)),
                       "act_answer_pos": round(A["last_act"][l, n].item(), 3)} for l, n in single]
        out[concept] = {"answer_token": pipe.tok.decode([tid[0]]), "recall": len(rp), "k_mean": km, "k_flip": kf,
                        "base_per_prompt": [round(x, 3) for x in base], "prompts": rp, "candidates_all200": candidates,
                        "per_prompt_drop_all200": per_prompt_drop,
                        "baseline_mean_logodds": round(sum(base) / len(base), 3),
                        "top": [{"neuron": f"L{l}:{n}", "single_drop": round(drop1[(l, n)], 3), "attr_rank": top.index((l, n)),
                                 "act_answer_pos": round(A["last_act"][l, n].item(), 3)} for l, n in single[:TOPN]],
                        "matched_original": bool(orig) and mismatch is None, "mismatch": mismatch}
        print(f"{concept:<30} recall {len(rp)} | k_mean {km} k_flip {kf}{' (= original)' if orig else ''} | top "
              f"{[e['neuron'] for e in out[concept]['top'][:5]]}", flush=True)
        out_p.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
