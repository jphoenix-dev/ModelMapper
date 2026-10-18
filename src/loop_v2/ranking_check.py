"""Loop v3 step C: does ranking candidates by log-odds instead of log-prob attribution change run 1's maps? (GPU; no API)

map_core ranks all neurons by activation x gradient of the answer's LOG-PROB summed over prompts; the gradient of log-odds
is the same per prompt times 1/(1-P), so log-prob ranking down-weights prompts Qwen is already sure of (and float32 zeroes
prompts with log-odds > ~16.6: 0.1% of run 1's 52,593 prompts). One attribution pass gives both rankings (per-prompt maps
reweighted by 1/(1-P_p)).
For 50 random run-1 concepts (seed 331): run 1's own prompts and answer ids; top-200 under each ranking; overlap; and how
many of run 1's final top-30 (ranked by knockout) fall inside the log-odds top-200.
Prediction (stated before running): top-200 overlap >= 90% (median) and >= 29 of the final top-30 inside the log-odds
top-200 for >= 90% of concepts -> keep log-prob ranking in v3 (no switch).
Output: D:/already-clean-data/loop_v3/ranking_check.jsonl, .txt.  Usage: python src/loop_v2/ranking_check.py
"""
import glob
import json
import os
import random
import statistics as stx
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
OUT = Path(os.environ.get("AC_DATA_DIR", "D:/already-clean-data")) / "loop_v3"
RUN1 = Path(os.environ.get("AC_DATA_DIR", "D:/already-clean-data")) / "loop_v2" / "run1" / "results"


def main():
    from config import DEFAULT_MODEL, model_path
    if not os.path.isdir(model_path(DEFAULT_MODEL)):
        raise SystemExit("set MODEL_MAPPER_MODELS (no local model)")
    from pipeline_fast import FastPipeline
    recs = [json.loads(l) for f in sorted(glob.glob(str(RUN1 / "**" / "records.jsonl"), recursive=True))
            for l in open(f, encoding="utf-8") if l.strip()]
    recs = [r for r in recs if (r.get("row") or {}).get("status") == "ok"]
    sample = random.Random(331).sample(recs, 50)
    pipe = FastPipeline(DEFAULT_MODEL, mode="add", reply_only=True); W = pipe.W
    OUT.mkdir(parents=True, exist_ok=True); f = open(OUT / "ranking_check.jsonl", "w", encoding="utf-8"); rows = []
    for r in sample:
        rk = r["rank"]; prompts = rk["prompts"]; tid = rk["answer_ids"]
        A = pipe.attribute(prompts, tid)
        P = torch.tensor(pipe.batch_probs([(p, tid) for p in prompts]), dtype=torch.float64)
        w = 1.0 / (1.0 - P).clamp_min(1e-12)                       # log-odds weight per prompt (reported, not capped silently)
        lp = torch.stack(A["per_prompt"]).double().sum(0)           # log-prob attribution (map_core's ranking, up to 1/n)
        lo = (torch.stack(A["per_prompt"]).double() * w[:, None, None]).sum(0)
        t_lp = {divmod(int(i), W) for i in lp.flatten().topk(200).indices}
        t_lo = {divmod(int(i), W) for i in lo.flatten().topk(200).indices}
        run1_200 = {tuple(map(int, x["neuron"][1:].split(":"))) for x in rk["candidates_all200"]}
        top30 = [tuple(map(int, x["neuron"][1:].split(":"))) for x in rk["top"][:30]]
        row = {"concept": r["concept"], "n_prompts": len(prompts), "p": [round(float(x), 6) for x in P],
               "overlap_200": len(t_lp & t_lo) / 200, "reproduces_run1_200": len(t_lp & run1_200) / 200,
               "top30_in_logodds_200": sum(u in t_lo for u in top30), "max_weight_ratio": float(w.max() / w.min())}
        f.write(json.dumps(row) + "\n"); f.flush(); rows.append(row)
        print(row["concept"], row["overlap_200"], row["top30_in_logodds_200"], flush=True)
    md = stx.median
    ok = md(x["overlap_200"] for x in rows) >= 0.9 and sum(x["top30_in_logodds_200"] >= 29 for x in rows) >= 0.9 * len(rows)
    txt = "\n".join([f"Ranking check (50 run-1 concepts): top-200 overlap log-prob vs log-odds median {md(x['overlap_200'] for x in rows):.2f} "
                     f"(min {min(x['overlap_200'] for x in rows):.2f}); log-prob top-200 reproduces run 1's candidates median {md(x['reproduces_run1_200'] for x in rows):.2f}",
                     f"run 1's final top-30 inside the log-odds top-200: median {md(x['top30_in_logodds_200'] for x in rows)}; >= 29 for "
                     f"{sum(x['top30_in_logodds_200'] >= 29 for x in rows)}/{len(rows)} concepts",
                     f"prompt weight ratio (max/min 1/(1-P)) median {md(x['max_weight_ratio'] for x in rows):.1f}",
                     f"PREDICTION (overlap >= 0.9; >= 29/30 inside for >= 90%): {'HOLDS - keep log-prob ranking' if ok else 'FAILS - switch v3 to log-odds ranking'}"])
    (OUT / "ranking_check.txt").write_text(txt + "\n", encoding="utf-8"); print(txt)


def disputed():
    """Stage 2 (fair test, written after stage 1 'failed' - stage 1 is biased towards log-prob because run 1's top 30 was
    chosen by knockouts FROM the log-prob list). For the same 50 concepts, every candidate in the log-odds top-200 but not
    in run 1's 200 is zeroed alone (map_core's own single-knockout procedure); does it beat run 1's 30th-best drop?
    Prediction (stated before running): median <= 1 such candidate per concept enters the top 30, and none in the top 10 for
    >= 90% of concepts -> keep the log-prob ranking (no switch). Otherwise switch v3 to the union of both top-200 lists."""
    from config import DEFAULT_MODEL, model_path
    if not os.path.isdir(model_path(DEFAULT_MODEL)):
        raise SystemExit("set MODEL_MAPPER_MODELS (no local model)")
    from neurons import steer
    from pipeline_fast import FastPipeline
    done = {json.loads(l)["concept"] for l in (OUT / "ranking_check.jsonl").read_text(encoding="utf-8").splitlines()}
    recs = [json.loads(l) for f in sorted(glob.glob(str(RUN1 / "**" / "records.jsonl"), recursive=True))
            for l in open(f, encoding="utf-8") if l.strip()]
    recs = [r for r in recs if r["concept"] in done]
    pipe = FastPipeline(DEFAULT_MODEL, mode="add", reply_only=True); W = pipe.W; m = pipe.model
    f = open(OUT / "ranking_disputed.jsonl", "w", encoding="utf-8"); rows = []
    for r in recs:
        rk = r["rank"]; prompts = rk["prompts"]; tid = rk["answer_ids"]; pairs = [(q, tid) for q in prompts]
        A = pipe.attribute(prompts, tid)
        P = torch.tensor(pipe.batch_probs(pairs), dtype=torch.float64); w = 1.0 / (1.0 - P).clamp_min(1e-12)
        lo = (torch.stack(A["per_prompt"]).double() * w[:, None, None]).sum(0)
        t_lo = [divmod(int(i), W) for i in lo.flatten().topk(200).indices]
        run1 = {tuple(map(int, x["neuron"][1:].split(":"))): x["single_drop"] for x in rk["candidates_all200"]}
        new = [u for u in t_lo if u not in run1]
        base = pipe.batch_logodds(pairs)
        dd = {}
        for (l, n) in new:
            los = pipe.batch_logodds(pairs, steer(m, [(l, n, 0.0)]))
            dd[(l, n)] = sum(b - x for b, x in zip(base, los)) / len(prompts)
        ranked = sorted(run1.values(), reverse=True); thr30, thr10 = ranked[29], ranked[9]
        enter = sorted(((v, u) for u, v in dd.items() if v > thr30), reverse=True)
        row = {"concept": r["concept"], "n_new": len(new), "n_enter_top30": len(enter), "n_enter_top10": sum(v > thr10 for v, _ in enter),
               "thr30": thr30, "new_best": max(dd.values()) if dd else None,
               "entering": [{"neuron": f"L{u[0]}:{u[1]}", "drop": round(v, 4), "logodds_rank": t_lo.index(u) + 1} for v, u in enter]}
        f.write(json.dumps(row) + "\n"); f.flush(); rows.append(row)
        print(r["concept"], row["n_new"], row["n_enter_top30"], row["n_enter_top10"], flush=True)
    md = stx.median
    ok = md(x["n_enter_top30"] for x in rows) <= 1 and sum(x["n_enter_top10"] == 0 for x in rows) >= 0.9 * len(rows)
    txt = "\n".join([f"Disputed candidates (log-odds top-200 not in run 1's 200; {len(rows)} concepts): median {md(x['n_new'] for x in rows)} per concept",
                     f"  entering run 1's top 30: median {md(x['n_enter_top30'] for x in rows)}, mean {stx.mean(x['n_enter_top30'] for x in rows):.2f}; "
                     f"concepts with >= 1: {sum(x['n_enter_top30'] > 0 for x in rows)}/{len(rows)}; entering the top 10: {sum(x['n_enter_top10'] for x in rows)} "
                     f"in {sum(x['n_enter_top10'] > 0 for x in rows)} concepts",
                     f"PREDICTION (median <= 1 enter top 30; none in top 10 for >= 90%): {'HOLDS - keep log-prob ranking' if ok else 'FAILS - use the union of both top-200 lists in v3'}"])
    (OUT / "ranking_disputed.txt").write_text(txt + "\n", encoding="utf-8"); print(txt)


if __name__ == "__main__":
    {"check": main, "disputed": disputed}[sys.argv[1] if len(sys.argv) > 1 else "check"]()
