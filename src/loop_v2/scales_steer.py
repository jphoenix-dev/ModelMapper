"""Two-ended neurons, step 4 (causal; GPU): does pushing a neuron one way raise one end's answers and lower the other's?

The map session's SCALES.md (method + pass rule written before any data; James approved: "Agree further experiments").
Steps 1-3 (Luna / Jev, API only) are in D:/already-clean-data/scales/. Candidates: steer_candidates.json - 4 strict
scales, 10 near-scales, 10 'unrelated' candidates.
Per neuron: end A / end B concept = the concepts the usage check used (cache useA / useB), else each sign's strongest
run-1 concept (drop >= 0.5); each end's sign = the neuron's activation at the answer position in that concept's run-1 map.
On each concept's run-1 recall questions: the neuron's activation + s x k x peak at the ANSWER POSITION only (the last prompt
token; one prompt per pass so the position is exact),
k in 1, 2, 4, s = +/- towards each end; peak = max(|max|, |min|) over 10M web tokens (data/stats). Measured: change in
each concept's exact answer log-odds (mean over its prompts). Control: a random same-layer neuron at its own peak, same k.
Pass rule (fixed before running):
  'moves' = |mean change| >= 0.5 log-odds.
  scale pattern at k: pushing towards A gives dA >= +0.5 and dB <= -0.5, AND pushing towards B gives dB >= +0.5 and
  dA <= -0.5; the control at that k moves neither concept in either direction (all four |d| < 0.5).
  two-job pattern at k: each push raises its own end (>= +0.5) and leaves the other end within +-0.5.
  A neuron is classified by the smallest k at which a pattern holds (else 'none').
Predictions (stated before running):
  P1 >= 3 of the 4 strict scales show the scale pattern.
  P2 the near-scales show it less often than the strict ones (rate below P1's).
  P3 <= 1 of the 10 'unrelated' candidates shows the scale pattern.
  P4 the control moves neither concept (all |d| < 0.5) in >= 90% of (neuron, k) cases.
Change before this run (18 Oct): the first run added the push at EVERY position and was invalid - the random control moved
the answers in 65/72 cases (peak x k at every token wrecks the model); kept as steer_allpos_invalid.jsonl. SCALES.md does
not fix the position; k, the pass rule and the predictions are unchanged.
Output: D:/already-clean-data/scales/steer.jsonl, steer_summary.txt.  Usage: python src/loop_v2/scales_steer.py
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
DATA = Path(os.environ.get("AC_DATA_DIR", "D:/already-clean-data"))
SC = DATA / "scales"
STATS = Path(os.environ.get("AC_STATS", "C:/dev/already-clean/data/stats_Qwen3-8B.pt"))
KS = [1, 2, 4]
MOVE = 0.5


def steer_at(model, layer, neuron, value, pos):
    """Add value to one neuron at one position (batch of 1)."""
    import contextlib
    from neurons import mlps

    @contextlib.contextmanager
    def cm():
        def hook(mod, args):
            x = args[0].clone(); x[0, pos, neuron] += torch.tensor(value, dtype=x.dtype, device=x.device); return (x,)
        h = mlps(model)[layer].down_proj.register_forward_pre_hook(hook)
        try:
            yield
        finally:
            h.remove()
    return cm()


def main():
    from config import DEFAULT_MODEL, model_path
    if not os.path.isdir(model_path(DEFAULT_MODEL)):
        raise SystemExit("set MODEL_MAPPER_MODELS (no local model)")
    from neurons import steer
    from pipeline_fast import FastPipeline
    cand = json.loads((SC / "steer_candidates.json").read_text(encoding="utf-8"))
    groups = {u: g for g, us in (("strict", cand["strict_scales"]), ("near", cand["near_scales"]), ("unrelated", cand["unrelated_candidates"])) for u in us}
    cache = {}
    for line in (SC / "scales.cache.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line); cache[r["k"]] = r["v"]
    verd = {v["n"]: v for v in json.loads((SC / "verdicts.json").read_text(encoding="utf-8"))}
    recs = {}
    for f in sorted(glob.glob(str(DATA / "loop_v2/run1/results/**/records.jsonl"), recursive=True)):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if (r.get("row") or {}).get("status") == "ok":
                recs[r["concept"]] = r
    use = {}                                    # neuron -> {concept: (drop, act)} over run 1
    for c, r in recs.items():
        for x in r["rank"]["candidates_all200"]:
            if x["neuron"] in groups and (x.get("single_drop") or 0) >= 0.5:
                use.setdefault(x["neuron"], {})[c] = (x["single_drop"], x.get("act_answer_pos") or 0.0)
    st = torch.load(STATS, map_location="cpu", weights_only=False)
    pipe = FastPipeline(DEFAULT_MODEL, mode="add", reply_only=True); m = pipe.model; W = pipe.W
    f = open(SC / "steer.jsonl", "w", encoding="utf-8"); rows = []
    for u, grp in groups.items():
        l, n = int(u[1:].split(":")[0]), int(u.split(":")[1]); U = use.get(u, {})
        ends = {}
        for end, key in (("A", f"useA|{u}"), ("B", f"useB|{u}")):
            c = (cache.get(key) or {}).get("concept")
            if c in U:
                ends[end] = c
        pos = sorted([c for c, (d, a) in U.items() if a > 0], key=lambda c: -U[c][0]); neg = sorted([c for c, (d, a) in U.items() if a < 0], key=lambda c: -U[c][0])
        if "A" not in ends or "B" not in ends or (U[ends["A"]][1] > 0) == (U[ends["B"]][1] > 0):
            if not pos or not neg:
                rows.append({"neuron": u, "group": grp, "error": "no concept on one of the two signs"}); continue
            ends = {"A": pos[0], "B": neg[0]}; picked = "strongest per sign"
        else:
            picked = "usage-check concepts"
        sgn = {e: (1.0 if U[c][1] > 0 else -1.0) for e, c in ends.items()}
        peak = float(max(st["max"][l, n], -st["min"][l, n]))
        rng = random.Random(f"scales|{u}"); cn = rng.randrange(W)
        while f"L{l}:{cn}" in U:
            cn = rng.randrange(W)
        cpeak = float(max(st["max"][l, cn], -st["min"][l, cn]))
        res = {"neuron": u, "group": grp, "ends": ends, "end_sign": sgn, "picked": picked, "peak": peak, "control": f"L{l}:{cn}", "control_peak": cpeak,
               "descriptions": {"A_or_B_luna": [verd.get(u, {}).get("a"), verd.get(u, {}).get("b")]}, "d": {}}
        for e, c in ends.items():
            r = recs[c]["rank"]; pairs = [(q, r["answer_ids"]) for q in r["prompts"]]
            poss = [len(pipe.tok.encode(q)) - 1 for q, _ in pairs]
            base = [pipe.batch_logodds([pr])[0] for pr in pairs]
            for k in KS:
                for towards in ("A", "B"):
                    s = sgn[towards]
                    for unit, nn, pk in (("neuron", n, peak), ("control", cn, cpeak)):
                        lo = []
                        for pr, ps in zip(pairs, poss):
                            with steer_at(m, l, nn, s * k * pk, ps):
                                lo.append(pipe.batch_logodds([pr])[0])
                        res["d"][f"{unit}|{e}|k{k}|to{towards}"] = [x - b for x, b in zip(lo, base)]
        mean = lambda key: stx.mean(res["d"][key])
        cls, ctrl_ok = "none", {}
        for k in KS:
            dA_A, dB_A = mean(f"neuron|A|k{k}|toA"), mean(f"neuron|B|k{k}|toA"); dA_B, dB_B = mean(f"neuron|A|k{k}|toB"), mean(f"neuron|B|k{k}|toB")
            c4 = [mean(f"control|{e}|k{k}|to{t}") for e in "AB" for t in "AB"]; ctrl_ok[k] = all(abs(x) < MOVE for x in c4)
            scale = dA_A >= MOVE and dB_A <= -MOVE and dB_B >= MOVE and dA_B <= -MOVE and ctrl_ok[k]
            jobs = dA_A >= MOVE and abs(dB_A) < MOVE and dB_B >= MOVE and abs(dA_B) < MOVE and ctrl_ok[k]
            res.setdefault("by_k", {})[k] = {"toA": {"dA": dA_A, "dB": dB_A}, "toB": {"dA": dA_B, "dB": dB_B}, "control": c4, "control_ok": ctrl_ok[k],
                                             "scale": scale, "two_jobs": jobs}
            if cls == "none" and (scale or jobs):
                cls = f"{'scale' if scale else 'two_jobs'}@k{k}"
        res["class"] = cls; res["control_ok"] = ctrl_ok
        f.write(json.dumps(res, ensure_ascii=False) + "\n"); f.flush(); rows.append(res)
        b2 = res["by_k"][2]
        print(f"{u:<10} {grp:<9} {cls:<12} k2 toA dA {b2['toA']['dA']:+.2f} dB {b2['toA']['dB']:+.2f} | toB dA {b2['toB']['dA']:+.2f} dB {b2['toB']['dB']:+.2f} | "
              f"{ends['A']} / {ends['B']}", flush=True)
    ok = [r for r in rows if "class" in r]
    rate = lambda g: [r for r in ok if r["group"] == g]
    sc = lambda g: sum(r["class"].startswith("scale") for r in rate(g))
    cc = [v for r in ok for v in r["control_ok"].values()]
    L = [f"Scales step 4: {len(ok)} neurons measured ({len(rows) - len(ok)} without a concept on both signs)"]
    for g in ("strict", "near", "unrelated"):
        L.append(f"  {g:<9}: scale pattern {sc(g)}/{len(rate(g))}; two-job pattern {sum(r['class'].startswith('two_jobs') for r in rate(g))}/{len(rate(g))}; "
                 f"none {sum(r['class'] == 'none' for r in rate(g))}")
    L.append(f"  control moved neither concept in {sum(cc)}/{len(cc)} (neuron, k) cases")
    p1 = sc("strict") >= 3; p2 = (sc("near") / max(1, len(rate("near")))) < (sc("strict") / max(1, len(rate("strict"))))
    p3 = sc("unrelated") <= 1; p4 = sum(cc) >= 0.9 * len(cc)
    L.append(f"  P1 (>= 3 of 4 strict): {'HOLDS' if p1 else 'FAILS'} | P2 (near < strict rate): {'HOLDS' if p2 else 'FAILS'} | "
             f"P3 (<= 1 unrelated): {'HOLDS' if p3 else 'FAILS'} | P4 (control quiet >= 90%): {'HOLDS' if p4 else 'FAILS'}")
    for r in ok:
        L.append(f"  {r['neuron']:<10} {r['group']:<9} {r['class']:<12} A={r['ends']['A']} | B={r['ends']['B']} ({r['picked']})")
    txt = "\n".join(L); (SC / "steer_summary.txt").write_text(txt + "\n", encoding="utf-8"); print(txt)


if __name__ == "__main__":
    main()
