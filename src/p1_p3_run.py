"""PREREG_MAP P1 (independent prompts) and P3 (shared neurons are causally shared). GPU; no API.

P1: for each concept in results/p1_prompts.json: the Wikipedia cloze prompts in the V3 chat format, scored on the
    concept's answer form; prompts the model answers (p >= 0.2) kept; concepts with >= 3 kept are tested. Knock out its
    top-5 neurons (from the Luna-prompt ranking) vs 5 random same-layer neurons (seed 51); exact log-odds drops and
    flips. Verdict per PREREG_MAP: median drop >= 1.0, >= 5x the random median, and top-5 flips >= half the correct
    prompts in >= 40% of tested concepts.
P3: concepts with saved prompts + answer forms (coverage-3: rank_save_fame3.json + wave_free.json, rank_save_fame3m.json
    + map_loop_fame3m.json). Pairs (A, B) whose top-10 lists share exactly one neuron n, n not generic
    (p4_predictions.json); up to 150 pairs (seed 52). On B: drop from n alone; control 1 = A's top-10 neuron nearest in
    rank to n that is not in B's top-10; control 2 = a random neuron from n's layer. Also n on A (sanity).
    Verdict: median drop on B from n >= 0.3 and >= 3x each control's median.
Everything is saved per concept / pair: results/p1_run.json, results/p3_run.json; verdicts in results/p1_p3_summary.txt.
Usage: python src/p1_p3_run.py [p1|p3|both]
"""
import json
import random
import statistics as stx
import sys
from pathlib import Path

import concept_pipeline as cp
from config import DEFAULT_MODEL
from neurons import steer
from pipeline_fast import FastPipeline
from wave_free import p_of

ROOT = Path(__file__).resolve().parents[1]
D3 = ROOT / "results/concepts_fame3"


def chat(pipe, q):
    return pipe.tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False)


def unit(u):
    l, n = (int(x) for x in u[1:].split(":")); return l, n


def drops(pipe, pairs, base, units):
    los = pipe.batch_logodds(pairs, steer(pipe.model, [(l, n, 0.0) for l, n in units]) if units else None)
    return [b - x for b, x in zip(base, los)], [p_of(x) for x in los]


def p1(pipe, L):
    P = json.loads((ROOT / "results/p1_prompts.json").read_text(encoding="utf-8")); res = {}
    for c, v in P.items():
        if len(v["prompts"]) < 3:
            res[c] = {"tested": False, "why": f"{len(v['prompts'])} cloze prompts"}; continue
        tid = pipe.tok.encode(v["answer_form"]); qs = [chat(pipe, q) for q in v["prompts"]]
        ps = pipe.batch_probs([(q, tid) for q in qs]); keep = [q for q, p in zip(qs, ps) if p >= cp.MIN_P]
        if len(keep) < 3:
            res[c] = {"tested": False, "why": f"model answers {len(keep)} of {len(qs)} cloze prompts", "p": [round(x, 4) for x in ps]}; continue
        rk = json.loads((D3 / v["rank_file"]).read_text(encoding="utf-8"))[c]
        top5 = [unit(e["neuron"]) for e in rk["top"][:5]]
        rng = random.Random(f"51|{c}"); rand5 = [(l, rng.randrange(pipe.W)) for l, _ in top5]
        pairs = [(q, tid) for q in keep]; base = pipe.batch_logodds(pairs); right = [b > 0 for b in base]
        dt, pt = drops(pipe, pairs, base, top5); dr, pr = drops(pipe, pairs, base, rand5)
        flip = sum(p < .5 for p, r in zip(pt, right) if r) / max(1, sum(right))
        res[c] = {"tested": True, "n_prompts": len(keep), "base": [round(x, 3) for x in base], "drop_top5": [round(x, 3) for x in dt],
                  "drop_rand5": [round(x, 3) for x in dr], "mean_drop_top5": round(stx.mean(dt), 3), "mean_drop_rand5": round(stx.mean(dr), 3),
                  "flip_share_top5": round(flip, 3), "top5": [f"L{l}:{n}" for l, n in top5], "rand5": [f"L{l}:{n}" for l, n in rand5], "prompts": keep}
        print(f"P1 {c[:30]:<30} n={len(keep)} drop top5 {stx.mean(dt):+.2f} rand {stx.mean(dr):+.2f} flips {flip:.2f}", flush=True)
    (ROOT / "results/p1_run.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    t = [r for r in res.values() if r["tested"]]
    L.append(f"P1 independent prompts: {len(P)} concepts sampled, {len(t)} tested (>= 3 cloze prompts the model answers)")
    if t:
        mt = stx.median(r["mean_drop_top5"] for r in t); mr = stx.median(r["mean_drop_rand5"] for r in t)
        fl = sum(r["flip_share_top5"] >= 0.5 for r in t) / len(t)
        ok = mt >= 1.0 and mt >= 5 * max(mr, 1e-9) and fl >= 0.4
        L.append(f"  median drop top-5 {mt:+.2f} vs random-5 {mr:+.3f}; top-5 flip >= half the correct prompts in {fl:.0%} of concepts")
        L.append(f"  PREDICTION {'HOLDS' if ok else 'FAILS'} (needs median >= 1.0, >= 5x random, flips in >= 40%)")


def p3(pipe, L):
    gen = set(json.loads((ROOT / "results/p4_predictions.json").read_text(encoding="utf-8"))["generic_neurons"])
    info = {}
    wf = json.loads((D3 / "wave_free.json").read_text(encoding="utf-8")); ml = json.loads((D3 / "map_loop_fame3m.json").read_text(encoding="utf-8"))
    for rf, rows in (("rank_save_fame3.json", wf), ("rank_save_fame3m.json", ml)):
        for c, v in json.loads((D3 / rf).read_text(encoding="utf-8")).items():
            if "top" in v and v.get("prompts") and rows.get(c, {}).get("answer_form"):
                info[c] = {"top10": [e["neuron"] for e in v["top"][:10]], "prompts": v["prompts"], "form": rows[c]["answer_form"]}
    cs = sorted(info); cand = []
    for i, a in enumerate(cs):
        for b in cs[i + 1:]:
            sh = set(info[a]["top10"]) & set(info[b]["top10"])
            if len(sh) == 1 and not (sh & gen):
                cand.append((a, b, next(iter(sh))))
    random.Random(52).shuffle(cand); cand = cand[:150]
    base_cache = {}

    def base_of(c):
        if c not in base_cache:
            tid = pipe.tok.encode(info[c]["form"]); pairs = [(q, tid) for q in info[c]["prompts"]]
            base_cache[c] = (pairs, pipe.batch_logodds(pairs))
        return base_cache[c]
    res = []
    for a, b, n in cand:
        ln = unit(n); topa = info[a]["top10"]; ra = topa.index(n)
        c1 = min((u for u in topa if u != n and u not in info[b]["top10"]), key=lambda u: abs(topa.index(u) - ra))
        rng = random.Random(f"52|{a}|{b}"); c2 = (ln[0], rng.randrange(pipe.W))
        pa, ba = base_of(a); pb, bb = base_of(b)
        d_nA = stx.mean(drops(pipe, pa, ba, [ln])[0]); d_nB = stx.mean(drops(pipe, pb, bb, [ln])[0])
        d_c1 = stx.mean(drops(pipe, pb, bb, [unit(c1)])[0]); d_c2 = stx.mean(drops(pipe, pb, bb, [c2])[0])
        res.append({"A": a, "B": b, "shared": n, "drop_A_shared": round(d_nA, 3), "drop_B_shared": round(d_nB, 3),
                    "control1": c1, "drop_B_control1": round(d_c1, 3), "control2": f"L{c2[0]}:{c2[1]}", "drop_B_control2": round(d_c2, 3)})
        print(f"P3 {a[:20]:<20} ~ {b[:20]:<20} {n:<10} on A {d_nA:+.2f} | on B {d_nB:+.2f} vs c1 {d_c1:+.2f} c2 {d_c2:+.2f}", flush=True)
    (ROOT / "results/p3_run.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    L.append(f"\nP3 shared neurons: {len(cand)} pairs (from {len(info)} concepts with saved prompts)")
    if res:
        mB = stx.median(r["drop_B_shared"] for r in res); m1 = stx.median(r["drop_B_control1"] for r in res)
        m2 = stx.median(r["drop_B_control2"] for r in res); mA = stx.median(r["drop_A_shared"] for r in res)
        ok = mB >= 0.3 and mB >= 3 * max(m1, 1e-9) and mB >= 3 * max(m2, 1e-9)
        L.append(f"  median drop on B: shared {mB:+.3f}, control 1 (A's non-shared) {m1:+.3f}, control 2 (random) {m2:+.3f}; on A {mA:+.3f}")
        L.append(f"  PREDICTION {'HOLDS' if ok else 'FAILS'} (needs >= 0.3 and >= 3x each control)")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    pipe = FastPipeline(DEFAULT_MODEL, mode="add", reply_only=True); L = []
    if which in ("p1", "both"):
        p1(pipe, L)
    if which in ("p3", "both"):
        p3(pipe, L)
    txt = "\n".join(L)
    (ROOT / "results/p1_p3_summary.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
