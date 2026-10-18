"""Loop v3 ENRICHMENT - smoke test (LOOP_V3.md). Per concept, run only the stages it lacks; skip what we already have.

James (18 Oct): v3 enriches existing concepts to parity; "V3 should skip answers we already have"; "75 hours feels like a
lot so let's smoke test it first ... and re-assess". This measures real per-stage cost and value on 50 run-1 concepts
(45 fresh + 5 whose E0 was already done by ranking_check.py, to exercise the skip logic). E1 (re-running v2 for
earlier-only concepts) is not in the smoke test.
Stages (inputs reused from run 1: its recall prompts, answer ids, 200 candidates and their drops - never recomputed):
  E0 missed candidates  attribution once; the log-odds-weighted top-200; every candidate not in run 1's 200 zeroed alone
                        (map_core's procedure: all positions, exact log-odds, mean over prompts); which would enter the top 30
  E2 formats            the recall prompts with two other instructions (F2 ' Answer in as few words as possible.',
                        F3 ' Answer in one short sentence.'); Qwen's own greedy answer as the target; activations of run 1's
                        200 candidates (+ E0 entrants) at the answer position in F1 / F2 / F3; run 1's top 30 zeroed in F2 and F3
  E3 nulls              6 random neurons from the layers of the top-6 candidates (outside the 200), zeroed alone in F1
  E4 subject pass       subject_pass.subject_map_one (Luna's 6 questions ABOUT the concept; name-token attribution; top 100
                        zeroed at the name; 6 nulls) - skipped where the ledger already has subject-side evidence
  E7 inhibitors         (MECHANISMS.md H1; added 18 Oct, opt-in: ENRICH_STAGES must name it) run 1's saved attribution array
                        (arrays/<slug>.npz, key 'attr'; recomputed only if missing or of another model's shape); the E7_K (30)
                        most NEGATIVE-attribution neurons - never knocked out by run 1, which took the top 200 positive -
                        each zeroed alone with map_core's procedure (all positions, exact log-odds, mean over recall prompts),
                        plus 4 layer-matched random nulls. A negative drop = the answer went UP when the neuron was removed.
                        Also: change in log-prob of the strongest rival token at the first answer position (d_rival).
Logged per concept: seconds per stage, GPU busy %, peak GPU memory (the 5090 lessons: keep ~25 GB, measure first).
Predictions (stated before running):
  C1 cost: E0 + E2 + E3 + E4 take <= 60 s per concept on the 5090 (vs the ~42 s estimate + E0's ~20 s), GPU busy >= 80%.
  C2 value: E0 changes run 1's top 30 for >= 70% of concepts (ranking check: 44/50); E4 maps >= 70% of concepts
     (>= 3 recalled facts; pilot 9/12); E2 finds >= 20% of each concept's top-30 neurons format-bound (activation in F3 <
     half of F1, or knockout in F3 < half of F1).
  E7 predictions (from the 0.6B study; stated before any 8B run): I1 >= 80% of concepts have >= 1 bottom neuron raising the
     answer by more than run 1's pooled null floor (0.10 log-odds); I2 the median concept's best raise is >= 0.3; I3 a few
     neurons recur as inhibitors in >= 5% of concepts (shared dampeners); I4 removing an inhibitor lowers the strongest rival
     token in >= 70% of cases; I5 E7's own random nulls stay below the floor in >= 95% of cases.
Output: D:/already-clean-data/loop_v3/enrich_smoke/.  Usage: python src/loop_v2/enrich.py sample | questions | run [N] | summary
E7 pilot: ENRICH_NAME=e7_pilot ENRICH_ALL=1 ENRICH_SAMPLE_N=500 python src/loop_v2/enrich.py sample, then
          ENRICH_NAME=e7_pilot ENRICH_STAGES=E7 python src/loop_v2/enrich.py run   (= the first 500 of the full run's fixed order)
(questions needs Luna: run from the main checkout; run needs MODEL_MAPPER_MODELS from the research worktree)
"""
import glob
import json
import os
import random
import statistics as stx
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "src" / "loop_v2"))
DATA = Path(os.environ.get("AC_DATA_DIR", "D:/already-clean-data"))
OUT = DATA / "loop_v3" / os.environ.get("ENRICH_NAME", "enrich_smoke")   # enrich_run1 = the full run
PRIOR = [DATA / "loop_v3" / "enrich_smoke"]                     # earlier enrich dirs: their stages count as done
STAGES = os.environ.get("ENRICH_STAGES", "E0,E2,E3,E4").split(",")
CHEAP = os.environ.get("ENRICH_CHEAP", "1") == "1"                # James 18 Oct: cheapened E2 (no F3 generation / knockouts; batched activations)
RUN1 = DATA / "loop_v2" / "run1" / "results"
V3 = DATA / "loop_v3"
F1S, F2S, F3S = " Answer with just the name.", " Answer in as few words as possible.", " Answer in one short sentence."
ALL_STAGES = ("E0", "E2", "E3", "E4", "E7")
E7_K, E7_NULLS = int(os.environ.get("E7_K", "30")), 4
E7_FLOOR = 0.100                                                  # run 1's pooled null floor (99th pct |drop|), as loop_v3 importer


def run1_attr(r, L, W):
    """Run 1's saved attribution array for record r (its batch folder's arrays/), or (None, path) if missing / other shape."""
    import numpy as np
    name = (r.get("v2") or {}).get("arrays")
    p = Path(r["_file"]).parent / "arrays" / name if name and r.get("_file") else None
    if not p or not p.exists():
        return None, None
    a = np.load(p)["attr"].astype("float32")
    return (a, p) if a.shape == (L, W) else (None, p)


def run1_records():
    for f in sorted(glob.glob(str(RUN1 / "**" / "records.jsonl"), recursive=True)):
        for line in open(f, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if (r.get("row") or {}).get("status") == "ok":
                    r["_file"] = f
                    yield r


def sample():
    OUT.mkdir(parents=True, exist_ok=True)
    done_e0 = [json.loads(l)["concept"] for l in (V3 / "ranking_disputed.jsonl").read_text(encoding="utf-8").splitlines()]
    gate = {json.loads(l)["concept"] for l in (DATA / "loop_v2/gate_repro/records.jsonl").read_text(encoding="utf-8").splitlines()}
    names = [r["concept"] for r in run1_records()]
    fresh = [c for c in names if c not in set(done_e0) | gate]
    rng = random.Random(401)
    if os.environ.get("ENRICH_ALL") == "1":                       # the full run: every run-1 concept, fixed shuffled order
        S = sorted(names); random.Random(501).shuffle(S)
        if os.environ.get("ENRICH_SAMPLE_N"):                      # a pilot = the first N of that same order
            S = S[:int(os.environ["ENRICH_SAMPLE_N"])]
    else:
        S = rng.sample(fresh, 45) + rng.sample(done_e0, 5)
    (OUT / "sample.json").write_text(json.dumps(S, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(S)} concepts" + ("" if os.environ.get("ENRICH_ALL") == "1" else " (45 fresh + 5 with E0 already done)"))


def questions():
    import oai
    import subject_pass as SP
    from concurrent.futures import ThreadPoolExecutor
    S = set(json.loads((OUT / "sample.json").read_text(encoding="utf-8")))
    R = {r["concept"]: r for r in run1_records() if r["concept"] in S}
    p = OUT / "questions.json"; Q = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    def ask(c):
        name = (R[c]["row"].get("answer_form") or c).strip()
        for _ in range(3):
            try:
                return c, {"name": name, "facts": oai.chat_json(SP.question_prompt(c, name), "You write precise quiz questions. Reply in JSON only.")["facts"]}
            except Exception:  # noqa: BLE001
                pass
        return c, None
    todo = [c for c in S if c not in Q]
    with ThreadPoolExecutor(16) as ex:
        for c0 in range(0, len(todo), 400):                         # save as it goes (resumable)
            Q.update(dict(ex.map(ask, todo[c0:c0 + 400])))
            tmp = p.with_suffix(".tmp"); tmp.write_text(json.dumps(Q, indent=1, ensure_ascii=False), encoding="utf-8"); tmp.replace(p)
            print(f"questions {min(c0 + 400, len(todo))}/{len(todo)}; Luna {oai.usage}", flush=True)
    print(f"questions for {sum(1 for v in Q.values() if v)} / {len(S)} concepts; Luna {oai.usage}")


def build_index(con):
    """Everything 'what_we_have' needs, read ONCE (re-reading a growing records file per concept would be quadratic)."""
    from ledger.core import concept_id
    idx = {"E0": {}, "E4": {}, "done": {}}
    rd = V3 / "ranking_disputed.jsonl"
    if rd.exists():
        for l in rd.read_text(encoding="utf-8").splitlines():
            idx["E0"][json.loads(l)["concept"]] = "loop_v3/ranking_disputed.jsonl"
    sp = V3 / "subject" / "subject_maps.jsonl"
    if sp.exists():
        for l in sp.read_text(encoding="utf-8").splitlines():
            idx["E4"][json.loads(l)["concept"]] = "loop_v3/subject/subject_maps.jsonl"
    idx["facts2"] = {r[0] for r in con.execute("SELECT DISTINCT concept_id FROM evidence WHERE experiment_id = 'facts2_entity'").fetchall()}
    idx["cid"] = concept_id
    for d in PRIOR + [OUT]:
        rp = d / "records.jsonl"
        if rp.exists():
            with open(rp, encoding="utf-8") as fh:
                for l in fh:
                    r = json.loads(l)
                    for st in r.get("stages_done", []):
                        idx["done"].setdefault(r["concept"], {})[st] = f"{d.name}/records.jsonl"
    return idx


def have_from(idx, concept):
    h = dict(idx["done"].get(concept, {}))
    if concept in idx["E0"]:
        h.setdefault("E0", idx["E0"][concept])
    if concept in idx["E4"]:
        h.setdefault("E4", idx["E4"][concept])
    if idx["cid"]("thing", concept) in idx["facts2"]:
        h.setdefault("E4", "ledger facts2_entity")
    return h


def what_we_have(concept, con):
    """Stages already covered for this concept: by an earlier enrich record, ranking_check (E0), the subject pilot or
    facts2's entity maps (E4), read from the ledger and the loop_v3 folder."""
    from ledger.core import concept_id
    have = {}
    rd = V3 / "ranking_disputed.jsonl"
    if rd.exists() and any(json.loads(l)["concept"] == concept for l in rd.read_text(encoding="utf-8").splitlines()):
        have["E0"] = "loop_v3/ranking_disputed.jsonl"
    sp = V3 / "subject" / "subject_maps.jsonl"
    if sp.exists() and any(json.loads(l)["concept"] == concept for l in sp.read_text(encoding="utf-8").splitlines()):
        have["E4"] = "loop_v3/subject/subject_maps.jsonl"
    n = con.execute("SELECT count(*) FROM evidence WHERE concept_id = ? AND experiment_id = 'facts2_entity'", [concept_id("thing", concept)]).fetchone()[0]
    if n:
        have["E4"] = "ledger facts2_entity"
    rp = OUT / "records.jsonl"
    if rp.exists():
        for l in rp.read_text(encoding="utf-8").splitlines():
            r = json.loads(l)
            if r["concept"] == concept:
                have.update({s: "enrich_smoke/records.jsonl" for s in r.get("stages_done", [])})
    return have


def run(limit=None):
    import duckdb
    import torch
    from config import DEFAULT_MODEL, model_path
    if not os.path.isdir(model_path(DEFAULT_MODEL)):
        raise SystemExit("set MODEL_MAPPER_MODELS (no local model)")
    sys.path.insert(0, str(ROOT / "src"))
    import subject_pass as SP
    from jlens import GpuBusy
    from neurons import mlps, steer
    from pipeline_fast import FastPipeline
    S = json.loads((OUT / "sample.json").read_text(encoding="utf-8"))[:limit]
    qp = OUT / "questions.json"; Q = json.loads(qp.read_text(encoding="utf-8")) if qp.exists() else {}
    R = {r["concept"]: r for r in run1_records() if r["concept"] in set(S)}
    con = duckdb.connect(str(DATA / "ledger" / "ledger.duckdb"), read_only=True)
    pipe = FastPipeline(DEFAULT_MODEL, mode="add", reply_only=True); m, tok, W = pipe.model, pipe.tok, pipe.W
    gb = GpuBusy(); OUT.mkdir(parents=True, exist_ok=True); f = open(OUT / "records.jsonl", "a", encoding="utf-8"); idx = build_index(con)
    con.close()                                   # read once at start; holding it blocked ledger rebuilds for the whole run (18 Oct)
    pre = lambda u: tuple(map(int, u[1:].split(":")))
    for c in S:
        have = have_from(idx, c)
        for st_ in ALL_STAGES:
            if st_ not in STAGES:
                have.setdefault(st_, "not in this pass (ENRICH_STAGES)")
        if "E4" not in have and not Q.get(c):
            have["E4"] = "no questions yet (next pass)"
        if all(s in have for s in ALL_STAGES):
            print(c, "complete, skipped"); continue
        r = R[c]; rk = r["rank"]; prompts = rk["prompts"]; tid = rk["answer_ids"]; pairs = [(q, tid) for q in prompts]
        run1 = {pre(x["neuron"]): x["single_drop"] for x in rk["candidates_all200"]}
        top30 = [pre(x["neuron"]) for x in rk["top"][:30]]
        rec = {"concept": c, "skipped": have, "stages_done": [], "t": {}, "gpu_busy": {}, "peak_gb": {}}
        if torch.cuda.device_count() > 0:                          # (is_available() can say True with the GPU hidden on Windows)
            torch.cuda.reset_peak_memory_stats()
        base = pipe.batch_logodds(pairs)
        enter = []
        # ---- E0
        if "E0" not in have:
            t = time.time(); gb.take()
            A = pipe.attribute(prompts, tid)
            P = torch.tensor(pipe.batch_probs(pairs), dtype=torch.float64); w = 1.0 / (1.0 - P).clamp_min(1e-12)
            lo = (torch.stack(A["per_prompt"]).double() * w[:, None, None]).sum(0); del A
            t_lo = [divmod(int(i), W) for i in lo.flatten().topk(200).indices]
            new = [u for u in t_lo if u not in run1]; dd = {}
            for (l, n) in new:
                los = pipe.batch_logodds(pairs, steer(m, [(l, n, 0.0)]))
                dd[(l, n)] = [b - x for b, x in zip(base, los)]
            thr30 = sorted(run1.values(), reverse=True)[29]
            enter = sorted([u for u in new if stx.mean(dd[u]) > thr30], key=lambda u: -stx.mean(dd[u]))
            rec["E0"] = {"n_new": len(new), "thr30": thr30, "p": [float(x) for x in P],
                         "new": [{"neuron": f"L{l}:{n}", "mean_drop": stx.mean(dd[(l, n)]), "per_prompt": dd[(l, n)], "logodds_rank": t_lo.index((l, n)) + 1}
                                 for (l, n) in new],
                         "entering_top30": [f"L{l}:{n}" for l, n in enter]}
            rec["stages_done"].append("E0"); rec["t"]["E0"] = round(time.time() - t, 1); rec["gpu_busy"]["E0"] = gb.take()
        # ---- E2
        if "E2" not in have:
            t = time.time(); gb.take()
            fm = {"F1": prompts, "F2": [q.replace(F1S, F2S) for q in prompts], "F3": [q.replace(F1S, F3S) for q in prompts]}
            ok_suffix = all(F1S in q for q in prompts)
            own = {}
            gen = ("F2",) if CHEAP else ("F2", "F3")          # cheapened: F3 knockouts were not like-for-like (whole-sentence target)
            for k in gen:
                ids = [tok.encode(q) for q in fm[k]]; Tm = max(map(len, ids)); pad = tok.pad_token_id or 0
                x = torch.full((len(ids), Tm), pad, device=m.device); am = torch.zeros_like(x)
                for i, s_ in enumerate(ids):
                    x[i, Tm - len(s_):] = torch.tensor(s_); am[i, Tm - len(s_):] = 1
                with torch.no_grad():
                    g = m.generate(input_ids=x, attention_mask=am, max_new_tokens=16, do_sample=False, pad_token_id=pad)[:, Tm:].tolist()
                stop = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
                own[k] = [[t_ for t_ in (s_[:next((j for j, t_ in enumerate(s_) if t_ in stop), len(s_))])] or s_[:1] for s_ in g]
            U = sorted(set(run1) | set(enter)); by_l = {}
            for (l, n) in U:
                by_l.setdefault(l, []).append(n)
            acts = {}
            for k in ("F1", "F2", "F3"):                        # one right-padded pass per format; read each row at its own last token
                ids = [tok.encode(q) for q in fm[k]]; Tm = max(map(len, ids)); pad = tok.pad_token_id or 0
                x = torch.full((len(ids), Tm), pad, device=m.device); am = torch.zeros_like(x)
                for i, s_ in enumerate(ids):
                    x[i, :len(s_)] = torch.tensor(s_); am[i, :len(s_)] = 1
                last = torch.tensor([len(s_) - 1 for s_ in ids], device=m.device); rows_ = torch.arange(len(ids), device=m.device)
                cap = {}
                hks = [mlps(m)[l].down_proj.register_forward_pre_hook(lambda mod, a, l=l: cap.__setitem__(l, a[0][rows_, last][:, by_l[l]].float().cpu())) for l in by_l]
                try:
                    with torch.no_grad():
                        m(input_ids=x, attention_mask=am)
                finally:
                    for h in hks:
                        h.remove()
                acts[k] = {(l, n): cap[l][:, j].tolist() for l, ns in by_l.items() for j, n in enumerate(ns)}
            ko = {}
            for k in gen:
                pk = list(zip(fm[k], own[k])); bk = pipe.batch_logodds(pk)
                for u in top30:
                    los = pipe.batch_logodds(pk, steer(m, [(u[0], u[1], 0.0)]))
                    ko.setdefault(f"L{u[0]}:{u[1]}", {})[k] = [b_ - x_ for b_, x_ in zip(bk, los)]
            for u in top30:
                ko.setdefault(f"L{u[0]}:{u[1]}", {})["F1"] = rk["per_prompt_drop_all200"].get(f"L{u[0]}:{u[1]}")
            rec["E2"] = {"suffix_found": ok_suffix, "own_answers": {k: [tok.decode(a) for a in v] for k, v in own.items()},
                         "acts": {k: {f"L{l}:{n}": v for (l, n), v in a.items()} for k, a in acts.items()}, "knockouts_top30": ko}
            rec["stages_done"].append("E2"); rec["t"]["E2"] = round(time.time() - t, 1); rec["gpu_busy"]["E2"] = gb.take()
        # ---- E3
        if "E3" not in have:
            t = time.time(); gb.take(); rng = random.Random(f"e3|{c}"); nulls = []
            for (l, _) in sorted(run1, key=lambda u: -run1[u])[:6]:
                while True:
                    n = rng.randrange(W)
                    if (l, n) not in run1:
                        nulls.append((l, n)); break
            nd = {}
            for (l, n) in nulls:
                los = pipe.batch_logodds(pairs, steer(m, [(l, n, 0.0)]))
                nd[f"L{l}:{n}"] = [b - x for b, x in zip(base, los)]
            rec["E3"] = {"nulls": nd}
            rec["stages_done"].append("E3"); rec["t"]["E3"] = round(time.time() - t, 1); rec["gpu_busy"]["E3"] = gb.take()
        # ---- E4
        if "E4" not in have and Q.get(c):
            t = time.time(); gb.take()
            rec["E4"] = SP.subject_map_one(m, tok, c, Q[c])
            rec["stages_done"].append("E4"); rec["t"]["E4"] = round(time.time() - t, 1); rec["gpu_busy"]["E4"] = gb.take()
        # ---- E7 inhibitors
        if "E7" not in have:
            rec["E7"] = e7_stage(pipe, m, tok, W, c, r, prompts, tid, pairs, base, run1, steer, mlps)
            rec["stages_done"].append("E7"); rec["t"]["E7"] = rec["E7"].pop("_t"); rec["gpu_busy"]["E7"] = gb.take()
        rec["peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 1) if torch.cuda.device_count() > 0 else None; rec["t"]["total"] = round(sum(v for v in rec["t"].values()), 1)
        f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
        for st_ in rec["stages_done"]:
            idx["done"].setdefault(c, {})[st_] = OUT.name
        print(c, rec["t"], rec["gpu_busy"], rec["peak_gb"], "skipped:", list(have), flush=True)


def e7_stage(pipe, m, tok, W, c, r, prompts, tid, pairs, base, run1, steer, mlps):
    """E7: knock out the most negative-attribution neurons alone (see the docstring). Returns the record's E7 block."""
    import numpy as np
    import torch
    t = time.time(); L = len(mlps(m))
    attr, apath = run1_attr(r, L, W); src = "run1_arrays"
    if attr is None:                                              # no saved array of this model's shape: attribute once
        A = pipe.attribute(prompts, tid); attr = A["attr"].detach().float().cpu().numpy(); del A; src = "recomputed"
    order = np.argsort(attr.reshape(-1), kind="stable")
    bottom = [divmod(int(i), W) for i in order[:E7_K]]
    rng = random.Random(f"e7|{c}"); excl = set(run1) | set(bottom); nulls = []
    for (l, _) in bottom[:E7_NULLS]:
        while True:
            u = (l, rng.randrange(W))
            if u not in excl and u not in nulls:
                nulls.append(u); break
    ids0 = [tok.encode(q) for q in prompts]
    Tm = max(map(len, ids0)); pad = tok.pad_token_id or 0
    x = torch.full((len(ids0), Tm), pad, device=m.device); am = torch.zeros_like(x)
    for i, s_ in enumerate(ids0):
        x[i, :len(s_)] = torch.tensor(s_); am[i, :len(s_)] = 1
    last = torch.tensor([len(s_) - 1 for s_ in ids0], device=m.device); rows_ = torch.arange(len(ids0), device=m.device)

    def first_lp(edits=None):
        """log-probs at the first answer position, optionally with neurons zeroed."""
        ctx = steer(m, edits) if edits else torch.no_grad()
        with torch.no_grad(), ctx:
            lg = m(input_ids=x, attention_mask=am).logits[rows_, last]
        return lg.double().log_softmax(-1).cpu()
    lp0 = first_lp(); lg_ = lp0.clone(); lg_[:, tid[0]] = -1e30; rival = lg_.argmax(-1); ri = torch.arange(len(prompts))

    def ko(u):
        ed = [(u[0], u[1], 0.0)]
        los = pipe.batch_logodds(pairs, steer(m, ed))
        d_r = (first_lp(ed)[ri, rival] - lp0[ri, rival]).tolist()
        return [b - y for b, y in zip(base, los)], d_r
    out_b, out_n = [], {}
    for (l, n) in bottom:
        dd, dr = ko((l, n))
        out_b.append({"neuron": f"L{l}:{n}", "attr": float(attr[l, n]), "mean_drop": stx.mean(dd), "per_prompt": dd,
                      "d_rival": stx.mean(dr), "d_rival_per_prompt": dr})
    for (l, n) in nulls:
        dd, dr = ko((l, n))
        out_n[f"L{l}:{n}"] = {"per_prompt": dd, "d_rival": stx.mean(dr)}
    return {"attr_source": src, "attr_file": str(apath) if apath else None, "k": E7_K, "base": base,
            "rival_tokens": [tok.decode([int(v)]) for v in rival], "bottom": out_b, "nulls": out_n,
            "run1_null_single_drops": (r.get("v2") or {}).get("null_single_drops"), "_t": round(time.time() - t, 1)}


def summary():
    md = stx.median
    R = [json.loads(l) for l in (OUT / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    L = [f"Enrichment smoke test: {len(R)} concept records"]
    for s in ALL_STAGES:
        ts = [r["t"][s] for r in R if s in r["t"]]; gb = [r["gpu_busy"][s] for r in R if r["gpu_busy"].get(s) is not None]
        sk = sum(1 for r in R if s in r["skipped"])
        if ts:
            L.append(f"  {s}: ran {len(ts)}, skipped {sk} (already had it); median {md(ts):.1f} s, total {sum(ts) / 60:.1f} min; GPU busy median {md(gb) if gb else 'n/a'}%")
    tot = [r["t"]["total"] for r in R if not r["skipped"]]
    if tot:
        L.append(f"  per fresh concept: median {md(tot):.1f} s; peak memory max {max((r['peak_gb'] or 0) for r in R)} GB")
        L.append(f"  extrapolated to 5,028 concepts at this rate: {md(tot) * 5028 / 3600:.0f} GPU-hours on the 5090")
    e0 = [r["E0"] for r in R if "E0" in r]
    if e0:
        L.append(f"  E0 value: concepts whose top 30 changes {sum(bool(x['entering_top30']) for x in e0)}/{len(e0)}; entrants median {md(len(x['entering_top30']) for x in e0)}")
    e4 = [r["E4"] for r in R if "E4" in r]
    if e4:
        L.append(f"  E4 value: mapped (>= 3 recalled facts) {sum(bool(x.get('top')) for x in e4)}/{len(e4)}; top-5 drop median "
                 f"{md([y['mean_drop'] for x in e4 if x.get('top') for y in x['top'][:5]] or [0]):.2f} vs |null| "
                 f"{md([abs(y['mean_drop']) for x in e4 if x.get('nulls') for y in x['nulls']] or [0]):.3f}")
    e2 = [r["E2"] for r in R if "E2" in r]
    if e2:
        fb = []
        for x in e2:
            for u, k in x["knockouts_top30"].items():
                if k.get("F1") and k.get("F3"):
                    d1, d3 = stx.mean(k["F1"]), stx.mean(k["F3"])
                    if d1 > 0.1:
                        fb.append(d3 < 0.5 * d1)
        L.append(f"  E2 value: top-30 neurons format-bound (F3 knockout < half of F1): {sum(fb)}/{len(fb)} ({sum(fb) / max(1, len(fb)):.0%})")
    e7 = [r["E7"] for r in R if "E7" in r]
    if e7:
        from collections import Counter
        best = [-min(x["mean_drop"] for x in e["bottom"]) for e in e7]
        hits = [x for e in e7 for x in e["bottom"] if x["mean_drop"] < -E7_FLOOR]
        share = Counter(x["neuron"] for x in hits)
        nul = [abs(stx.mean(v["per_prompt"])) for e in e7 for v in e["nulls"].values()]
        L.append(f"  E7 inhibitors: {len(e7)} concepts; I1 with >= 1 raise beyond {E7_FLOOR}: {sum(1 for b in best if b > E7_FLOOR)}/{len(e7)}; "
                 f"I2 median best raise {md(best):.2f}; raises {len(hits)}/{sum(len(e['bottom']) for e in e7)} bottom neurons")
        L.append(f"  I3 shared (>= 5% of concepts): {[(u, k) for u, k in share.most_common(15) if k >= 0.05 * len(e7)]}")
        L.append(f"  I4 inhibitors lowering the rival: {sum(1 for x in hits if x['d_rival'] < 0)}/{len(hits)}")
        L.append(f"  I5 E7 nulls below the floor: {sum(1 for v in nul if v < E7_FLOOR)}/{len(nul)} (median |drop| {md(nul) if nul else float('nan'):.3f}); "
                 f"attribution from run 1 arrays {sum(1 for e in e7 if e['attr_source'] == 'run1_arrays')}/{len(e7)}")
    txt = "\n".join(L); (OUT / "summary.txt").write_text(txt + "\n", encoding="utf-8"); print(txt)


if __name__ == "__main__":
    a = sys.argv[1]
    {"sample": sample, "questions": questions, "run": lambda: run(int(sys.argv[2]) if len(sys.argv) > 2 else None), "summary": summary}[a]()
