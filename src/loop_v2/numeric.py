"""Numeric facts: value neurons, prefix / digit machinery, or magnitude? (Luna; GPU)

James (18 Oct). Qwen writes numbers one digit per token ('1889' -> 1, 8, 8, 9; digit tokens 15..24), so a number is a
sequence of digit choices. Entities: the 90 facts2 entities - landmarks (year completed, height in metres), people (birth
year), works (year created). 3 fixed phrasings per relation; numbers from Luna; kept where Qwen's probability of the full
number is >= MIN_P on >= 2 of 3 phrasings.
Stages:
  entities | build   (as facts.py)
  map       attribution of the full number's log-prob -> top-100 -> each zeroed alone: drop of the full number and of each
            digit position (teacher-forced) -> top-30 with per-position drops
  share     each fact's top-3 zeroed on EVERY numeric fact of the same relation: full and per-position drops, grouped by
            shared prefix (years: first two digits) / same digit at a position
  magnitude each fact's top-3 on its own fact: (a) zeroed, (b) doubled (activation x2 at every position); per position the
            shift in the EXPECTED digit (sum d.p(d) over 0-9, renormalised) and in the correct digit's log-prob.
            Graded / magnitude coding = zeroing and doubling shift the expected digit in OPPOSITE directions.
  position  each fact's top-3 zeroed only at the name tokens vs only at the last prompt position
  summary
Predictions (stated before running): early entity neurons act at the name and late writers at the answer, as for other
facts; late writers are prefix / digit neurons shared across facts (e.g. an '18xx' neuron), not whole-number neurons; some
neurons code magnitude (opposite expected-digit shifts when zeroed vs doubled).
Output: D:/already-clean-data/numeric/.  Usage: python src/loop_v2/numeric.py entities | build | map | share | magnitude | position | summary
"""
import json
import math
import os
import statistics as stx
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "src" / "loop_v2"))
import facts as F1  # noqa: E402
import facts2 as F2  # noqa: E402

OUT = Path(os.environ.get("AC_DATA_DIR", "D:/already-clean-data")) / "numeric"
DIG = list(range(15, 25))                                 # token ids of '0'..'9'
RELS = {
    "landmark": {"year": ["In which year was {X} completed?", "{X} was finished in which year?", "What year was {X} completed?"],
                 "height": ["How tall is {X} in metres?", "What is the height of {X} in metres?", "{X} is how many metres tall?"]},
    "person": {"year": ["In which year was {X} born?", "{X} was born in which year?", "What year was {X} born?"]},
    "work": {"year": ["In which year was {X} created or first published?", "{X} dates from which year?", "What year was {X} made?"]}}
SUFFIX = " Answer with just the number."


def entities():
    import oai
    from concurrent.futures import ThreadPoolExecutor
    OUT.mkdir(parents=True, exist_ok=True)
    E = json.loads((OUT.parent / "facts2" / "entities.json").read_text(encoding="utf-8"))
    SYS = "You are a precise encyclopedia. Reply in JSON only."

    def ask(key):
        kind, e = key.split("|", 1); rels = list(RELS[kind])
        spec = {"year": "<the year as digits, e.g. 1889; for a work the year of creation or first publication / performance; for a "
                        "landmark the year of completion; for a person the birth year>", "height": "<height in metres as a whole number, e.g. 330>"}
        q = f'For "{e}" reply JSON with exactly these keys: ' + ", ".join(f'"{r}": "{spec[r]}"' for r in rels) + ". Most widely cited value; digits only."
        for _ in range(3):
            try:
                return key, oai.chat_json(q, SYS)
            except Exception:  # noqa: BLE001
                pass
        return key, None
    with ThreadPoolExecutor(16) as ex:
        res = dict(ex.map(ask, sorted(E)))
    (OUT / "entities.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(res)} entities; Luna {oai.usage}")


def build():
    import concept_pipeline as cp
    E = json.loads((OUT / "entities.json").read_text(encoding="utf-8")); model, tok = F1.setup()
    F1.SUFFIX = SUFFIX
    facts = []
    for key, ans in E.items():
        if not ans:
            continue
        kind, e = key.split("|", 1)
        for rel, tpls in RELS[kind].items():
            a = "".join(ch for ch in str(ans.get(rel, "")) if ch.isdigit())
            if not a or len(a) > 4:
                continue
            ids = tok.encode(a)
            if not all(i in DIG for i in ids):
                continue
            enc = [F1.encode_with_subject(tok, t.format(X=e), e) for t in tpls]
            lo = F1.logodds_groups(model, tok, [[(p, ids, None) for p, _ in enc]])[0]
            ok = [i for i, l in enumerate(lo) if 1 / (1 + 2.718281828 ** (-l)) >= cp.MIN_P]
            if len(ok) >= 2:
                facts.append({"kind": kind, "entity": e, "relation": rel, "answer": a, "answer_ids": ids,
                              "prompts": [enc[i][0] for i in ok], "subject_pos": [enc[i][1] for i in ok]})
    (OUT / "facts.json").write_text(json.dumps(facts, ensure_ascii=False), encoding="utf-8")
    print(f"{len(facts)} numeric facts kept; {dict(Counter((f['kind'], f['relation']) for f in facts))}")


def digit_pass(model, tok, rows):
    """rows: [(prompt_ids, answer_ids, edits)], edits = list of (layer, neuron, positions or None, factor) (factor 0 = zero,
    2 = double). One pass. Returns per row: list over answer positions of (exact float64 log-odds of the correct digit,
    expected digit)."""
    import torch
    from neurons import DEVICE, mlps
    seqs = [p + a[:-1] for p, a, _ in rows]; T = max(map(len, seqs)); pad = tok.pad_token_id or 0
    x = torch.full((len(rows), T), pad, device=DEVICE); mask = torch.zeros_like(x)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s, device=DEVICE); mask[i, :len(s)] = 1
    by = defaultdict(list)
    for i, (_, _, ed) in enumerate(rows):
        for (l, n, pos, fac) in (ed or []):
            by[l].append((i, n, pos, fac))
    hks = []
    for l, items in by.items():
        def h(mod, a, items=items):
            y = a[0].clone()
            for i, n, pos, fac in items:
                if pos is None:
                    y[i, :, n] = y[i, :, n] * fac
                else:
                    y[i, pos, n] = y[i, pos, n] * fac
            return (y,)
        hks.append(mlps(model)[l].down_proj.register_forward_pre_hook(h))
    try:
        with torch.no_grad():
            hs = model.model(input_ids=x, attention_mask=mask).last_hidden_state
    finally:
        for hk in hks:
            hk.remove()
    dig = torch.tensor(DIG, device=DEVICE); vals = torch.arange(10, device=DEVICE, dtype=torch.float64)
    out = []
    for i, (p, a, _) in enumerate(rows):
        pos = torch.arange(len(p) - 1, len(p) - 1 + len(a), device=DEVICE)
        lp = model.lm_head(hs[i, pos]).double().log_softmax(-1)                        # [len(a), V]
        corr = lp[torch.arange(len(a), device=DEVICE), torch.tensor(a, device=DEVICE)].tolist()
        if max(corr) >= 0.0:
            raise ValueError(f"log P(digit) rounds to 0 even in float64 (log-odds > ~36): {tok.decode(p)[-60:]!r}")
        corr = [c - math.log(-math.expm1(c)) for c in corr]     # exact log-odds: Qwen is > 0.9999 sure of 87% of digits, so log-prob drops saturate (ISSUES #35)
        pd = lp[:, dig].exp(); pd = pd / pd.sum(-1, keepdim=True)
        out.append([(c, float((pd[j] * vals).sum())) for j, c in enumerate(corr)])
    return out


def num_map():
    import torch
    from neurons import DEVICE, mlps
    Fs = json.loads((OUT / "facts.json").read_text(encoding="utf-8")); model, tok = F1.setup()
    L, W = model.config.num_hidden_layers, model.config.intermediate_size
    f = open(OUT / "maps.jsonl", "w", encoding="utf-8")
    for k, fa in enumerate(Fs):
        attr = torch.zeros(L, W, device=DEVICE)
        for p in fa["prompts"]:
            acts = {}

            def keep(l):
                def h(mod, a):
                    a0 = a[0]
                    if not a0.requires_grad:
                        a0 = a0.detach().requires_grad_(True); acts[l] = a0; return (a0,)
                    a0.retain_grad(); acts[l] = a0
                return h
            hks = [mlps(model)[l].down_proj.register_forward_pre_hook(keep(l)) for l in range(L)]
            ids = torch.tensor([p + fa["answer_ids"][:-1]], device=DEVICE); n = len(p)
            try:
                with torch.enable_grad():
                    lp = model(input_ids=ids).logits[0, n - 1:].double().log_softmax(-1)
                    c = lp[torch.arange(len(fa["answer_ids"]), device=DEVICE), torch.tensor(fa["answer_ids"], device=DEVICE)]
                    (c - torch.log(-torch.expm1(c))).sum().backward()          # attribute the digits' summed log-odds (log-prob's gradient vanishes near certainty)
            finally:
                for h in hks:
                    h.remove()
            for l in range(L):
                attr[l] += (acts[l][0].float() * acts[l].grad[0].float()).sum(0) / len(fa["prompts"])   # all positions: later digits are written at the answer positions
        top = [divmod(int(i), W) for i in attr.flatten().topk(100).indices]
        P = len(fa["prompts"]); res = {}
        for c0 in range(0, len(top), 24):
            rows = [(p, fa["answer_ids"], None) for p in fa["prompts"]]
            for (l, nn) in top[c0:c0 + 24]:
                rows += [(p, fa["answer_ids"], [(l, nn, None, 0.0)]) for p in fa["prompts"]]
            v = digit_pass(model, tok, rows); base = v[:P]
            for j, (l, nn) in enumerate(top[c0:c0 + 24]):
                ab = v[P * (j + 1):P * (j + 2)]
                per = [stx.mean(b[q][0] - z[q][0] for b, z in zip(base, ab)) for q in range(len(fa["answer_ids"]))]
                res[f"L{l}:{nn}"] = {"full": round(sum(per), 4), "per_pos": [round(x, 4) for x in per]}
        ranked = sorted(res, key=lambda u: -res[u]["full"])[:30]
        f.write(json.dumps({"kind": fa["kind"], "entity": fa["entity"], "relation": fa["relation"], "answer": fa["answer"],
                            "top": [{"neuron": u, **res[u]} for u in ranked]}, ensure_ascii=False) + "\n"); f.flush()
        if k % 10 == 0:
            print(f"map {k + 1}/{len(Fs)} {fa['entity']} {fa['relation']}={fa['answer']}: {[(u, res[u]['full'], res[u]['per_pos']) for u in ranked[:2]]}", flush=True)


def _units(M, k=3):
    return [(m["entity"], m["relation"], x["neuron"]) for m in M for x in m["top"][:k] if x["full"] > 0.2]


def share():
    Fs = json.loads((OUT / "facts.json").read_text(encoding="utf-8")); M = [json.loads(l) for l in (OUT / "maps.jsonl").read_text(encoding="utf-8").splitlines()]
    model, tok = F1.setup(); units = _units(M)
    import random
    rng = random.Random(181); L_, W_ = model.config.num_hidden_layers, model.config.intermediate_size
    nulls = [f"L{rng.randrange(L_)}:{rng.randrange(W_)}" for _ in range(24)]     # noise floor: random neurons zeroed on every fact
    uniq = sorted({u for _, _, u in units}) + [u for u in nulls if u not in {x for _, _, x in units}]
    f = open(OUT / "share.jsonl", "w", encoding="utf-8")
    for fa in Fs:
        P = len(fa["prompts"]); res = {}
        for c0 in range(0, len(uniq), 24):
            chunk = uniq[c0:c0 + 24]
            rows = [(p, fa["answer_ids"], None) for p in fa["prompts"]]
            for u in chunk:
                l, nn = int(u[1:].split(":")[0]), int(u.split(":")[1])
                rows += [(p, fa["answer_ids"], [(l, nn, None, 0.0)]) for p in fa["prompts"]]
            v = digit_pass(model, tok, rows); base = v[:P]
            for j, u in enumerate(chunk):
                ab = v[P * (j + 1):P * (j + 2)]
                res[u] = [round(stx.mean(b[q][0] - z[q][0] for b, z in zip(base, ab)), 4) for q in range(len(fa["answer_ids"]))]
        f.write(json.dumps({"kind": fa["kind"], "entity": fa["entity"], "relation": fa["relation"], "answer": fa["answer"], "per_pos": res},
                           ensure_ascii=False) + "\n"); f.flush()
    (OUT / "share_units.json").write_text(json.dumps(units, ensure_ascii=False), encoding="utf-8")
    (OUT / "share_nulls.json").write_text(json.dumps(nulls), encoding="utf-8")


def magnitude():
    Fs = {(x["entity"], x["relation"]): x for x in json.loads((OUT / "facts.json").read_text(encoding="utf-8"))}
    M = [json.loads(l) for l in (OUT / "maps.jsonl").read_text(encoding="utf-8").splitlines()]
    model, tok = F1.setup(); f = open(OUT / "magnitude.jsonl", "w", encoding="utf-8")
    for m in M:
        fa = Fs[(m["entity"], m["relation"])]; P = len(fa["prompts"])
        for x in m["top"][:3]:
            l, n = int(x["neuron"][1:].split(":")[0]), int(x["neuron"].split(":")[1])
            rows = [(p, fa["answer_ids"], None) for p in fa["prompts"]] + [(p, fa["answer_ids"], [(l, n, None, 0.0)]) for p in fa["prompts"]] + \
                   [(p, fa["answer_ids"], [(l, n, None, 2.0)]) for p in fa["prompts"]]
            v = digit_pass(model, tok, rows); base, zer, dbl = v[:P], v[P:2 * P], v[2 * P:]
            Q = len(fa["answer_ids"])
            sh = lambda ab, q: stx.mean(z[q][1] - b[q][1] for b, z in zip(base, ab))
            f.write(json.dumps({"entity": m["entity"], "relation": m["relation"], "answer": fa["answer"], "neuron": x["neuron"], "layer": l,
                                "exp_shift_zero": [round(sh(zer, q), 4) for q in range(Q)], "exp_shift_double": [round(sh(dbl, q), 4) for q in range(Q)],
                                "corr_drop_zero": [round(stx.mean(b[q][0] - z[q][0] for b, z in zip(base, zer)), 4) for q in range(Q)],
                                "corr_drop_double": [round(stx.mean(b[q][0] - z[q][0] for b, z in zip(base, dbl)), 4) for q in range(Q)]}, ensure_ascii=False) + "\n")
    f.flush()


def position():
    Fs = {(x["entity"], x["relation"]): x for x in json.loads((OUT / "facts.json").read_text(encoding="utf-8"))}
    M = [json.loads(l) for l in (OUT / "maps.jsonl").read_text(encoding="utf-8").splitlines()]
    model, tok = F1.setup(); f = open(OUT / "position.jsonl", "w", encoding="utf-8")
    for m in M:
        fa = Fs[(m["entity"], m["relation"])]; P = len(fa["prompts"])
        for x in m["top"][:3]:
            l, n = int(x["neuron"][1:].split(":")[0]), int(x["neuron"].split(":")[1])
            rows = [(p, fa["answer_ids"], None) for p in fa["prompts"]] + [(p, fa["answer_ids"], [(l, n, None, 0.0)]) for p in fa["prompts"]] + \
                   [(p, fa["answer_ids"], [(l, n, sp, 0.0)] if sp else []) for p, sp in zip(fa["prompts"], fa["subject_pos"])] + \
                   [(p, fa["answer_ids"], [(l, n, list(range(len(p) - 1, len(p) - 1 + len(fa["answer_ids"]))), 0.0)]) for p in fa["prompts"]]
            v = digit_pass(model, tok, rows); base = v[:P]
            d = lambda j: round(stx.mean(sum(b[q][0] - z[q][0] for q in range(len(fa["answer_ids"]))) for b, z in zip(base, v[P * j:P * (j + 1)])), 4)
            f.write(json.dumps({"entity": m["entity"], "relation": m["relation"], "neuron": x["neuron"], "layer": l,
                                "drop_all": d(1), "drop_subject_only": d(2), "drop_answer_only": d(3)}, ensure_ascii=False) + "\n")
    f.flush()


def summary_v0():
    """First summary (threshold 0.2 - below the random-neuron noise floor; superseded by summary())."""
    md = stx.median; lay = lambda u: int(u[1:].split(":")[0])
    Fs = json.loads((OUT / "facts.json").read_text(encoding="utf-8"))
    M = [json.loads(l) for l in (OUT / "maps.jsonl").read_text(encoding="utf-8").splitlines()]
    S = [json.loads(l) for l in (OUT / "share.jsonl").read_text(encoding="utf-8").splitlines()]
    MG = [json.loads(l) for l in (OUT / "magnitude.jsonl").read_text(encoding="utf-8").splitlines()]
    PZ = [json.loads(l) for l in (OUT / "position.jsonl").read_text(encoding="utf-8").splitlines()]
    L = [f"Numeric facts: {len(Fs)} kept ({dict(Counter((f['kind'], f['relation']) for f in Fs))}); mapped {len(M)}"]
    top5 = [lay(x["neuron"]) for m in M for x in m["top"][:5] if x["full"] > 0.2]
    L.append(f"layers of numeric-fact neurons (top-5, drop > 0.2; n={len(top5)}): median {md(top5)}; < 20: {sum(l < 20 for l in top5) / len(top5):.0%}; >= 28: {sum(l >= 28 for l in top5) / len(top5):.0%}")
    # which digit position each top neuron mainly affects
    pos_main = Counter()
    for m in M:
        for x in m["top"][:3]:
            if x["full"] > 0.2:
                pos_main[(len(x["per_pos"]), max(range(len(x["per_pos"])), key=lambda q: x["per_pos"][q]))] += 1
    L.append(f"main digit position affected by top-3 neurons (number length, position 0 = first digit): {dict(sorted(pos_main.items()))}")
    for b in (0, 20, 28):
        ps = [p for p in PZ if b <= p["layer"] < {0: 20, 20: 28, 28: 36}[b]]
        if ps:
            L.append(f"position, layers {b}+: n={len(ps)} all {md(p['drop_all'] for p in ps):.2f} | name tokens only {md(p['drop_subject_only'] for p in ps):.2f} | "
                     f"answer positions only {md(p['drop_answer_only'] for p in ps):.2f}")
    # sharing by prefix: years only - each unit from fact f, mean drop (full) on other year facts with the same first two digits vs a different prefix
    units = json.loads((OUT / "share_units.json").read_text(encoding="utf-8"))
    nulls = json.loads((OUT / "share_nulls.json").read_text(encoding="utf-8"))
    nv = sorted(abs(sum(s["per_pos"][u])) for s in S for u in nulls if u in s["per_pos"])
    q = lambda f: nv[min(len(nv) - 1, int(f * len(nv)))]
    L.append(f"\nNOISE FLOOR: {len(nulls)} random neurons zeroed on every fact ({len(nv)} drops, summed log-odds): |drop| median {q(0.5):.3f}, "
             f"95th pct {q(0.95):.3f}, 99th {q(0.99):.3f}, max {nv[-1]:.3f}  (threshold used below: 0.2)")
    dS = {(s["entity"], s["relation"]): s for s in S}
    rows = []
    for e, r, u in units:
        if r != "year":
            continue
        a = dS[(e, r)]["answer"]; own = sum(dS[(e, r)]["per_pos"].get(u, [0]))
        same = [sum(s["per_pos"][u]) for k, s in dS.items() if k != (e, r) and s["relation"] == "year" and s["answer"][:2] == a[:2] and len(s["answer"]) == len(a) and u in s["per_pos"]]
        diff = [sum(s["per_pos"][u]) for k, s in dS.items() if s["relation"] == "year" and (s["answer"][:2] != a[:2] or len(s["answer"]) != len(a)) and u in s["per_pos"]]
        if same and diff:
            rows.append((u, e, a, own, stx.mean(same), stx.mean(diff), len(same)))
    pref = [x for x in rows if x[4] > max(0.2, 0.25 * x[3]) and x[4] > 3 * max(x[5], 0.03)]
    gen = [x for x in rows if x[5] > max(0.2, 0.25 * x[3])]
    L.append(f"\nsharing (years; each fact's top-3 zeroed on every year fact): {len(rows)} neurons; PREFIX-shared (same first two digits >3x others): "
             f"{len(pref)}; shared across prefixes (generic year machinery): {len(gen)}; specific to its own fact: {len(rows) - len(pref) - len(gen)}")
    for x in sorted(pref, key=lambda x: -x[4])[:10]:
        L.append(f"   {x[0]:<10} from {x[1]} ({x[2]}): own {x[3]:.2f}, same prefix {x[4]:.2f} (n={x[6]}), other prefixes {x[5]:.2f}")
    # magnitude: opposite expected-digit shifts when zeroed vs doubled, at the neuron's main position
    opp, cnt, ex = 0, 0, []
    for g in MG:
        q = max(range(len(g["corr_drop_zero"])), key=lambda j: g["corr_drop_zero"][j])
        z, d_ = g["exp_shift_zero"][q], g["exp_shift_double"][q]
        if abs(z) > 0.2 or abs(d_) > 0.2:
            cnt += 1
            if z * d_ < 0:
                opp += 1; ex.append((g["neuron"], g["entity"], g["answer"], q, z, d_))
    L.append(f"\nmagnitude test (expected digit at the neuron's main position, |shift| > 0.2 in either condition: {cnt} neurons): opposite shifts when zeroed vs "
             f"doubled in {opp}/{cnt} ({opp / max(1, cnt):.0%})")
    L.append("   examples (neuron, fact, position, shift zeroed, shift doubled): " + "; ".join(f"{n} {e}={a} pos{q} {z:+.2f}/{d:+.2f}" for n, e, a, q, z, d in ex[:10]))
    txt = "\n".join(L)
    (OUT / "summary.txt").write_text(txt + "\n", encoding="utf-8"); print(txt)


THR = 1.0   # summed log-odds; random neurons zeroed on every fact: 99th pct 0.65, max 1.33 (share_nulls)


def _purity(items):
    return Counter(items).most_common(1)[0][1] / len(items)


def summary():
    import random
    md = stx.median; lay = lambda u: int(u[1:].split(":")[0]); rng = random.Random(191)
    Fs = json.loads((OUT / "facts.json").read_text(encoding="utf-8"))
    M = [json.loads(l) for l in (OUT / "maps.jsonl").read_text(encoding="utf-8").splitlines()]
    S = [json.loads(l) for l in (OUT / "share.jsonl").read_text(encoding="utf-8").splitlines()]
    MG = [json.loads(l) for l in (OUT / "magnitude.jsonl").read_text(encoding="utf-8").splitlines()]
    PZ = [json.loads(l) for l in (OUT / "position.jsonl").read_text(encoding="utf-8").splitlines()]
    units = json.loads((OUT / "share_units.json").read_text(encoding="utf-8")); nulls = json.loads((OUT / "share_nulls.json").read_text(encoding="utf-8"))
    nv = sorted(abs(sum(s["per_pos"][u])) for s in S for u in nulls if u in s["per_pos"]); q_ = lambda f: nv[min(len(nv) - 1, int(f * len(nv)))]
    L = [f"Numeric facts: {len(Fs)} kept ({dict(Counter((f['kind'], f['relation']) for f in Fs))}); mapped {len(M)}. Measure: exact float64 log-odds of each digit "
         f"(teacher-forced), summed over the number's digits.",
         f"NOISE FLOOR: {len(nulls)} random neurons zeroed on every fact ({len(nv)} drops): |drop| median {q_(0.5):.3f}, 95th pct {q_(0.95):.3f}, 99th {q_(0.99):.3f}, "
         f"max {nv[-1]:.3f}. Threshold used throughout: {THR}."]
    top5 = [lay(x["neuron"]) for m in M for x in m["top"][:5] if x["full"] > THR]
    L.append(f"layers of numeric-fact neurons (top-5 per fact, drop > {THR}; n={len(top5)}): median {md(top5)}; < 20: {sum(l < 20 for l in top5) / len(top5):.0%}; "
             f"20-27: {sum(20 <= l < 28 for l in top5) / len(top5):.0%}; >= 28: {sum(l >= 28 for l in top5) / len(top5):.0%}")
    for lo_, hi_ in ((0, 20), (20, 28), (28, 36)):
        xs = [x for m in M for x in m["top"][:5] if x["full"] > THR and lo_ <= lay(x["neuron"]) < hi_]
        if xs:
            spread = [sum(v > 0.25 * x["full"] for v in x["per_pos"]) / len(x["per_pos"]) for x in xs]
            L.append(f"   layers {lo_}-{hi_ - 1}: n={len(xs)}; share of the number's digits each neuron hurts (> 25% of its total): median {md(spread):.0%}")
    for lo_, hi_ in ((0, 20), (20, 28), (28, 36)):
        ps = [p for p in PZ if lo_ <= p["layer"] < hi_ and p["drop_all"] > THR]
        if ps:
            L.append(f"position, layers {lo_}-{hi_ - 1} (drop > {THR}): n={len(ps)} all {md(p['drop_all'] for p in ps):.2f} | name tokens only "
                     f"{md(p['drop_subject_only'] for p in ps):.2f} | answer positions only {md(p['drop_answer_only'] for p in ps):.2f}")
    # sharing: each unit's hurt facts (drop > THR), what they have in common, vs a permutation baseline (same number of facts drawn at random)
    dS = {(s["entity"], s["relation"]): s for s in S}; keys = list(dS); src = defaultdict(list)
    for e, r, u in units:
        if sum(dS[(e, r)]["per_pos"].get(u, [0])) > THR:
            src[u].append((e, r))
    yrs = [k for k in keys if k[1] == "year" and len(dS[k]["answer"]) == 4]

    def props(ks, u):
        dig, pd, p3 = [], [], []
        for k in ks:
            pp, a = dS[k]["per_pos"][u], dS[k]["answer"]; qm = max(range(len(pp)), key=lambda j: pp[j])
            dig.append(a[qm]); pd.append((len(a), qm, a[qm]))
            if k in yrs:
                p3.append(a[:3])
        return dig, pd, p3
    rows = []
    for u in src:
        hurt = [k for k in keys if sum(dS[k]["per_pos"][u]) > THR]
        if len(hurt) < 3:
            rows.append({"u": u, "n": len(hurt)}); continue
        dig, pd, p3 = props(hurt, u)
        perm = [props(rng.sample(keys, len(hurt)), u) for _ in range(200)]
        r = {"u": u, "n": len(hurt), "layer": lay(u), "digit": _purity(dig), "digit_null": stx.mean(_purity(p[0]) for p in perm),
             "posdigit": _purity(pd), "posdigit_null": stx.mean(_purity(p[1]) for p in perm), "top_digit": Counter(pd).most_common(1)[0][0],
             "hurt": [(k[0], dS[k]["answer"], round(sum(dS[k]["per_pos"][u]), 1)) for k in sorted(hurt, key=lambda k: -sum(dS[k]["per_pos"][u]))]}
        if len(p3) >= 3:
            r["prefix3"] = _purity(p3); r["prefix3_null"] = stx.mean(_purity(p[2]) for p in perm if len(p[2]) >= 1); r["top_prefix"] = Counter(p3).most_common(1)[0][0]
        rows.append(r)
    big = [r for r in rows if r["n"] >= 3]
    L.append(f"\nSHARING ({len(src)} neurons that hurt their own fact by > {THR}, each zeroed on all {len(keys)} facts):")
    L.append(f"   facts hurt per neuron: {dict(sorted(Counter(min(r['n'], 20) // 5 * 5 if r['n'] >= 5 else r['n'] for r in rows).items()))} (bins: 1,2,3,4, 5-9, 10-14, 15-19, 20+)")
    L.append(f"   neurons hurting >= 3 facts: {len(big)}. What the hurt facts share (observed modal share vs random sets of the same size):")
    L.append(f"     same digit (at each fact's most-hurt position): median {md(r['digit'] for r in big):.0%} vs {md(r['digit_null'] for r in big):.0%}")
    L.append(f"     same digit at the same position:               median {md(r['posdigit'] for r in big):.0%} vs {md(r['posdigit_null'] for r in big):.0%}")
    pb = [r for r in big if "prefix3" in r]
    if pb:
        L.append(f"     same first 3 digits (4-digit years):           median {md(r['prefix3'] for r in pb):.0%} vs {md(r['prefix3_null'] for r in pb):.0%} (n={len(pb)})")
    dn = [r for r in big if r["posdigit"] >= 0.8 and r["posdigit"] > r["posdigit_null"] + 0.3]
    pn = [r for r in pb if r["prefix3"] >= 0.8 and r["prefix3"] > r["prefix3_null"] + 0.3]
    L.append(f"   DIGIT neurons (>= 80% of hurt facts share one digit at one position, > null + 30 points): {len(dn)}/{len(big)}; "
             f"PREFIX neurons (>= 80% share the first 3 digits): {len(pn)}/{len(pb)}")
    for r in sorted(dn, key=lambda r: -r["n"])[:12]:
        n_, q, d = r["top_digit"]
        L.append(f"     {r['u']:<10} digit '{d}' at position {q} of {n_}-digit numbers: {r['n']} facts, purity {r['posdigit']:.0%} | {r['hurt'][:6]}")
    for r in sorted(pn, key=lambda r: -r["n"])[:8]:
        L.append(f"     {r['u']:<10} prefix '{r['top_prefix']}': {r['n']} facts, purity {r['prefix3']:.0%} | {r['hurt'][:6]}")
    wide = sorted([r for r in big if r["n"] >= 20], key=lambda r: -r["n"])
    L.append(f"   broad neurons (>= 20 facts): {len(wide)}: " + "; ".join(f"{r['u']} ({r['n']}, digit purity {r['posdigit']:.0%})" for r in wide[:10]))
    # magnitude: opposite expected-digit shifts when zeroed vs doubled, at the neuron's main position
    opp, cnt, ex = 0, 0, []
    for g in MG:
        q = max(range(len(g["corr_drop_zero"])), key=lambda j: g["corr_drop_zero"][j])
        if sum(g["corr_drop_zero"]) <= THR:
            continue
        z, d_ = g["exp_shift_zero"][q], g["exp_shift_double"][q]
        if abs(z) > 0.2 or abs(d_) > 0.2:
            cnt += 1
            if z * d_ < 0:
                opp += 1; ex.append((g["neuron"], g["entity"], g["answer"], q, z, d_))
    L.append(f"\nMAGNITUDE (top-3 per fact with drop > {THR}; expected digit at the neuron's most-hurt position; |shift| > 0.2 digits in either condition: "
             f"{cnt} neurons): opposite shifts when zeroed vs doubled in {opp}/{cnt} ({opp / max(1, cnt):.0%})")
    L.append("   examples (neuron, fact, position, shift zeroed / doubled): " + "; ".join(f"{n} {e}={a} pos{q} {z:+.2f}/{d:+.2f}" for n, e, a, q, z, d in ex[:10]))
    txt = "\n".join(L)
    (OUT / "summary_rows.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    (OUT / "summary.txt").write_text(txt + "\n", encoding="utf-8"); print(txt)


if __name__ == "__main__":
    {"entities": entities, "build": build, "map": num_map, "share": share, "magnitude": magnitude, "position": position, "summary": summary}[sys.argv[1]]()
